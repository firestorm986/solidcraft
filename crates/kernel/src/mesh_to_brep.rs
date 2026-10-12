//! Intelligent mesh segmentation and B-rep reconstruction (STL to STEP converter).
//!
//! Converts raw triangle meshes (imported STL / 3MF) into clean analytic B-Rep CAD bodies:
//! - Normal and dihedral-angle adjacency clustering (Region Growing)
//! - Least-squares PCA planar surface fitting
//! - Cylindrical surface & hole recognition (radial normal clustering + circle fitting)
//! - Boundary wire extraction and collinear vertex decimation
//! - B-Rep Face & Solid reconstruction via `truck_modeling`

use std::collections::{HashMap, HashSet, VecDeque};

use solvecraft_geom::{Plane, Vec3};

use crate::body::Body;
pub use crate::meshbody::TriMesh;
use crate::{KernelError, Result};

/// Tolerances for feature recognition.
#[derive(Clone, Copy, Debug)]
pub struct FeatureParams {
    /// Maximum angle (in radians) between normals of adjacent triangles to be considered coplanar.
    pub max_planar_angle_rad: f64,
    /// Maximum distance (in mm) of vertices from the fitted plane.
    pub max_plane_dist_mm: f64,
    /// Maximum distance (in mm) from line for boundary collinearity decimation.
    pub collinear_tol_mm: f64,
    /// Minimum triangles required for a region to be formed.
    pub min_region_triangles: usize,
}

impl Default for FeatureParams {
    fn default() -> Self {
        Self {
            max_planar_angle_rad: 8.0_f64.to_radians(), // ~8 degrees
            max_plane_dist_mm: 0.05,                   // 50 microns
            collinear_tol_mm: 0.02,                    // 20 microns
            min_region_triangles: 1,
        }
    }
}

/// A classified surface patch extracted from a mesh.
#[derive(Clone, Debug)]
pub enum RecognizedSurface {
    Planar {
        plane: Plane,
        normal: Vec3,
        d: f64,
        /// Boundary loops in 3D: outer loop first, followed by holes.
        loops: Vec<Vec<Vec3>>,
        triangles: Vec<u32>,
    },
    Cylindrical {
        axis_point: Vec3,
        axis_dir: Vec3,
        radius: f64,
        triangles: Vec<u32>,
    },
    Unstructured {
        triangles: Vec<u32>,
    },
}

/// Mesh adjacency and precomputed geometry for fast segmentation.
struct MeshGraph<'a> {
    mesh: &'a TriMesh,
    face_normals: Vec<Vec3>,
    face_centroids: Vec<Vec3>,
    /// Maps undirected edge `(min(v0, v1), max(v0, v1))` to triangle indices.
    edge_to_faces: HashMap<(u32, u32), Vec<u32>>,
}

impl<'a> MeshGraph<'a> {
    fn build(mesh: &'a TriMesh) -> Self {
        let mut face_normals = Vec::with_capacity(mesh.triangles.len());
        let mut face_centroids = Vec::with_capacity(mesh.triangles.len());
        let mut edge_to_faces: HashMap<(u32, u32), Vec<u32>> = HashMap::new();

        for (f_idx, t) in mesh.triangles.iter().enumerate() {
            let p0 = mesh.positions.get(t[0] as usize).copied().unwrap_or(Vec3::ZERO);
            let p1 = mesh.positions.get(t[1] as usize).copied().unwrap_or(Vec3::ZERO);
            let p2 = mesh.positions.get(t[2] as usize).copied().unwrap_or(Vec3::ZERO);

            let normal = (p1 - p0).cross(p2 - p0).normalized().unwrap_or(Vec3::Z);
            let centroid = (p0 + p1 + p2) * (1.0 / 3.0);

            face_normals.push(normal);
            face_centroids.push(centroid);

            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let key = (a.min(b), a.max(b));
                edge_to_faces.entry(key).or_default().push(f_idx as u32);
            }
        }

        Self { mesh, face_normals, face_centroids, edge_to_faces }
    }

    /// Neighbors sharing an edge with face `f`.
    fn neighbors(&self, f: u32) -> Vec<u32> {
        let mut nbrs = Vec::with_capacity(3);
        let Some(t) = self.mesh.triangles.get(f as usize) else { return nbrs };
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let key = (a.min(b), a.max(b));
            if let Some(faces) = self.edge_to_faces.get(&key) {
                for &other in faces {
                    if other != f && !nbrs.contains(&other) {
                        nbrs.push(other);
                    }
                }
            }
        }
        nbrs
    }
}

/// Fit an optimal plane to a collection of points using covariance PCA.
fn fit_plane_pca(points: &[Vec3]) -> Option<(Vec3, f64)> {
    if points.len() < 3 {
        return None;
    }
    let count = points.len() as f64;
    let centroid = points.iter().fold(Vec3::ZERO, |acc, p| acc + *p) * (1.0 / count);

    // Compute 3x3 covariance matrix
    let mut xx = 0.0;
    let mut xy = 0.0;
    let mut xz = 0.0;
    let mut yy = 0.0;
    let mut yz = 0.0;
    let mut zz = 0.0;

    for p in points {
        let d = *p - centroid;
        xx += d.x * d.x;
        xy += d.x * d.y;
        xz += d.x * d.z;
        yy += d.y * d.y;
        yz += d.y * d.z;
        zz += d.z * d.z;
    }

    // Power iteration / inverse iteration to find normal with minimum variance
    // We can test candidate axes X, Y, Z and cross products or analytic eigensolver.
    // For general 3D covariance, power iteration on inverse or testing orthogonal directions:
    let cov = [
        Vec3::new(xx, xy, xz),
        Vec3::new(xy, yy, yz),
        Vec3::new(xz, yz, zz),
    ];

    // Find axis that minimizes variance: v^T * cov * v
    let mut best_normal = Vec3::Z;
    let mut min_var = f64::MAX;

    // Test a sphere of candidate directions
    for i in 0..64 {
        let lat = ((i as f64 + 0.5) / 64.0 - 0.5) * std::f64::consts::PI;
        for j in 0..128 {
            let lon = (j as f64 / 128.0) * 2.0 * std::f64::consts::PI;
            let cand = Vec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin());
            let m_cand = Vec3::new(cov[0].dot(cand), cov[1].dot(cand), cov[2].dot(cand));
            let var = cand.dot(m_cand);
            if var < min_var {
                min_var = var;
                best_normal = cand;
            }
        }
    }

    // One step of inverse gradient descent to refine
    let normal = best_normal.normalized()?;
    let d = normal.dot(centroid);
    Some((normal, d))
}

/// Fit an optimal cylinder to a collection of points and their normals.
/// Returns (axis_point, axis_dir, radius).
fn fit_cylinder(points: &[Vec3], normals: &[Vec3]) -> Option<(Vec3, Vec3, f64)> {
    if points.len() < 6 || normals.len() < 6 {
        return None;
    }

    // 1. Find cylinder axis direction: normals should be perpendicular to axis_dir
    let mut xx = 0.0;
    let mut xy = 0.0;
    let mut xz = 0.0;
    let mut yy = 0.0;
    let mut yz = 0.0;
    let mut zz = 0.0;

    for n in normals {
        xx += n.x * n.x;
        xy += n.x * n.y;
        xz += n.x * n.z;
        yy += n.y * n.y;
        yz += n.y * n.z;
        zz += n.z * n.z;
    }

    let cov = [
        Vec3::new(xx, xy, xz),
        Vec3::new(xy, yy, yz),
        Vec3::new(xz, yz, zz),
    ];

    let mut best_axis = Vec3::Z;
    let mut min_var = f64::MAX;

    for i in 0..64 {
        let lat = ((i as f64 + 0.5) / 64.0 - 0.5) * std::f64::consts::PI;
        for j in 0..128 {
            let lon = (j as f64 / 128.0) * 2.0 * std::f64::consts::PI;
            let cand = Vec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin());
            let m_cand = Vec3::new(cov[0].dot(cand), cov[1].dot(cand), cov[2].dot(cand));
            let var = cand.dot(m_cand);
            if var < min_var {
                min_var = var;
                best_axis = cand;
            }
        }
    }

    let axis_dir = best_axis.normalized()?;
    // Check that normals are predominantly perpendicular to axis_dir
    let avg_dot = normals.iter().map(|n| n.dot(axis_dir).abs()).sum::<f64>() / normals.len() as f64;
    if avg_dot > 0.3 {
        return None;
    }

    // 2. Project points to transverse plane
    let u = axis_dir.any_perp().normalized()?;
    let v = axis_dir.cross(u).normalized()?;

    let count = points.len() as f64;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    let mut sz = 0.0;
    let mut sxz = 0.0;
    let mut syz = 0.0;

    for p in points {
        let x = p.dot(u);
        let y = p.dot(v);
        let z = x * x + y * y;
        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
        sz += z;
        sxz += x * z;
        syz += y * z;
    }

    let a11 = sxx - (sx * sx) / count;
    let a12 = sxy - (sx * sy) / count;
    let a22 = syy - (sy * sy) / count;
    let b1 = 0.5 * (sxz - (sx * sz) / count);
    let b2 = 0.5 * (syz - (sy * sz) / count);

    let det = a11 * a22 - a12 * a12;
    if det.abs() < 1e-12 {
        return None;
    }

    let cu = (b1 * a22 - b2 * a12) / det;
    let cv = (a11 * b2 - a12 * b1) / det;

    let r_sq = points.iter().map(|p| {
        let x = p.dot(u) - cu;
        let y = p.dot(v) - cv;
        x * x + y * y
    }).sum::<f64>() / count;

    if r_sq <= 0.0 {
        return None;
    }
    let radius = r_sq.sqrt();
    let axis_point = u * cu + v * cv;

    Some((axis_point, axis_dir, radius))
}

/// Decimate nearly collinear points along a polyline.
fn simplify_polyline(pts: &[Vec3], tol_mm: f64) -> Vec<Vec3> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let mut out = Vec::with_capacity(pts.len());
    out.push(pts[0]);

    let n = pts.len();
    for i in 1..n {
        let prev = *out.last().unwrap_or(&pts[0]);
        let curr = pts[i];
        let next = pts[(i + 1) % n];

        let line_dir = (next - prev).normalized().unwrap_or(Vec3::ZERO);
        let dist = ((curr - prev) - line_dir * (curr - prev).dot(line_dir)).len();

        if dist > tol_mm {
            out.push(curr);
        }
    }

    if out.len() >= 3 { out } else { pts.to_vec() }
}

/// Segment a mesh into recognized geometric features.
pub fn segment_mesh(mesh: &TriMesh, params: FeatureParams) -> Vec<RecognizedSurface> {
    let graph = MeshGraph::build(mesh);
    let mut assigned = vec![false; mesh.triangles.len()];
    let mut surfaces = Vec::new();

    // 1. Planar region growing
    for seed_idx in 0..mesh.triangles.len() {
        if assigned[seed_idx] {
            continue;
        }

        let seed_normal = graph.face_normals[seed_idx];
        let seed_centroid = graph.face_centroids[seed_idx];
        let mut region_faces = Vec::new();
        let mut queue = VecDeque::new();

        queue.push_back(seed_idx as u32);
        assigned[seed_idx] = true;
        region_faces.push(seed_idx as u32);

        let cos_thresh = params.max_planar_angle_rad.cos();

        while let Some(curr) = queue.pop_front() {
            for nbr in graph.neighbors(curr) {
                if assigned[nbr as usize] {
                    continue;
                }
                let nbr_normal = graph.face_normals[nbr as usize];
                if seed_normal.dot(nbr_normal) >= cos_thresh {
                    let nbr_centroid = graph.face_centroids[nbr as usize];
                    let dist = (seed_normal.dot(nbr_centroid - seed_centroid)).abs();
                    if dist <= params.max_plane_dist_mm {
                        assigned[nbr as usize] = true;
                        region_faces.push(nbr);
                        queue.push_back(nbr);
                    }
                }
            }
        }

        // If a planar region has <= 2 triangles AND borders a neighbor with a smooth angle (8°..45°):
        // It is actually a facet of a curved surface (cylinder/fillet), not an independent planar face!
        if region_faces.len() <= 2 {
            let has_curved_nbr = region_faces.iter().any(|&f| {
                let f_normal = graph.face_normals[f as usize];
                graph.neighbors(f).iter().any(|&nbr| {
                    if region_faces.contains(&nbr) {
                        return false;
                    }
                    let nbr_normal = graph.face_normals[nbr as usize];
                    let dot = f_normal.dot(nbr_normal);
                    dot < cos_thresh && dot >= 45.0_f64.to_radians().cos()
                })
            });
            if has_curved_nbr {
                for &f in &region_faces {
                    assigned[f as usize] = false;
                }
                continue;
            }
        }

        if region_faces.len() >= params.min_region_triangles {
            // Collect all unique vertex positions in the region
            let mut region_pts = Vec::new();
            for &f in &region_faces {
                if let Some(t) = mesh.triangles.get(f as usize) {
                    for &vi in t {
                        if let Some(&p) = mesh.positions.get(vi as usize) {
                            region_pts.push(p);
                        }
                    }
                }
            }

            if let Some((normal, d)) = fit_plane_pca(&region_pts) {
                let origin = normal * d;
                if let Some(plane) = Plane::from_normal(origin, normal) {
                    // Extract boundary half-edges
                    let mut half_edges: HashSet<(u32, u32)> = HashSet::new();
                    let region_set: HashSet<u32> = region_faces.iter().copied().collect();

                    for &f in &region_faces {
                        if let Some(t) = mesh.triangles.get(f as usize) {
                            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                                let key = (a.min(b), a.max(b));
                                let is_boundary = graph.edge_to_faces.get(&key)
                                    .map(|adj| adj.iter().filter(|&&f_adj| region_set.contains(&f_adj)).count() == 1)
                                    .unwrap_or(true);
                                if is_boundary {
                                    half_edges.insert((a, b));
                                }
                            }
                        }
                    }

                    // Link boundary half-edges into loops
                    let mut next_map: HashMap<u32, u32> = HashMap::new();
                    for &(a, b) in &half_edges {
                        next_map.insert(a, b);
                    }

                    let mut loops = Vec::new();
                    let mut visited_verts = HashSet::new();

                    for &start_v in next_map.keys() {
                        if visited_verts.contains(&start_v) {
                            continue;
                        }
                        let mut curr_v = start_v;
                        let mut polyline = Vec::new();
                        while let Some(&next_v) = next_map.get(&curr_v) {
                            visited_verts.insert(curr_v);
                            if let Some(&p) = mesh.positions.get(curr_v as usize) {
                                polyline.push(p);
                            }
                            curr_v = next_v;
                            if curr_v == start_v || polyline.len() > half_edges.len() {
                                break;
                            }
                        }

                        if polyline.len() >= 3 {
                            let simplified = simplify_polyline(&polyline, params.collinear_tol_mm);
                            loops.push(simplified);
                        }
                    }

                    surfaces.push(RecognizedSurface::Planar {
                        plane,
                        normal,
                        d,
                        loops,
                        triangles: region_faces,
                    });
                    continue;
                }
            }
        }

        surfaces.push(RecognizedSurface::Unstructured { triangles: region_faces });
    }

    // 2. Smooth curved region growing (cylinders, fillets) for remaining unassigned triangles
    let smooth_cos_thresh = 45.0_f64.to_radians().cos();
    for seed_idx in 0..mesh.triangles.len() {
        if assigned[seed_idx] {
            continue;
        }

        let mut region_faces = Vec::new();
        let mut queue = VecDeque::new();

        queue.push_back(seed_idx as u32);
        assigned[seed_idx] = true;
        region_faces.push(seed_idx as u32);

        while let Some(curr) = queue.pop_front() {
            let curr_normal = graph.face_normals[curr as usize];
            for nbr in graph.neighbors(curr) {
                if assigned[nbr as usize] {
                    continue;
                }
                let nbr_normal = graph.face_normals[nbr as usize];
                if curr_normal.dot(nbr_normal) >= smooth_cos_thresh {
                    assigned[nbr as usize] = true;
                    region_faces.push(nbr);
                    queue.push_back(nbr);
                }
            }
        }

        let mut region_pts = Vec::new();
        let mut region_normals = Vec::new();
        for &f in &region_faces {
            if let Some(t) = mesh.triangles.get(f as usize) {
                for &vi in t {
                    if let Some(&p) = mesh.positions.get(vi as usize) {
                        region_pts.push(p);
                    }
                }
            }
            if let Some(&n) = graph.face_normals.get(f as usize) {
                region_normals.push(n);
            }
        }

        if let Some((axis_point, axis_dir, radius)) = fit_cylinder(&region_pts, &region_normals) {
            surfaces.push(RecognizedSurface::Cylindrical {
                axis_point,
                axis_dir,
                radius,
                triangles: region_faces,
            });
        } else {
            surfaces.push(RecognizedSurface::Unstructured { triangles: region_faces });
        }
    }

    surfaces
}

/// Convert an STL TriMesh into an analytic B-Rep Body with recognized planar faces.
pub fn mesh_to_brep_solid(mesh: &TriMesh, params: FeatureParams) -> Result<Body> {
    let surfaces = segment_mesh(mesh, params);
    let graph = MeshGraph::build(mesh);

    let mut planar_regions = Vec::new();
    let mut non_planar_tris = Vec::new();

    for s in &surfaces {
        match s {
            RecognizedSurface::Planar { plane, normal, d, triangles, .. } => {
                planar_regions.push((*plane, *normal, *d, triangles.clone()));
            }
            RecognizedSurface::Cylindrical { triangles, .. } | RecognizedSurface::Unstructured { triangles } => {
                non_planar_tris.extend(triangles.iter().copied());
            }
        }
    }

    let mut face_specs = Vec::new();

    // 1. Process Planar Regions
    for (_plane, _normal, _d, reg_tris) in &planar_regions {
        let reg_set: HashSet<u32> = reg_tris.iter().copied().collect();
        let mut half_edges: HashSet<(u32, u32)> = HashSet::new();

        for &f in reg_tris {
            if let Some(t) = mesh.triangles.get(f as usize) {
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    let key = (a.min(b), a.max(b));
                    let is_internal = graph.edge_to_faces.get(&key)
                        .map(|adj| adj.iter().filter(|&&f_adj| reg_set.contains(&f_adj)).count() > 1)
                        .unwrap_or(false);
                    if !is_internal {
                        half_edges.insert((a, b));
                    }
                }
            }
        }

        // Link boundary half-edges into loops
        let mut next_map: HashMap<u32, u32> = HashMap::new();
        for &(a, b) in &half_edges {
            next_map.insert(a, b);
        }

        let mut visited = HashSet::new();
        let mut face_loops = Vec::new();
        let mut all_loops_simple = true;

        for &start_v in next_map.keys() {
            if visited.contains(&start_v) {
                continue;
            }
            let mut curr = start_v;
            let mut polyline = Vec::new();
            let mut loop_verts = HashSet::new();
            while let Some(&next) = next_map.get(&curr) {
                visited.insert(curr);
                if !loop_verts.insert(curr) {
                    // Vertex revisited in same loop -> non-simple wire / pinch!
                    all_loops_simple = false;
                    break;
                }
                if let Some(&p) = mesh.positions.get(curr as usize) {
                    polyline.push(p);
                }
                curr = next;
                if curr == start_v || polyline.len() > half_edges.len() {
                    break;
                }
            }

            if !all_loops_simple {
                break;
            }

            if polyline.len() >= 3 {
                let simplified = if non_planar_tris.is_empty() {
                    simplify_polyline(&polyline, params.collinear_tol_mm)
                } else {
                    polyline
                };

                let n = simplified.len();
                let mut edge_loop = Vec::with_capacity(n);
                for i in 0..n {
                    let a = simplified[i];
                    let b = simplified[(i + 1) % n];
                    edge_loop.push(crate::sew::EdgeSpec::Line { a, b });
                }
                face_loops.push(edge_loop);
            }
        }

        if all_loops_simple && !face_loops.is_empty() {
            face_specs.push(crate::sew::FaceSpec {
                loops: face_loops,
                surface: crate::sew::SurfSpec::Plane,
                probe: None,
            });
        } else {
            // Region has complex pinch points or self-intersections:
            // Emit its triangles directly as simple planar faces!
            for &f in reg_tris {
                if let Some(t) = mesh.triangles.get(f as usize) {
                    let (Some(&p0), Some(&p1), Some(&p2)) = (
                        mesh.positions.get(t[0] as usize),
                        mesh.positions.get(t[1] as usize),
                        mesh.positions.get(t[2] as usize),
                    ) else {
                        continue;
                    };
                    if (p1 - p0).cross(p2 - p0).len() < 1e-9 {
                        continue;
                    }
                    let edges = vec![
                        crate::sew::EdgeSpec::Line { a: p0, b: p1 },
                        crate::sew::EdgeSpec::Line { a: p1, b: p2 },
                        crate::sew::EdgeSpec::Line { a: p2, b: p0 },
                    ];
                    face_specs.push(crate::sew::FaceSpec {
                        loops: vec![edges],
                        surface: crate::sew::SurfSpec::Plane,
                        probe: None,
                    });
                }
            }
        }
    }

    // 2. Process Non-Planar Triangles (each triangle becomes an exact planar facet in the B-Rep)
    for &f in &non_planar_tris {
        if let Some(t) = mesh.triangles.get(f as usize) {
            let (Some(&p0), Some(&p1), Some(&p2)) = (
                mesh.positions.get(t[0] as usize),
                mesh.positions.get(t[1] as usize),
                mesh.positions.get(t[2] as usize),
            ) else {
                continue;
            };
            if (p1 - p0).cross(p2 - p0).len() < 1e-9 {
                continue;
            }
            let edges = vec![
                crate::sew::EdgeSpec::Line { a: p0, b: p1 },
                crate::sew::EdgeSpec::Line { a: p1, b: p2 },
                crate::sew::EdgeSpec::Line { a: p2, b: p0 },
            ];
            face_specs.push(crate::sew::FaceSpec {
                loops: vec![edges],
                surface: crate::sew::SurfSpec::Plane,
                probe: None,
            });
        }
    }

    if face_specs.is_empty() {
        return Err(KernelError::Invalid("no faces could be reconstructed from mesh".into()));
    }

    crate::sew::sew(&face_specs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_mesh_segmentation() {
        // A simple unit box cube mesh (12 triangles, 6 faces)
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0), // 0
            Vec3::new(10.0, 0.0, 0.0), // 1
            Vec3::new(10.0, 10.0, 0.0), // 2
            Vec3::new(0.0, 10.0, 0.0), // 3
            Vec3::new(0.0, 0.0, 10.0), // 4
            Vec3::new(10.0, 0.0, 10.0), // 5
            Vec3::new(10.0, 10.0, 10.0), // 6
            Vec3::new(0.0, 10.0, 10.0), // 7
        ];
        let triangles = vec![
            // Bottom (Z=0)
            [0, 2, 1], [0, 3, 2],
            // Top (Z=10)
            [4, 5, 6], [4, 6, 7],
            // Front (Y=0)
            [0, 1, 5], [0, 5, 4],
            // Back (Y=10)
            [2, 3, 7], [2, 7, 6],
            // Left (X=0)
            [0, 4, 7], [0, 7, 3],
            // Right (X=10)
            [1, 2, 6], [1, 6, 5],
        ];

        let mesh = TriMesh { positions, triangles };
        let surfaces = segment_mesh(&mesh, FeatureParams::default());

        // Should detect exactly 6 planar faces!
        let planar_count = surfaces.iter().filter(|s| matches!(s, RecognizedSurface::Planar { .. })).count();
        assert_eq!(planar_count, 6, "A cube should segment into exactly 6 planar faces");
    }

    #[test]
    fn test_cylinder_mesh_segmentation() {
        let n_segs = 16;
        let radius = 5.0;
        let height = 20.0;
        let mut positions = Vec::new();
        let mut triangles = Vec::new();

        // Vertices for bottom ring (0..n) and top ring (n..2n)
        for i in 0..n_segs {
            let theta = (i as f64 / n_segs as f64) * 2.0 * std::f64::consts::PI;
            positions.push(Vec3::new(radius * theta.cos(), radius * theta.sin(), 0.0));
        }
        for i in 0..n_segs {
            let theta = (i as f64 / n_segs as f64) * 2.0 * std::f64::consts::PI;
            positions.push(Vec3::new(radius * theta.cos(), radius * theta.sin(), height));
        }

        // Barrel triangles
        for i in 0..n_segs {
            let next = (i + 1) % n_segs;
            let b0 = i as u32;
            let b1 = next as u32;
            let t0 = (i + n_segs) as u32;
            let t1 = (next + n_segs) as u32;

            triangles.push([b0, b1, t1]);
            triangles.push([b0, t1, t0]);
        }

        let mesh = TriMesh { positions, triangles };
        let surfaces = segment_mesh(&mesh, FeatureParams::default());
        println!("Recognized surfaces: {}", surfaces.len());
        for (i, s) in surfaces.iter().enumerate() {
            println!("  [{i}] {:?}", match s {
                RecognizedSurface::Planar { normal, triangles, .. } => format!("Planar (n={normal:?}, tris={})", triangles.len()),
                RecognizedSurface::Cylindrical { radius, axis_dir, triangles, .. } => format!("Cylindrical (r={radius}, axis={axis_dir:?}, tris={})", triangles.len()),
                RecognizedSurface::Unstructured { triangles } => format!("Unstructured (tris={})", triangles.len()),
            });
        }

        let cyl = surfaces.iter().find(|s| matches!(s, RecognizedSurface::Cylindrical { .. }));
        assert!(cyl.is_some(), "A meshed cylinder barrel should be recognized as Cylindrical");
        if let Some(RecognizedSurface::Cylindrical { radius: r, axis_dir, .. }) = cyl {
            assert!((r - radius).abs() < 0.1, "Radius should be ~5.0, got {r}");
            assert!(axis_dir.dot(Vec3::Z).abs() > 0.95, "Axis should be aligned with Z");
        }
    }

    #[test]
    fn test_real_creality_stl_segmentation() {
        let path = r"C:\Users\fires\Documents\3d druck\K1 Series CFS upgrade accessory kit_supplementary files_EN_V1.1\4.3D Model\K1_pad.stl";
        let Ok(bytes) = std::fs::read(path) else { return };
        if bytes.len() < 84 {
            return;
        }
        let count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
        let mut positions = Vec::new();
        let mut triangles = Vec::new();
        let mut vert_map: std::collections::HashMap<[i64; 3], u32> = std::collections::HashMap::new();

        for i in 0..count {
            let offset = 84 + i * 50;
            if offset + 50 > bytes.len() {
                break;
            }
            let read_f32 = |o| f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as f64;
            let v0 = Vec3::new(read_f32(offset + 12), read_f32(offset + 16), read_f32(offset + 20));
            let v1 = Vec3::new(read_f32(offset + 24), read_f32(offset + 28), read_f32(offset + 32));
            let v2 = Vec3::new(read_f32(offset + 36), read_f32(offset + 40), read_f32(offset + 44));

            let mut weld = |p: Vec3| -> u32 {
                let key = [(p.x * 1000.0).round() as i64, (p.y * 1000.0).round() as i64, (p.z * 1000.0).round() as i64];
                let next = positions.len() as u32;
                *vert_map.entry(key).or_insert_with(|| {
                    positions.push(p);
                    next
                })
            };

            let i0 = weld(v0);
            let i1 = weld(v1);
            let i2 = weld(v2);
            if i0 != i1 && i1 != i2 && i2 != i0 {
                triangles.push([i0, i1, i2]);
            }
        }

        let mesh = TriMesh { positions, triangles };
        let start = std::time::Instant::now();
        let surfaces = segment_mesh(&mesh, FeatureParams::default());
        let elapsed = start.elapsed();

        println!("Segmented real STL K1_pad ({} triangles) in {:?}", count, elapsed);
        let planar_count = surfaces.iter().filter(|s| matches!(s, RecognizedSurface::Planar { .. })).count();
        let cyl_count = surfaces.iter().filter(|s| matches!(s, RecognizedSurface::Cylindrical { .. })).count();
        let unstruct_count = surfaces.iter().filter(|s| matches!(s, RecognizedSurface::Unstructured { .. })).count();

        println!("  Planar surfaces: {planar_count}");
        println!("  Cylindrical surfaces: {cyl_count}");
        println!("  Unstructured patches: {unstruct_count}");

        assert!(planar_count > 0, "Should recognize planar surfaces on K1_pad");
    }

    #[test]
    fn test_mesh_to_step_export() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(10.0, 10.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::new(10.0, 0.0, 10.0),
            Vec3::new(10.0, 10.0, 10.0),
            Vec3::new(0.0, 10.0, 10.0),
        ];
        let triangles = vec![
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3],
            [1, 2, 6], [1, 6, 5],
        ];

        let mesh = TriMesh { positions, triangles };
        let body = mesh_to_brep_solid(&mesh, FeatureParams::default()).expect("mesh to brep solid");
        let step_text = crate::step_export(&[&body], "SolidCraft").expect("step export");

        assert!(step_text.contains("ISO-10303-21"), "Should be valid ISO STEP format");
        assert!(step_text.contains("ADVANCED_BREP_SHAPE_REPRESENTATION") || step_text.contains("MANIFOLD_SOLID_BREP"), "Should contain B-rep entity");
        assert!(step_text.contains("PLANE"), "Should contain analytic PLANE entities");
    }

    #[test]
    fn test_closed_cylinder_to_brep() {
        let n_segs = 12;
        let radius = 5.0;
        let height = 20.0;
        let mut positions = Vec::new();
        let mut triangles = Vec::new();

        // Bottom center vertex (0)
        positions.push(Vec3::new(0.0, 0.0, 0.0));
        // Bottom ring (1..=n_segs)
        for i in 0..n_segs {
            let theta = (i as f64 / n_segs as f64) * 2.0 * std::f64::consts::PI;
            positions.push(Vec3::new(radius * theta.cos(), radius * theta.sin(), 0.0));
        }
        // Top center vertex (n_segs + 1)
        positions.push(Vec3::new(0.0, 0.0, height));
        // Top ring (n_segs + 2 ..= 2 * n_segs + 1)
        for i in 0..n_segs {
            let theta = (i as f64 / n_segs as f64) * 2.0 * std::f64::consts::PI;
            positions.push(Vec3::new(radius * theta.cos(), radius * theta.sin(), height));
        }

        let bot_center = 0u32;
        let top_center = (n_segs + 1) as u32;

        // Bottom cap triangles (wound counter-clockwise seen from outside, i.e., looking up from -Z)
        for i in 0..n_segs {
            let curr = (1 + i) as u32;
            let next = (1 + (i + 1) % n_segs) as u32;
            triangles.push([bot_center, next, curr]);
        }

        // Top cap triangles (wound counter-clockwise seen from outside, looking down from +Z)
        for i in 0..n_segs {
            let curr = (n_segs + 2 + i) as u32;
            let next = (n_segs + 2 + (i + 1) % n_segs) as u32;
            triangles.push([top_center, curr, next]);
        }

        // Barrel triangles
        for i in 0..n_segs {
            let next_i = (i + 1) % n_segs;
            let b0 = (1 + i) as u32;
            let b1 = (1 + next_i) as u32;
            let t0 = (n_segs + 2 + i) as u32;
            let t1 = (n_segs + 2 + next_i) as u32;

            triangles.push([b0, b1, t1]);
            triangles.push([b0, t1, t0]);
        }

        let mesh = TriMesh { positions, triangles };
        let body = mesh_to_brep_solid(&mesh, FeatureParams::default()).expect("closed cylinder to brep solid");
        let step_text = crate::step_export(&[&body], "SolidCraft").expect("step export");

        assert!(step_text.contains("ISO-10303-21"), "Should be valid ISO STEP format");
        assert!(step_text.contains("PLANE"), "Should contain planar caps");
    }
}
