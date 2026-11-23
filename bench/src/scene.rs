//! Deterministic benchmark scene: a fixed terrain, fixed lights and a list of
//! fixed camera views. Nothing here depends on time or randomness, so two runs
//! of the same code produce the same pixels.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use plumesplat::prelude::*;

/// A named camera position. Add new views here; names become file names.
pub struct View {
    pub name: &'static str,
    pub eye: Vec3,
    pub target: Vec3,
}

pub const VIEWS: &[View] = &[
    // Same framing as examples/basic.rs.
    View { name: "overview", eye: Vec3::new(15.0, 10.0, 15.0), target: Vec3::ZERO },
    // Ground fills the screen: worst case for per-pixel shader cost.
    View { name: "closeup", eye: Vec3::new(2.0, 3.2, 2.0), target: Vec3::new(-1.0, 1.0, -1.0) },
    // Low angle toward the horizon: mips, filtering and tiling at distance.
    View { name: "grazing", eye: Vec3::new(20.0, 3.5, 20.0), target: Vec3::new(-10.0, 0.0, -10.0) },
    // Straight down: makes texture repetition and tile seams obvious.
    View { name: "topdown", eye: Vec3::new(0.0, 14.0, 0.01), target: Vec3::ZERO },
];

#[derive(Component)]
pub struct BenchCamera;

#[derive(Component)]
pub struct BenchTerrain;

pub const TERRAIN_SIZE: f32 = 48.0;
const TERRAIN_RES: u32 = 256;
const MAX_HEIGHT: f32 = 4.0;

pub fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    asset_server: Res<AssetServer>,
) {
    let tex = |name: &str| asset_server.load(format!("textures/raw/{name}"));

    let pending = SplatMaterialBuilder::new()
        .add_layer(MaterialLayer::new(tex("Grass004_1K-PNG_Color.png")).with_normal(tex("Grass004_1K-PNG_NormalGL.png")))
        .add_layer(MaterialLayer::new(tex("Ground054_1K-PNG_Color.png")).with_normal(tex("Ground054_1K-PNG_NormalGL.png")))
        .add_layer(MaterialLayer::new(tex("Rock051_1K-PNG_Color.png")).with_normal(tex("Rock051_1K-PNG_NormalGL.png")))
        .add_layer(MaterialLayer::new(tex("snow_diff.png")))
        .with_uv_scale(0.5)
        .with_triplanar_sharpness(4.0)
        .with_blend_offset(0.1)
        .with_blend_exponent(2.0)
        .build();

    commands.spawn((Mesh3d(meshes.add(terrain_mesh())), pending, BenchTerrain));

    commands.spawn((
        DirectionalLight { illuminance: 5000.0, shadow_maps_enabled: true, ..default() },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.6, 0.4, 0.0)),
    ));
    commands.spawn(AmbientLight { color: Color::WHITE, brightness: 200.0, ..default() });

    let v = &VIEWS[0];
    commands.spawn((
        Camera3d::default(),
        Msaa::Sample4,
        Transform::from_translation(v.eye).looking_at(v.target, Vec3::Y),
        BenchCamera,
    ));
}

fn height(x: f32, z: f32) -> f32 {
    let h1 = (x * 0.3).sin() * (z * 0.3).cos();
    let h2 = (x * 0.7 + 1.0).cos() * (z * 0.7 - 0.5).sin() * 0.5;
    let h3 = (x * 1.5).sin() * (z * 1.5).sin() * 0.25;
    ((h1 + h2 + h3) * 0.5 + 0.5) * MAX_HEIGHT
}

fn band(v: f32, start: f32, end: f32, fade: f32) -> f32 {
    let s = |e0: f32, e1: f32, x: f32| {
        let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    s(start - fade, start + fade, v) * (1.0 - s(end - fade, end + fade, v))
}

fn terrain_mesh() -> Mesh {
    let n = TERRAIN_RES;
    let step = TERRAIN_SIZE / (n - 1) as f32;
    let half = TERRAIN_SIZE / 2.0;
    let count = (n * n) as usize;

    let mut positions = Vec::with_capacity(count);
    let mut normals = Vec::with_capacity(count);
    let mut verts = Vec::with_capacity(count);

    for z in 0..n {
        for x in 0..n {
            let (px, pz) = (-half + x as f32 * step, -half + z as f32 * step);
            let h = height(px, pz);
            let e = step * 0.5;
            let normal = Vec3::new(
                height(px - e, pz) - height(px + e, pz),
                2.0 * e,
                height(px, pz - e) - height(px, pz + e),
            )
            .normalize();

            let nh = h / MAX_HEIGHT;
            let slope = ((1.0 - normal.y) * 3.0).clamp(0.0, 1.0);
            let grass = band(nh, 0.0, 0.4, 0.2) * (1.0 - slope * 0.7);
            let dirt = band(nh, 0.2, 0.6, 0.2);
            let rock = band(nh, 0.4, 0.9, 0.25) + slope * 0.5;
            let snow = band(nh, 0.65, 1.0, 0.2) * (1.0 - slope * 0.5);

            positions.push([px, h, pz]);
            normals.push(normal.to_array());
            verts.push(MaterialVertex::blend4([0, 1, 2, 3], [grass, dirt, rock, snow]));
        }
    }

    let mut indices = Vec::with_capacity(((n - 1) * (n - 1) * 6) as usize);
    for z in 0..n - 1 {
        for x in 0..n - 1 {
            let (tl, tr) = (z * n + x, z * n + x + 1);
            let (bl, br) = (tl + n, tr + n);
            if (x + z) % 2 == 0 {
                indices.extend([tl, bl, tr, tr, bl, br]);
            } else {
                indices.extend([tl, bl, br, tl, br, tr]);
            }
        }
    }

    let (mat_indices, mat_weights) = encode_material_data(&verts);
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(ATTRIBUTE_MATERIAL_INDICES, mat_indices)
        .with_inserted_attribute(ATTRIBUTE_MATERIAL_WEIGHTS, mat_weights)
        .with_inserted_indices(Indices::U32(indices))
}
