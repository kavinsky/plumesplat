//! Basic example demonstrating PlumeSplat terrain splatting.
//!
//! This example creates a simple terrain mesh with multiple materials
//! blended based on height and slope.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use plumesplat::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PlumeSplatPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (convert_texture_to_array, rotate_camera))
        .run();
}

/// Marker for the terrain entity so we can update its material
#[derive(Component)]
struct TerrainMarker;

/// Resource to track the texture array handle
#[derive(Resource)]
struct TextureArrayHandle {
    handle: Handle<Image>,
    converted: bool,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<PlumeSplatMaterial>>,
    asset_server: Res<AssetServer>,
) {
    // Create a terrain-like mesh
    let terrain_mesh = create_terrain_mesh(64, 64, 10.0, 2.0);
    let mesh_handle = meshes.add(terrain_mesh);

    // Load the stacked texture (4 layers stacked vertically: grass, dirt, rock, snow)
    let albedo_handle: Handle<Image> = asset_server.load("textures/terrain_albedo_array.png");

    // Store handle for conversion
    commands.insert_resource(TextureArrayHandle {
        handle: albedo_handle.clone(),
        converted: false,
    });

    // Create material with the texture array
    let material = PlumeSplatMaterial::new()
        .with_albedo(albedo_handle)
        .with_uv_scale(0.5)
        .with_triplanar_sharpness(4.0)
        .with_roughness(0.8);

    let material_handle = materials.add(material);

    // Spawn terrain
    commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(material_handle),
        Transform::from_xyz(0.0, 0.0, 0.0),
        TerrainMarker,
    ));

    // Spawn light
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.5, 0.0)),
    ));

    // Spawn camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(15.0, 10.0, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
        CameraController { angle: 0.0 },
    ));

    info!("PlumeSplat basic example running!");
    info!("Terrain uses 4 materials: grass, dirt, rock, snow");
    info!("Materials blend based on height and slope.");
}

/// Convert the loaded stacked 2D texture into a 2D array texture
fn convert_texture_to_array(
    mut texture_handle: ResMut<TextureArrayHandle>,
    mut images: ResMut<Assets<Image>>,
) {
    if texture_handle.converted {
        return;
    }

    if let Some(image) = images.get_mut(&texture_handle.handle) {
        // Convert stacked 2D image to 2D array with 4 layers
        const NUM_LAYERS: u32 = 4;
        image.reinterpret_stacked_2d_as_array(NUM_LAYERS);
        info!("Converted texture to 2D array with {} layers", NUM_LAYERS);
        texture_handle.converted = true;
    }
}

#[derive(Component)]
struct CameraController {
    angle: f32,
}

fn rotate_camera(time: Res<Time>, mut query: Query<(&mut Transform, &mut CameraController)>) {
    for (mut transform, mut controller) in query.iter_mut() {
        controller.angle += time.delta_secs() * 0.2;
        let radius = 20.0;
        let height = 12.0;
        transform.translation = Vec3::new(
            controller.angle.cos() * radius,
            height,
            controller.angle.sin() * radius,
        );
        transform.look_at(Vec3::ZERO, Vec3::Y);
    }
}

/// Create a terrain mesh with PlumeSplat vertex attributes.
fn create_terrain_mesh(width: u32, height: u32, size: f32, max_height: f32) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut material_indices: Vec<u32> = Vec::new();
    let mut material_weights: Vec<u32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let half_size = size / 2.0;
    let step_x = size / (width - 1) as f32;
    let step_z = size / (height - 1) as f32;

    // Generate vertices
    for z in 0..height {
        for x in 0..width {
            let px = -half_size + x as f32 * step_x;
            let pz = -half_size + z as f32 * step_z;

            // Simple heightmap using sine waves
            let h = heightmap(px, pz, max_height);

            positions.push([px, h, pz]);
            uvs.push([
                x as f32 / (width - 1) as f32,
                z as f32 / (height - 1) as f32,
            ]);

            // Calculate normal (will be recalculated more accurately below)
            normals.push([0.0, 1.0, 0.0]);

            // Assign materials based on height and slope
            // Material 0: Grass (low areas)
            // Material 1: Dirt (transition)
            // Material 2: Rock (steep areas)
            // Material 3: Snow (high areas)
            let material_data = compute_material_blend(h, max_height);
            material_indices.push(material_data.packed_indices());
            material_weights.push(material_data.packed_weights());
        }
    }

    // Calculate normals properly
    for z in 0..height {
        for x in 0..width {
            let idx = (z * width + x) as usize;
            let px = positions[idx][0];
            let pz = positions[idx][2];

            // Sample heights around this point for normal calculation
            let eps = step_x * 0.5;
            let h_left = heightmap(px - eps, pz, max_height);
            let h_right = heightmap(px + eps, pz, max_height);
            let h_back = heightmap(px, pz - eps, max_height);
            let h_front = heightmap(px, pz + eps, max_height);

            let normal = Vec3::new(h_left - h_right, 2.0 * eps, h_back - h_front).normalize();
            normals[idx] = normal.to_array();

            // Recalculate material blend with slope information
            let slope = 1.0 - normal.y; // 0 = flat, 1 = vertical
            let h = positions[idx][1];
            let material_data = compute_material_blend_with_slope(h, max_height, slope);
            material_indices[idx] = material_data.packed_indices();
            material_weights[idx] = material_data.packed_weights();
        }
    }

    // Generate indices for triangles
    for z in 0..(height - 1) {
        for x in 0..(width - 1) {
            let top_left = z * width + x;
            let top_right = top_left + 1;
            let bottom_left = (z + 1) * width + x;
            let bottom_right = bottom_left + 1;

            // First triangle
            indices.push(top_left);
            indices.push(bottom_left);
            indices.push(top_right);

            // Second triangle
            indices.push(top_right);
            indices.push(bottom_left);
            indices.push(bottom_right);
        }
    }

    // Build the mesh
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(ATTRIBUTE_MATERIAL_INDICES, material_indices);
    mesh.insert_attribute(ATTRIBUTE_MATERIAL_WEIGHTS, material_weights);
    mesh.insert_indices(Indices::U32(indices));

    mesh
}

/// Simple heightmap function using multiple sine waves.
fn heightmap(x: f32, z: f32, max_height: f32) -> f32 {
    let freq1 = 0.3;
    let freq2 = 0.7;
    let freq3 = 1.5;

    let h1 = (x * freq1).sin() * (z * freq1).cos();
    let h2 = (x * freq2 + 1.0).cos() * (z * freq2 - 0.5).sin() * 0.5;
    let h3 = (x * freq3).sin() * (z * freq3).sin() * 0.25;

    ((h1 + h2 + h3) * 0.5 + 0.5) * max_height
}

/// Compute material blend based on height alone.
fn compute_material_blend(height: f32, max_height: f32) -> MaterialVertex {
    let normalized_height = height / max_height;

    // Define height bands for each material
    // 0: Grass (0.0 - 0.3)
    // 1: Dirt (0.2 - 0.5)
    // 2: Rock (0.4 - 0.8)
    // 3: Snow (0.7 - 1.0)

    let grass = smooth_band(normalized_height, 0.0, 0.35, 0.1);
    let dirt = smooth_band(normalized_height, 0.25, 0.55, 0.1);
    let rock = smooth_band(normalized_height, 0.45, 0.85, 0.15);
    let snow = smooth_band(normalized_height, 0.7, 1.0, 0.1);

    MaterialVertex::blend4([0, 1, 2, 3], [grass, dirt, rock, snow])
}

/// Compute material blend based on height and slope.
fn compute_material_blend_with_slope(height: f32, max_height: f32, slope: f32) -> MaterialVertex {
    let normalized_height = height / max_height;

    // Base height-based weights
    let grass = smooth_band(normalized_height, 0.0, 0.35, 0.1);
    let dirt = smooth_band(normalized_height, 0.25, 0.55, 0.1);
    let rock = smooth_band(normalized_height, 0.45, 0.85, 0.15);
    let snow = smooth_band(normalized_height, 0.7, 1.0, 0.1);

    // Steep slopes get more rock
    let slope_factor = (slope * 3.0).clamp(0.0, 1.0);
    let rock = rock + slope_factor * 0.5;
    let grass = grass * (1.0 - slope_factor * 0.7);
    let snow = snow * (1.0 - slope_factor * 0.5);

    MaterialVertex::blend4([0, 1, 2, 3], [grass, dirt, rock, snow])
}

/// Smooth band function for height-based material blending.
fn smooth_band(value: f32, start: f32, end: f32, fade: f32) -> f32 {
    let fade_in = smoothstep(start - fade, start + fade, value);
    let fade_out = 1.0 - smoothstep(end - fade, end + fade, value);
    fade_in * fade_out
}

/// Hermite smoothstep interpolation.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
