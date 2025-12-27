// PlumeSplat Material Extension Shader
// Extends StandardMaterial to add texture array splatting with triplanar mapping

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    mesh_functions,
}
#import bevy_render::instance_index::get_instance_index

// Custom vertex input with our material attributes
struct PlumeSplatVertexInput {
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(10) material_indices: u32,
    @location(11) material_weights: u32,
}

// Custom vertex output that includes our material data
struct PlumeSplatVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(10) @interpolate(flat) material_indices: u32,
    @location(11) material_weights: u32,
}

// Extension bindings (high binding numbers to avoid conflicts)
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var albedo_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var albedo_array_sampler: sampler;

struct PlumeSplatSettings {
    uv_scale: f32,
    triplanar_sharpness: f32,
    height_blend_sharpness: f32,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var<uniform> settings: PlumeSplatSettings;

// Unpack 4 u8 values from a u32
fn unpack_u8x4(packed: u32) -> vec4<u32> {
    return vec4<u32>(
        packed & 0xFFu,
        (packed >> 8u) & 0xFFu,
        (packed >> 16u) & 0xFFu,
        (packed >> 24u) & 0xFFu
    );
}

// Compute triplanar blend weights from world normal
// Uses squared weights for sharper transitions on steep slopes
fn triplanar_weights(world_normal: vec3<f32>, sharpness: f32) -> vec3<f32> {
    var weights = abs(world_normal);
    // Apply sharpness - higher values give sharper transitions
    weights = pow(weights, vec3<f32>(sharpness));
    // Normalize
    let sum = weights.x + weights.y + weights.z;
    return weights / max(sum, 0.0001);
}

// Sample texture array using triplanar projection
// Properly handles axis flipping to prevent texture mirroring/stretching
fn sample_triplanar(
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    layer: i32,
    uv_scale: f32,
    sharpness: f32
) -> vec4<f32> {
    let weights = triplanar_weights(world_normal, sharpness);
    let scaled_pos = world_pos * uv_scale;

    // Compute UVs with proper axis orientation to prevent mirroring
    // X-axis projection (YZ plane) - flip based on normal direction
    var uv_x = scaled_pos.zy;
    if world_normal.x < 0.0 {
        uv_x.x = -uv_x.x;
    }

    // Y-axis projection (XZ plane) - most common for terrain tops
    var uv_y = scaled_pos.xz;
    if world_normal.y < 0.0 {
        uv_y.y = -uv_y.y;
    }

    // Z-axis projection (XY plane) - flip based on normal direction
    var uv_z = scaled_pos.xy;
    if world_normal.z < 0.0 {
        uv_z.x = -uv_z.x;
    }

    let sample_x = textureSample(albedo_array, albedo_array_sampler, uv_x, layer);
    let sample_y = textureSample(albedo_array, albedo_array_sampler, uv_y, layer);
    let sample_z = textureSample(albedo_array, albedo_array_sampler, uv_z, layer);

    return sample_x * weights.x + sample_y * weights.y + sample_z * weights.z;
}

// Height-based blending for better transitions
fn height_blend(heights: vec4<f32>, weights: vec4<f32>, sharpness: f32) -> vec4<f32> {
    if sharpness <= 0.0 {
        let sum = weights.x + weights.y + weights.z + weights.w;
        if sum > 0.0 {
            return weights / sum;
        }
        return vec4<f32>(1.0, 0.0, 0.0, 0.0);
    }

    let height_weighted = heights + weights;
    let max_h = max(max(height_weighted.x, height_weighted.y),
                    max(height_weighted.z, height_weighted.w));

    var blend = max(height_weighted - max_h + sharpness, vec4<f32>(0.0));
    let sum = blend.x + blend.y + blend.z + blend.w;

    if sum > 0.0 {
        return blend / sum;
    }
    return weights;
}

@vertex
fn vertex(in: PlumeSplatVertexInput) -> PlumeSplatVertexOutput {
    var out: PlumeSplatVertexOutput;

    let instance_index = get_instance_index(in.instance_index);
    var world_from_local = mesh_functions::get_world_from_local(instance_index);

    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(in.position, 1.0));
    out.position = mesh_functions::mesh_position_world_to_clip(out.world_position);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(in.normal, instance_index);
    out.uv = in.uv;

    // Pass through our custom attributes
    out.material_indices = in.material_indices;
    out.material_weights = in.material_weights;

    return out;
}

@fragment
fn fragment(
    in: PlumeSplatVertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    // Get world position and normal for triplanar
    let world_pos = in.world_position.xyz;
    var world_normal = normalize(in.world_normal);

    // Handle back faces
    if !is_front {
        world_normal = -world_normal;
    }

    // Unpack material data
    let indices = unpack_u8x4(in.material_indices);
    let idx0 = i32(indices.x);
    let idx1 = i32(indices.y);
    let idx2 = i32(indices.z);
    let idx3 = i32(indices.w);

    let raw_weights = unpack_u8x4(in.material_weights);
    var weights = vec4<f32>(raw_weights) / 255.0;

    // Sample each material layer with triplanar mapping
    let uv_scale = settings.uv_scale;
    let sharpness = settings.triplanar_sharpness;

    let albedo0 = sample_triplanar(world_pos, world_normal, idx0, uv_scale, sharpness);
    let albedo1 = sample_triplanar(world_pos, world_normal, idx1, uv_scale, sharpness);
    let albedo2 = sample_triplanar(world_pos, world_normal, idx2, uv_scale, sharpness);
    let albedo3 = sample_triplanar(world_pos, world_normal, idx3, uv_scale, sharpness);

    // Optional height-based blending
    if settings.height_blend_sharpness > 0.0 {
        let heights = vec4<f32>(albedo0.a, albedo1.a, albedo2.a, albedo3.a);
        weights = height_blend(heights, weights, settings.height_blend_sharpness);
    } else {
        let sum = weights.x + weights.y + weights.z + weights.w;
        if sum > 0.0 {
            weights = weights / sum;
        }
    }

    // Blend final albedo
    let final_albedo = albedo0 * weights.x + albedo1 * weights.y +
                       albedo2 * weights.z + albedo3 * weights.w;

    // Simple directional lighting
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let ndotl = max(dot(world_normal, light_dir), 0.0);
    let ambient = 0.3;
    let diffuse = ndotl * 0.7;

    var out: FragmentOutput;
    out.color = vec4<f32>(final_albedo.rgb * (ambient + diffuse), 1.0);

    return out;
}
