// PlumeSplat Material Extension Shader
// Extends StandardMaterial to add texture array splatting with triplanar mapping

#import bevy_pbr::{
    mesh_functions,
    forward_io::{Vertex, VertexOutput, FragmentOutput},
    view_transformations::position_world_to_clip,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
}

// Extension bindings - using bindless/shader_defs compatible syntax
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var albedo_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var albedo_array_sampler: sampler;

struct PlumeSplatSettings {
    uv_scale: f32,
    triplanar_sharpness: f32,
    height_blend_sharpness: f32,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var<uniform> settings: PlumeSplatSettings;

// Extended vertex output with our custom material data
struct PlumeSplatVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // Custom attributes for material splatting
    @location(10) @interpolate(flat) material_indices: u32,
    @location(11) material_weights: u32,
}

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
fn triplanar_weights(world_normal: vec3<f32>, sharpness: f32) -> vec3<f32> {
    var weights = abs(world_normal);
    weights = pow(weights, vec3<f32>(sharpness));
    let sum = weights.x + weights.y + weights.z;
    return weights / max(sum, 0.0001);
}

// Sample texture array using triplanar projection
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
    var uv_x = scaled_pos.zy;
    if world_normal.x < 0.0 {
        uv_x.x = -uv_x.x;
    }

    var uv_y = scaled_pos.xz;
    if world_normal.y < 0.0 {
        uv_y.y = -uv_y.y;
    }

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
fn vertex(
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(10) material_indices: u32,
    @location(11) material_weights: u32,
) -> PlumeSplatVertexOutput {
    var out: PlumeSplatVertexOutput;

    let world_from_local = mesh_functions::get_world_from_local(instance_index);
    let world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(position, 1.0));

    out.position = position_world_to_clip(world_position.xyz);
    out.world_position = world_position;
    out.world_normal = mesh_functions::mesh_normal_local_to_world(normal, instance_index);
    out.uv = uv;
    out.material_indices = material_indices;
    out.material_weights = material_weights;

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

    // Construct VertexOutput for PBR functions
    var vertex_out: VertexOutput;
    vertex_out.position = in.position;
    vertex_out.world_position = in.world_position;
    vertex_out.world_normal = in.world_normal;
    vertex_out.uv = in.uv;

    // Generate PBR input from standard material
    var pbr_input = pbr_input_from_standard_material(vertex_out, is_front);

    // Override the base color with our splatted texture
    pbr_input.material.base_color = final_albedo;

    // Apply alpha discard if needed
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

    // Apply full PBR lighting
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);

    return out;
}
