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
    // Weights unpacked to vec4<f32> for proper interpolation
    @location(11) material_weights: vec4<f32>,
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

// Hash function for stochastic tiling
fn hash2(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(q) * 43758.5453);
}

// Stochastic tiling - samples texture at 3 random offsets and blends based on proximity
// This breaks up visible tiling patterns
fn sample_stochastic(
    tex: texture_2d_array<f32>,
    tex_sampler: sampler,
    uv: vec2<f32>,
    layer: i32,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>
) -> vec4<f32> {
    // Get integer tile coordinate
    let tile = floor(uv);
    let f = fract(uv);
    
    // Sample from 3 nearby tiles with random offsets
    var color = vec4<f32>(0.0);
    var total_weight = 0.0;
    
    for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
            let offset = vec2<f32>(f32(i), f32(j));
            let tile_pos = tile + offset;
            
            // Random offset for this tile (0-1 range)
            let rand = hash2(tile_pos);
            
            // Offset UV within tile
            let offset_uv = uv + rand;
            
            // Weight based on distance from tile center (smooth falloff)
            let dist = f - offset + 0.5;
            let weight = max(0.0, 1.0 - dot(dist, dist) * 2.0);
            
            if weight > 0.001 {
                color += textureSampleGrad(tex, tex_sampler, offset_uv, layer, ddx_uv, ddy_uv) * weight;
                total_weight += weight;
            }
        }
    }
    
    return color / max(total_weight, 0.001);
}

// Compute triplanar blend weights from world normal
// Uses squared weights for smoother falloff, then applies sharpness
fn triplanar_weights(world_normal: vec3<f32>, sharpness: f32) -> vec3<f32> {
    // Square the absolute normal components for initial weights
    var weights = world_normal * world_normal;
    
    // Apply additional sharpness power if needed
    if sharpness > 1.0 {
        weights = pow(weights, vec3<f32>(sharpness * 0.5));
    }
    
    // Normalize
    let sum = weights.x + weights.y + weights.z;
    return weights / max(sum, 0.0001);
}

// Sample at a single scale with triplanar + stochastic
fn sample_triplanar_single_scale(
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    layer: i32,
    uv_scale: f32,
    sharpness: f32
) -> vec4<f32> {
    let weights = triplanar_weights(world_normal, sharpness);
    let scaled_pos = world_pos * uv_scale;

    // X projection: YZ plane (for surfaces facing +/- X)
    let uv_x = scaled_pos.yz;
    // Y projection: XZ plane (for surfaces facing +/- Y) - main terrain projection
    let uv_y = scaled_pos.xz;
    // Z projection: XY plane (for surfaces facing +/- Z)
    let uv_z = scaled_pos.xy;

    // Compute smooth derivatives from world position (these are continuous across triangles)
    let dpdx_pos = dpdx(scaled_pos);
    let dpdy_pos = dpdy(scaled_pos);
    
    // Derive UV gradients from world position gradients
    let ddx_x = dpdx_pos.yz;
    let ddy_x = dpdy_pos.yz;
    let ddx_y = dpdx_pos.xz;
    let ddy_y = dpdy_pos.xz;
    let ddx_z = dpdx_pos.xy;
    let ddy_z = dpdy_pos.xy;

    // Sample with stochastic tiling to break up repetition patterns
    let sample_x = sample_stochastic(albedo_array, albedo_array_sampler, uv_x, layer, ddx_x, ddy_x);
    let sample_y = sample_stochastic(albedo_array, albedo_array_sampler, uv_y, layer, ddx_y, ddy_y);
    let sample_z = sample_stochastic(albedo_array, albedo_array_sampler, uv_z, layer, ddx_z, ddy_z);

    return sample_x * weights.x + sample_y * weights.y + sample_z * weights.z;
}

// Sample texture array using triplanar projection with multi-scale blending
// Blends 3 different scales to break up visible repetition
fn sample_triplanar(
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    layer: i32,
    uv_scale: f32,
    sharpness: f32
) -> vec4<f32> {
    // Sample at 3 different scales
    let scale1 = uv_scale;        // Base scale
    let scale2 = uv_scale * 0.37; // Medium scale (use prime-ish ratio to avoid alignment)
    let scale3 = uv_scale * 0.11; // Large scale for overall variation
    
    let sample1 = sample_triplanar_single_scale(world_pos, world_normal, layer, scale1, sharpness);
    let sample2 = sample_triplanar_single_scale(world_pos, world_normal, layer, scale2, sharpness);
    let sample3 = sample_triplanar_single_scale(world_pos, world_normal, layer, scale3, sharpness);
    
    // Blend scales together
    // Use overlay-style blending: detail from small scale, color variation from large scales
    let base_color = sample3 * 0.3 + sample2 * 0.35 + sample1 * 0.35;
    
    // Add detail contrast from the fine scale
    let detail = sample1 - vec4<f32>(0.5);
    let final_color = base_color + detail * 0.3;
    
    return clamp(final_color, vec4<f32>(0.0), vec4<f32>(1.0));
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
    
    // Unpack weights to vec4<f32> for proper interpolation across the triangle
    let raw_weights = unpack_u8x4(material_weights);
    out.material_weights = vec4<f32>(raw_weights) / 255.0;

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

    // Unpack material indices (flat interpolated - same for whole triangle)
    let indices = unpack_u8x4(in.material_indices);
    let idx0 = i32(indices.x);
    let idx1 = i32(indices.y);
    let idx2 = i32(indices.z);
    let idx3 = i32(indices.w);

    // Weights are already unpacked in vertex shader and properly interpolated
    var weights = in.material_weights;

    // Normalize weights
    let sum = weights.x + weights.y + weights.z + weights.w;
    if sum > 0.0 {
        weights = weights / sum;
    }

    // Full terrain rendering with material blending
    let uv_scale = settings.uv_scale;
    let sharpness = settings.triplanar_sharpness;
    
    // Sample each material layer
    let albedo0 = sample_triplanar(world_pos, world_normal, idx0, uv_scale, sharpness);
    let albedo1 = sample_triplanar(world_pos, world_normal, idx1, uv_scale, sharpness);
    let albedo2 = sample_triplanar(world_pos, world_normal, idx2, uv_scale, sharpness);
    let albedo3 = sample_triplanar(world_pos, world_normal, idx3, uv_scale, sharpness);

    // Optional height-based blending
    if settings.height_blend_sharpness > 0.0 {
        let heights = vec4<f32>(albedo0.a, albedo1.a, albedo2.a, albedo3.a);
        weights = height_blend(heights, weights, settings.height_blend_sharpness);
    }

    // Blend final albedo
    let final_albedo = albedo0 * weights.x + albedo1 * weights.y +
                       albedo2 * weights.z + albedo3 * weights.w;
    
    // Simple lighting
    let light_dir = normalize(vec3<f32>(0.4, 0.8, 0.4));
    let ndotl = max(dot(world_normal, light_dir), 0.0);
    let ambient = 0.3;
    let diffuse = ndotl * 0.7;
    let lit_color = (ambient + diffuse) * final_albedo.rgb;
    
    var out: FragmentOutput;
    out.color = vec4<f32>(lit_color, 1.0);
    return out;

    // FULL PBR (uncomment when textures are verified working):
    // // Construct VertexOutput for PBR functions
    // var vertex_out: VertexOutput;
    // vertex_out.position = in.position;
    // vertex_out.world_position = in.world_position;
    // vertex_out.world_normal = in.world_normal;
    // vertex_out.uv = in.uv;
    //
    // // Generate PBR input from standard material
    // var pbr_input = pbr_input_from_standard_material(vertex_out, is_front);
    //
    // // Override the base color with our splatted texture
    // pbr_input.material.base_color = final_albedo;
    //
    // // Apply alpha discard if needed
    // pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);
    //
    // // Apply full PBR lighting
    // var out: FragmentOutput;
    // out.color = apply_pbr_lighting(pbr_input);
    // out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    //
    // return out;
}
