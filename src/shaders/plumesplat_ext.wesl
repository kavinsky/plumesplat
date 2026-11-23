// PlumeSplat material extension shader (WESL, Bevy 0.20).
//
// Blends up to 4 layers from texture arrays per vertex, using triplanar
// projection and stochastic tiling, then lights the result with Bevy's PBR.
//
// Shader defs (set in `PlumeSplatExtension::specialize`):
// - HAS_NORMAL_ARRAY: `normal_array` is bound, otherwise the geometry normal is used.
// - HAS_PBR_ARRAY: `pbr_array` is bound, otherwise the base StandardMaterial's
//   metallic / roughness are used and height comes from albedo alpha.

import bevy_pbr::render::{
    forward_io::{Vertex, FragmentOutput, decompress_vertex},
    mesh_functions,
    mesh_bindings::mesh,
    mesh_view_bindings::view,
    view_transformations::position_world_to_clip,
    pbr_bindings,
    pbr_types,
    pbr_functions,
};

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var albedo_array: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var albedo_sampler: sampler;

struct PlumeSplatSettings {
    uv_scale: f32,
    triplanar_sharpness: f32,
    height_blend_sharpness: f32,
    blend_offset: f32,
    blend_exponent: f32,
    // Multi-scale albedo: far sample scale (relative to uv_scale) and how
    // strongly it modulates the near sample (0 = off, skips the far sample).
    far_scale: f32,
    far_mix: f32,
    // Random per-tile rotation in stochastic tiling: 0 = off, 1 = full turn.
    stochastic_rotation: f32,
}
@group(constants::MATERIAL_BIND_GROUP) @binding(102) var<uniform> settings: PlumeSplatSettings;

@group(constants::MATERIAL_BIND_GROUP) @binding(103) var normal_array: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(104) var normal_sampler: sampler;

// R = metallic, G = roughness, B = AO, A = height
@group(constants::MATERIAL_BIND_GROUP) @binding(105) var pbr_array: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(106) var pbr_sampler: sampler;

// Per-layer overrides of `settings`, one vec4 per layer. A negative value means
// "inherit the global setting". x = stochastic rotation; y, z, w reserved.
struct PlumeSplatLayerSettings {
    layers: array<vec4<f32>, 256>,
}
@group(constants::MATERIAL_BIND_GROUP) @binding(107) var<uniform> layer_settings: PlumeSplatLayerSettings;

// Layers / projection planes below these weights are not sampled.
const MIN_LAYER_WEIGHT: f32 = 0.01;
const MIN_PLANE_WEIGHT: f32 = 0.01;
// The far sample is rotated by this fixed angle (~37°, cos/sin below) so its
// tile grid never lines up with the near grid's axes.
const FAR_ROT: mat2x2<f32> = mat2x2<f32>(0.7986, 0.6018, -0.6018, 0.7986);
// Upper bound on the far/average ratio, so layers whose average is very dark in
// one channel can't blow that channel up.
const FAR_MAX_RATIO: f32 = 4.0;

struct SplatVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) @interpolate(flat) material_indices: u32,
    @location(3) material_weights: vec4<f32>,
    @location(4) @interpolate(flat) instance_index: u32,
}

// Everything needed to sample one triplanar projection, computed once per fragment.
struct Triplanar {
    p: vec3<f32>,   // world position * uv_scale
    dx: vec3<f32>,  // dpdx(p)
    dy: vec3<f32>,  // dpdy(p)
    n: vec3<f32>,   // geometry normal
    w: vec3<f32>,   // plane weights (sum to 1)
}

fn unpack_u8x4(packed: u32) -> vec4<u32> {
    return (vec4<u32>(packed) >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4<u32>(0xFFu);
}

// Random values in [0, 1) for an integer tile cell: xy = offset, z = rotation.
// Uses the pcg2d integer hash (Jarzynski & Olano 2020) instead of a sin-based
// one: `sin` loses precision for large arguments, which made the pattern blocky
// or repeat far from the origin, and its result differed between GPU vendors.
fn hash3(cell: vec2<f32>) -> vec3<f32> {
    var v = bitcast<vec2<u32>>(vec2<i32>(cell));
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    // Offset: top 24 bits of each word, exact in f32. Rotation: the 8 low bits
    // of each word, which the offset doesn't use (16 bits, 65536 angles).
    let offset = vec2<f32>(v >> vec2<u32>(8u)) * (1.0 / 16777216.0);
    let rot = f32((v.x & 0xFFu) | ((v.y & 0xFFu) << 8u)) * (1.0 / 65536.0);
    return vec3<f32>(offset, rot);
}

// Stochastic rotation strength of a layer: its override, or the global value.
fn layer_rotation(layer: i32) -> f32 {
    let value = layer_settings.layers[layer].x;
    return select(value, settings.stochastic_rotation, value < 0.0);
}

// Stochastic tiling: blend the 4 nearest tile centers, each with its own random
// offset (and, with `stochastic_rotation`, its own random rotation). Weights are
// bilinear (sharpened), so they reach 0 exactly at the point where a tile center
// leaves the 2x2 neighbourhood: no seams. With the sharpening, most pixels need
// 1 or 2 taps instead of 4.
//
// `is_normal`: the texture holds tangent-space normals. They are returned
// decoded (-1..1), with each tap's xy rotated back into the unrotated UV frame,
// so the result is consistent with `uv` whatever the rotation.
fn sample_stochastic(
    tex: texture_2d_array<f32>,
    tex_sampler: sampler,
    uv: vec2<f32>,
    layer: i32,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>,
    is_normal: bool,
) -> vec4<f32> {
    let g = uv - 0.5;
    let cell = floor(g);
    let b = clamp((fract(g) - 0.5) * 2.0 + 0.5, vec2<f32>(0.0), vec2<f32>(1.0));
    let rotation = layer_rotation(layer);

    var color = vec4<f32>(0.0);
    for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
            let o = vec2<f32>(f32(i), f32(j));
            let axis_w = mix(1.0 - b, b, o);
            let w = axis_w.x * axis_w.y;
            if w > 0.0 {
                let h = hash3(cell + o);
                // Rotation is skipped entirely when off for this layer. Not a
                // uniform branch (it depends on the layer), which is fine:
                // textureSampleGrad takes explicit gradients.
                var r = mat2x2<f32>(1.0, 0.0, 0.0, 1.0);
                if rotation > 0.0 {
                    let angle = (h.z - 0.5) * 6.2831853 * rotation;
                    let c = cos(angle);
                    let s = sin(angle);
                    r = mat2x2<f32>(c, s, -s, c);
                }
                // Gradients must be rotated with the UV, or mip selection breaks.
                var texel = textureSampleGrad(tex, tex_sampler, r * uv + h.xy, layer, r * ddx_uv, r * ddy_uv);
                if is_normal {
                    var nrm = texel.xyz * 2.0 - 1.0;
                    // Inverse rotation (transpose) back to the unrotated frame.
                    nrm = vec3<f32>(nrm.xy * r, nrm.z);
                    texel = vec4<f32>(nrm, texel.w);
                }
                color += texel * w;
            }
        }
    }
    return color; // weights already sum to 1
}

// Triplanar sample of a color-like texture. Planes: X -> zy, Y -> xz, Z -> xy.
fn sample_triplanar(
    tex: texture_2d_array<f32>,
    tex_sampler: sampler,
    t: Triplanar,
    scale: f32,
    layer: i32,
) -> vec4<f32> {
    let p = t.p * scale;
    let dx = t.dx * scale;
    let dy = t.dy * scale;
    var color = vec4<f32>(0.0);
    if t.w.x > MIN_PLANE_WEIGHT {
        color += sample_stochastic(tex, tex_sampler, p.zy, layer, dx.zy, dy.zy, false) * t.w.x;
    }
    if t.w.y > MIN_PLANE_WEIGHT {
        color += sample_stochastic(tex, tex_sampler, p.xz, layer, dx.xz, dy.xz, false) * t.w.y;
    }
    if t.w.z > MIN_PLANE_WEIGHT {
        color += sample_stochastic(tex, tex_sampler, p.xy, layer, dx.xy, dy.xy, false) * t.w.z;
    }
    return color;
}

// Triplanar albedo at the far scale: one plain tap per plane, no stochastic
// tiling. The large scale change already hides the near tile pattern, and the
// fixed rotation keeps the far grid off the near grid's axes, so this costs at
// most 3 taps instead of up to 12.
fn sample_albedo_far(t: Triplanar, scale: f32, layer: i32) -> vec3<f32> {
    let p = t.p * scale;
    let dx = t.dx * scale;
    let dy = t.dy * scale;
    var color = vec3<f32>(0.0);
    if t.w.x > MIN_PLANE_WEIGHT {
        color += textureSampleGrad(albedo_array, albedo_sampler, FAR_ROT * p.zy, layer, FAR_ROT * dx.zy, FAR_ROT * dy.zy).rgb * t.w.x;
    }
    if t.w.y > MIN_PLANE_WEIGHT {
        color += textureSampleGrad(albedo_array, albedo_sampler, FAR_ROT * p.xz, layer, FAR_ROT * dx.xz, FAR_ROT * dy.xz).rgb * t.w.y;
    }
    if t.w.z > MIN_PLANE_WEIGHT {
        color += textureSampleGrad(albedo_array, albedo_sampler, FAR_ROT * p.xy, layer, FAR_ROT * dx.xy, FAR_ROT * dy.xy).rgb * t.w.z;
    }
    return color;
}

// Albedo with large-scale variation. Averaging two scales would wash out the
// texture's contrast, so the far sample only modulates the near one relative to
// the layer's average color (its 1x1 top mip).
fn sample_albedo(t: Triplanar, layer: i32) -> vec4<f32> {
    let near = sample_triplanar(albedo_array, albedo_sampler, t, 1.0, layer);
    // Uniform branch; legal because every sample uses explicit gradients / LOD.
    if settings.far_mix <= 0.0 {
        return near;
    }
    let far = sample_albedo_far(t, settings.far_scale, layer);
    let top_mip = f32(textureNumLevels(albedo_array) - 1u);
    let avg = textureSampleLevel(albedo_array, albedo_sampler, vec2<f32>(0.5), layer, top_mip).rgb;
    let ratio = min(far / max(avg, vec3<f32>(0.01)), vec3<f32>(FAR_MAX_RATIO));
    let variation = mix(vec3<f32>(1.0), ratio, settings.far_mix);
    return vec4<f32>(clamp(near.rgb * variation, vec3<f32>(0.0), vec3<f32>(1.0)), near.a);
}

// Triplanar normal map using "whiteout" blending (Ben Golus). Each tangent-space
// sample is combined with the geometry normal of its plane and swizzled to world
// space, so a flat normal map returns the geometry normal on every face.
fn sample_triplanar_normal(t: Triplanar, layer: i32) -> vec3<f32> {
    let n = t.n;
    var result = vec3<f32>(0.0);
    if t.w.x > MIN_PLANE_WEIGHT {
        let a = sample_stochastic(normal_array, normal_sampler, t.p.zy, layer, t.dx.zy, t.dy.zy, true).xyz;
        result += vec3<f32>(abs(a.z) * n.x, a.y + n.y, a.x + n.z) * t.w.x;
    }
    if t.w.y > MIN_PLANE_WEIGHT {
        let a = sample_stochastic(normal_array, normal_sampler, t.p.xz, layer, t.dx.xz, t.dy.xz, true).xyz;
        result += vec3<f32>(a.x + n.x, abs(a.z) * n.y, a.y + n.z) * t.w.y;
    }
    if t.w.z > MIN_PLANE_WEIGHT {
        let a = sample_stochastic(normal_array, normal_sampler, t.p.xy, layer, t.dx.xy, t.dy.xy, true).xyz;
        result += vec3<f32>(a.x + n.x, a.y + n.y, abs(a.z) * n.z) * t.w.z;
    }
    return normalize(result);
}

// Normalize vertex weights, then apply blend offset / exponent and renormalize.
fn blend_weights(raw: vec4<f32>) -> vec4<f32> {
    let sum = dot(raw, vec4<f32>(1.0));
    if sum <= 0.0 {
        return vec4<f32>(1.0, 0.0, 0.0, 0.0);
    }
    let w = raw / sum;
    var adjusted = max(w - vec4<f32>(settings.blend_offset), vec4<f32>(0.0));
    if settings.blend_exponent > 1.0 {
        adjusted = pow(adjusted, vec4<f32>(settings.blend_exponent));
    }
    let adjusted_sum = dot(adjusted, vec4<f32>(1.0));
    if adjusted_sum > 0.0 {
        return adjusted / adjusted_sum;
    }
    return w;
}

// Height-based blending: layers whose height + weight is close to the maximum win.
fn height_blend(heights: vec4<f32>, weights: vec4<f32>, sharpness: f32) -> vec4<f32> {
    let hw = heights + weights;
    let max_h = max(max(hw.x, hw.y), max(hw.z, hw.w));
    let blend = max(hw - max_h + sharpness, vec4<f32>(0.0));
    let sum = dot(blend, vec4<f32>(1.0));
    if sum > 0.0 {
        return blend / sum;
    }
    return weights;
}

@vertex
fn vertex(
    vertex_in: Vertex,
    @location(10) material_indices: u32,
    @location(11) material_weights: u32,
) -> SplatVertexOutput {
    var out: SplatVertexOutput;
    let v = decompress_vertex(vertex_in, vertex_in.instance_index);
    let world_from_local = mesh_functions::get_world_from_local(vertex_in.instance_index);

    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(v.position, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(v.normal, vertex_in.instance_index);
    out.material_indices = material_indices;
    // Unpacked here so the weights interpolate across the triangle.
    out.material_weights = vec4<f32>(unpack_u8x4(material_weights)) / 255.0;
    out.instance_index = vertex_in.instance_index;
    return out;
}

@fragment
fn fragment(
    in: SplatVertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    // The extension is not bindless, so the base StandardMaterial is a plain uniform.
    let base = pbr_bindings::material;
    let mesh_flags = mesh[in.instance_index].flags;

    var n = normalize(in.world_normal);
    let double_sided = (base.flags & pbr_types::STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT) != 0u;
    if double_sided && !pbr_functions::winding_corrected_front_facing(mesh_flags, is_front) {
        n = -n;
    }

    // Derivatives must be taken in uniform control flow, so compute them once here.
    var t: Triplanar;
    t.p = in.world_position.xyz * settings.uv_scale;
    t.dx = dpdx(t.p);
    t.dy = dpdy(t.p);
    t.n = n;
    let plane_w = pow(abs(n), vec3<f32>(max(settings.triplanar_sharpness, 1.0)));
    t.w = plane_w / max(dot(plane_w, vec3<f32>(1.0)), 0.0001);

    let idx = vec4<i32>(unpack_u8x4(in.material_indices));
    var w = blend_weights(in.material_weights);

    // Pass 1: albedo (+ PBR) for every layer that contributes.
    var albedo: array<vec4<f32>, 4>;
    var pbr: array<vec4<f32>, 4>;
    var heights = vec4<f32>(-1.0e4); // skipped layers never win height blending
    for (var i = 0u; i < 4u; i++) {
        if w[i] < MIN_LAYER_WEIGHT {
            w[i] = 0.0;
            continue;
        }
        albedo[i] = sample_albedo(t, idx[i]);
        heights[i] = albedo[i].a;
        @if(HAS_PBR_ARRAY) {
            pbr[i] = sample_triplanar(pbr_array, pbr_sampler, t, 1.0, idx[i]);
            heights[i] = pbr[i].a;
        }
    }
    w = w / max(dot(w, vec4<f32>(1.0)), 0.0001);

    if settings.height_blend_sharpness > 0.0 {
        w = height_blend(heights, w, settings.height_blend_sharpness);
    }

    // Pass 2: blend everything with the final weights.
    var color = vec4<f32>(0.0);
    var normal_sum = vec3<f32>(0.0);
    var mrao = vec3<f32>(0.0);
    for (var i = 0u; i < 4u; i++) {
        if w[i] <= 0.0 {
            continue;
        }
        color += albedo[i] * w[i];
        @if(HAS_NORMAL_ARRAY)
        normal_sum += sample_triplanar_normal(t, idx[i]) * w[i];
        @if(HAS_PBR_ARRAY)
        mrao += pbr[i].rgb * w[i];
    }

    var pbr_input = pbr_types::pbr_input_new();
    pbr_input.material = base;
    pbr_input.material.base_color = vec4<f32>(color.rgb * base.base_color.rgb, base.base_color.a);
    @if(HAS_PBR_ARRAY) {
        pbr_input.material.metallic = mrao.x;
        pbr_input.material.perceptual_roughness = mrao.y;
        pbr_input.diffuse_occlusion = vec3<f32>(mrao.z);
    }

    pbr_input.frag_coord = in.position;
    pbr_input.world_position = in.world_position;
    pbr_input.world_normal = n;
    @if(HAS_NORMAL_ARRAY)
    pbr_input.N = normalize(normal_sum);
    @else
    pbr_input.N = n;
    pbr_input.is_orthographic = view.clip_from_view[3].w == 1.0;
    pbr_input.V = pbr_functions::calculate_view(in.world_position, pbr_input.is_orthographic);
    pbr_input.flags = mesh_flags;

    var out: FragmentOutput;
    out.color = pbr_functions::apply_pbr_lighting(pbr_input);
    out.color = pbr_functions::main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
