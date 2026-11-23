//! Bevy plugin for PlumeSplat terrain splatting.
//!
//! This module provides the [`PlumeSplatPlugin`] which registers all necessary
//! shaders, materials, and systems with Bevy's rendering pipeline.
//!
//! # Usage
//!
//! Simply add the plugin to your Bevy app:
//!
//! ```rust,no_run
//! use bevy::prelude::*;
//! use plumesplat::PlumeSplatPlugin;
//!
//! App::new()
//!     .add_plugins(DefaultPlugins)
//!     .add_plugins(PlumeSplatPlugin)
//!     .run();
//! ```
//!
//! # What This Plugin Registers
//!
//! - **Embedded Shader**: The WESL material shader is embedded into the binary
//! - **Material Plugin**: Registers [`PlumeSplatMaterial`](crate::PlumeSplatMaterial)
//! - **Processing System**: Automatically converts [`PendingSplatMaterial`](crate::PendingSplatMaterial)
//!   into final materials when textures are loaded

use std::borrow::Cow;
use std::collections::HashMap;

use bevy::app::{App, Plugin, Update};
use bevy::asset::{embedded_asset, AssetId, AssetServer, Assets, Handle, RenderAssetUsages};
use bevy::ecs::entity::Entity;
use bevy::ecs::system::{Commands, Local, Query, Res, ResMut};
use bevy::image::{Image, ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::log::{error, warn};
use bevy::pbr::{MaterialPlugin, MeshMaterial3d};
use bevy::render::render_resource::{
    Extent3d, TextureDataOrder, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::tasks::{futures::check_ready, AsyncComputeTaskPool, Task};

use crate::builder::PendingSplatMaterial;
use crate::material::{PlumeSplatExtension, PlumeSplatMaterial};

/// Bevy plugin that enables PlumeSplat terrain splatting.
///
/// This plugin must be added to your Bevy app to use PlumeSplat materials.
/// It handles:
///
/// - Embedding and registering the WESL shader
/// - Registering the [`PlumeSplatMaterial`](crate::PlumeSplatMaterial) with Bevy's material system
/// - Processing [`PendingSplatMaterial`](crate::PendingSplatMaterial) components into final materials
///
/// # Example
///
/// ```rust,no_run
/// use bevy::prelude::*;
/// use plumesplat::prelude::*;
///
/// fn main() {
///     App::new()
///         .add_plugins(DefaultPlugins)
///         .add_plugins(PlumeSplatPlugin)
///         .add_systems(Startup, setup)
///         .run();
/// }
///
/// fn setup(
///     mut commands: Commands,
///     asset_server: Res<AssetServer>,
/// ) {
///     // Using the builder API (recommended)
///     let pending = SplatMaterialBuilder::new()
///         .add_layer(MaterialLayer::new(asset_server.load("grass.png")))
///         .add_layer(MaterialLayer::new(asset_server.load("dirt.png")))
///         .build();
///
///     commands.spawn((
///         Mesh3d(Handle::default()),
///         pending,
///     ));
///     // Plugin automatically creates the material when textures load!
/// }
/// ```
///
/// # Shader
///
/// The plugin embeds `shaders/plumesplat_ext.wesl`, the material extension shader.
pub struct PlumeSplatPlugin;

impl Plugin for PlumeSplatPlugin {
    fn build(&self, app: &mut App) {
        // Embed the shader into the binary for self-contained distribution
        embedded_asset!(app, "shaders/plumesplat_ext.wesl");

        // Register the extended material type with Bevy's material system
        app.add_plugins(MaterialPlugin::<PlumeSplatMaterial>::default());

        // Add system to process pending materials when textures are loaded
        app.add_systems(Update, process_pending_materials);
    }
}

/// The texture handles of every layer: (albedo, normal, pbr) per layer.
type LayerKey = Vec<(AssetId<Image>, Option<AssetId<Image>>, Option<AssetId<Image>>)>;
type ArrayIds = (AssetId<Image>, Option<AssetId<Image>>, Option<AssetId<Image>>);
type ArrayHandles = (Handle<Image>, Option<Handle<Image>>, Option<Handle<Image>>);

/// Texture arrays, shared by every entity that uses the same layer textures.
///
/// `ready` only stores ids: the arrays are freed once no material uses them,
/// and are rebuilt if that layer set shows up again.
#[derive(Default)]
struct ArrayCache {
    building: HashMap<LayerKey, Task<BuiltArrays>>,
    ready: HashMap<LayerKey, ArrayIds>,
}

struct BuiltArrays {
    albedo: Image,
    normal: Option<Image>,
    pbr: Option<Image>,
}

/// System that processes [`PendingSplatMaterial`] components.
///
/// When all textures for a pending material are loaded, this system:
/// 1. Combines individual textures into texture arrays on a background thread
///    (or reuses the arrays of another entity with the same layer textures)
/// 2. Creates the final [`PlumeSplatMaterial`]
/// 3. Removes the [`PendingSplatMaterial`] component
/// 4. Adds [`MeshMaterial3d<PlumeSplatMaterial>`] to the entity
///
/// If a texture fails to load, an error is logged and the pending component is removed.
fn process_pending_materials(
    mut commands: Commands,
    query: Query<(Entity, &PendingSplatMaterial)>,
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<PlumeSplatMaterial>>,
    mut cache: Local<ArrayCache>,
) {
    for (entity, pending) in &query {
        if let Some(handle) = pending.failed_texture(&asset_server) {
            error!(
                "plumesplat: texture {:?} failed to load; the splat material for {entity} was not created",
                handle.path()
            );
            commands.entity(entity).remove::<PendingSplatMaterial>();
            continue;
        }
        if !pending.is_ready(&images) {
            continue;
        }
        let Some((albedo_array, normal_array, pbr_array)) =
            shared_arrays(&mut cache, pending, &mut images)
        else {
            continue; // still building
        };

        let mut extension = PlumeSplatExtension::new(albedo_array);
        extension.settings = pending.settings;
        extension.layer_settings = pending.layer_settings();
        extension.normal_array = normal_array;
        extension.pbr_array = pbr_array;

        let material_handle = materials.add(PlumeSplatMaterial {
            base: pending.base_material.clone(),
            extension,
        });

        commands
            .entity(entity)
            .remove::<PendingSplatMaterial>()
            .insert(MeshMaterial3d(material_handle));
    }
}

/// Returns the texture arrays for this layer set, starting a background build
/// the first time. `None` while the build is running.
fn shared_arrays(
    cache: &mut ArrayCache,
    pending: &PendingSplatMaterial,
    images: &mut Assets<Image>,
) -> Option<ArrayHandles> {
    let key: LayerKey = pending
        .layers
        .iter()
        .map(|l| (l.albedo.id(), l.normal.as_ref().map(Handle::id), l.pbr.as_ref().map(Handle::id)))
        .collect();

    if let Some(handles) = cache.ready.get(&key).and_then(|ids| upgrade(ids, images)) {
        return Some(handles);
    }

    let task = cache
        .building
        .entry(key.clone())
        .or_insert_with(|| spawn_build(pending, images));
    let built = check_ready(task)?;
    cache.building.remove(&key);

    let handles = (
        images.add(built.albedo),
        built.normal.map(|image| images.add(image)),
        built.pbr.map(|image| images.add(image)),
    );
    let ids = (
        handles.0.id(),
        handles.1.as_ref().map(Handle::id),
        handles.2.as_ref().map(Handle::id),
    );
    cache.ready.insert(key, ids);
    Some(handles)
}

/// Strong handles for cached arrays, or `None` if any of them was already freed.
fn upgrade(ids: &ArrayIds, images: &mut Assets<Image>) -> Option<ArrayHandles> {
    let albedo = images.get_strong_handle(ids.0)?;
    let normal = match ids.1 {
        Some(id) => Some(images.get_strong_handle(id)?),
        None => None,
    };
    let pbr = match ids.2 {
        Some(id) => Some(images.get_strong_handle(id)?),
        None => None,
    };
    Some((albedo, normal, pbr))
}

/// Copies the layer pixels and builds the arrays (mipmaps included) on the
/// async compute pool, so loading does not stall a frame.
fn spawn_build(pending: &PendingSplatMaterial, images: &Assets<Image>) -> Task<BuiltArrays> {
    let first_albedo = images.get(&pending.layers[0].albedo).unwrap();
    let size = (
        first_albedo.texture_descriptor.size.width,
        first_albedo.texture_descriptor.size.height,
    );

    let albedo = layer_pixels(pending.layers.iter().map(|l| Some(&l.albedo)), size, images);
    let normal = pending
        .has_normals()
        .then(|| layer_pixels(pending.layers.iter().map(|l| l.normal.as_ref()), size, images));
    let pbr = pending
        .has_pbr()
        .then(|| layer_pixels(pending.layers.iter().map(|l| l.pbr.as_ref()), size, images));

    AsyncComputeTaskPool::get().spawn(async move {
        BuiltArrays {
            albedo: build_array(albedo, size, [255, 255, 255, 255], ArrayKind::Color),
            normal: normal.map(|layers| build_array(layers, size, [128, 128, 255, 255], ArrayKind::Normal)),
            // metallic=0, roughness=0.5, ao=1.0, height=0.5
            pbr: pbr.map(|layers| build_array(layers, size, [0, 128, 255, 128], ArrayKind::Data)),
        }
    })
}

/// RGBA8 pixels of each layer. `None` for a missing texture, or one that can't
/// be used (a warning is logged); the array fills those with a default color.
fn layer_pixels<'a>(
    layers: impl Iterator<Item = Option<&'a Handle<Image>>>,
    size: (u32, u32),
    images: &Assets<Image>,
) -> Vec<Option<Vec<u8>>> {
    layers
        .enumerate()
        .map(|(index, layer)| {
            let handle = layer?;
            let pixels = rgba8_bytes(images.get(handle)?, size);
            if pixels.is_none() {
                warn!(
                    "plumesplat: layer {index} texture {:?} must be {}x{}, 8- or 16-bit RGBA, \
                     with CPU-side data; using a default color instead",
                    handle.path(),
                    size.0,
                    size.1
                );
            }
            pixels.map(Cow::into_owned)
        })
        .collect()
}

/// Returns the image as RGBA8 bytes. Bevy loads 8-bit PNGs (also grayscale) as
/// RGBA8 and 16-bit PNGs (common for normal maps) as `Rgba16Unorm`, which is
/// reduced to its high bytes here. `None` if the size or format is unsupported,
/// or the image has no CPU-side data.
fn rgba8_bytes(image: &Image, (width, height): (u32, u32)) -> Option<Cow<'_, [u8]>> {
    let size = image.texture_descriptor.size;
    if (size.width, size.height) != (width, height) {
        return None;
    }
    let data = image.data.as_deref()?;
    match image.texture_descriptor.format {
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb => Some(Cow::Borrowed(data)),
        // Little-endian u16 per channel: the high byte is the 8-bit value.
        TextureFormat::Rgba16Unorm => Some(Cow::Owned(
            data.as_chunks::<2>().0.iter().map(|[_, high]| *high).collect(),
        )),
        _ => None,
    }
}

/// How an array's texels are stored and filtered into mipmaps.
#[derive(Clone, Copy)]
enum ArrayKind {
    /// sRGB color: mips are averaged in linear space.
    Color,
    /// Tangent-space normals: mips are renormalized.
    Normal,
    /// Linear data (metallic, roughness, AO, height).
    Data,
}

impl ArrayKind {
    fn format(self) -> TextureFormat {
        match self {
            ArrayKind::Color => TextureFormat::Rgba8UnormSrgb,
            ArrayKind::Normal | ArrayKind::Data => TextureFormat::Rgba8Unorm,
        }
    }
}

/// Creates a mipmapped texture array, one layer per entry. `None` layers are
/// filled with `default_color`.
fn build_array(
    layers: Vec<Option<Vec<u8>>>,
    (width, height): (u32, u32),
    default_color: [u8; 4],
    kind: ArrayKind,
) -> Image {
    let mip_count = calculate_mip_count(width, height);
    let default_data = default_color.repeat((width * height) as usize);

    let mut array_data = Vec::new();
    for layer in &layers {
        let data = layer.as_deref().unwrap_or(&default_data);
        array_data.extend(generate_mipmaps(data, width, height, mip_count, kind));
    }

    Image {
        data: Some(array_data),
        texture_descriptor: TextureDescriptor {
            label: Some("splat_texture_array"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: layers.len() as u32,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: kind.format(),
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        sampler: ImageSampler::Descriptor(ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::Repeat,
            address_mode_v: ImageAddressMode::Repeat,
            address_mode_w: ImageAddressMode::ClampToEdge,
            mag_filter: ImageFilterMode::Linear,
            min_filter: ImageFilterMode::Linear,
            mipmap_filter: ImageFilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        }),
        texture_view_descriptor: None,
        asset_usage: RenderAssetUsages::RENDER_WORLD,
        data_order: TextureDataOrder::LayerMajor,
        copy_on_resize: false,
    }
}

/// Calculates the number of mip levels for a texture.
fn calculate_mip_count(width: u32, height: u32) -> u32 {
    (width.min(height) as f32).log2().floor() as u32 + 1
}

/// Returns `data` followed by `mip_count - 1` downsampled levels.
///
/// Filtering happens in float: sRGB color is averaged in linear space and
/// normals are renormalized, so distant terrain keeps its brightness and relief.
fn generate_mipmaps(data: &[u8], width: u32, height: u32, mip_count: u32, kind: ArrayKind) -> Vec<u8> {
    let srgb_to_linear: [f32; 256] = std::array::from_fn(|i| {
        let c = i as f32 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    });
    let decode = |p: &[u8; 4]| -> [f32; 4] {
        match kind {
            ArrayKind::Color => [
                srgb_to_linear[p[0] as usize],
                srgb_to_linear[p[1] as usize],
                srgb_to_linear[p[2] as usize],
                p[3] as f32 / 255.0,
            ],
            ArrayKind::Normal | ArrayKind::Data => p.map(|c| c as f32 / 255.0),
        }
    };
    let encode = |p: &[f32; 4]| -> [u8; 4] {
        let to_u8 = |c: f32| (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        match kind {
            ArrayKind::Color => {
                let to_srgb = |c: f32| {
                    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
                };
                [to_u8(to_srgb(p[0])), to_u8(to_srgb(p[1])), to_u8(to_srgb(p[2])), to_u8(p[3])]
            }
            ArrayKind::Normal | ArrayKind::Data => p.map(to_u8),
        }
    };

    let mut result = data.to_vec();
    let mut level: Vec<[f32; 4]> = data.as_chunks::<4>().0.iter().map(decode).collect();
    let (mut w, mut h) = (width, height);
    for _ in 1..mip_count {
        let (next_w, next_h) = ((w / 2).max(1), (h / 2).max(1));
        level = downsample(&level, w, h, next_w, next_h);
        if let ArrayKind::Normal = kind {
            level.iter_mut().for_each(renormalize);
        }
        result.extend(level.iter().flat_map(encode));
        (w, h) = (next_w, next_h);
    }
    result
}

/// 2x2 box filter; edge texels are repeated for odd sizes.
fn downsample(src: &[[f32; 4]], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<[f32; 4]> {
    let mut dst = Vec::with_capacity((dst_w * dst_h) as usize);
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let mut sum = [0.0f32; 4];
            for (ox, oy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let x = (dx * 2 + ox).min(src_w - 1);
                let y = (dy * 2 + oy).min(src_h - 1);
                let texel = src[(y * src_w + x) as usize];
                for c in 0..4 {
                    sum[c] += texel[c];
                }
            }
            dst.push(sum.map(|c| c * 0.25));
        }
    }
    dst
}

/// Renormalizes an encoded normal (rgb in 0..1) to unit length.
fn renormalize(texel: &mut [f32; 4]) {
    let v = [texel[0] * 2.0 - 1.0, texel[1] * 2.0 - 1.0, texel[2] * 2.0 - 1.0];
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-6 {
        for c in 0..3 {
            texel[c] = (v[c] / len + 1.0) * 0.5;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(format: TextureFormat, data: Vec<u8>) -> Image {
        Image::new(
            Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            TextureDimension::D2,
            data,
            format,
            RenderAssetUsages::default(),
        )
    }

    #[test]
    fn rgba8_is_borrowed_as_is() {
        let img = image(TextureFormat::Rgba8Unorm, vec![1, 2, 3, 4]);
        assert_eq!(&*rgba8_bytes(&img, (1, 1)).unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn rgba16_keeps_high_bytes() {
        // Little-endian u16 channels: 0x80FF, 0x1200, 0xFF01, 0x0000
        let img = image(TextureFormat::Rgba16Unorm, vec![0xFF, 0x80, 0x00, 0x12, 0x01, 0xFF, 0x00, 0x00]);
        assert_eq!(&*rgba8_bytes(&img, (1, 1)).unwrap(), &[0x80, 0x12, 0xFF, 0x00]);
    }

    #[test]
    fn wrong_size_or_format_is_rejected() {
        let img = image(TextureFormat::Rgba8Unorm, vec![0; 4]);
        assert!(rgba8_bytes(&img, (2, 2)).is_none());
        let img = image(TextureFormat::R8Unorm, vec![0]);
        assert!(rgba8_bytes(&img, (1, 1)).is_none());
    }

    #[test]
    fn mip_chain_sizes() {
        // (width, height, expected texel count over all levels)
        for (w, h, texels) in [(4, 4, 16 + 4 + 1), (3, 5, 15 + 2), (1, 1, 1), (8, 2, 16 + 4)] {
            let mips = calculate_mip_count(w, h);
            let data = vec![0u8; (w * h * 4) as usize];
            let out = generate_mipmaps(&data, w, h, mips, ArrayKind::Data);
            assert_eq!(out.len(), texels * 4, "{w}x{h}");
        }
    }

    #[test]
    fn data_mips_average_linearly() {
        let data = [[0, 0, 0, 0], [255, 255, 255, 255], [0, 0, 0, 0], [255, 255, 255, 255]].concat();
        let out = generate_mipmaps(&data, 2, 2, 2, ArrayKind::Data);
        assert_eq!(&out[16..], &[128, 128, 128, 128]);
    }

    #[test]
    fn color_mips_average_in_linear_space() {
        // Half black, half white is 0.5 linear, which is 188 in sRGB (not 128).
        let data = [[0, 0, 0, 255], [255, 255, 255, 255], [0, 0, 0, 255], [255, 255, 255, 255]].concat();
        let out = generate_mipmaps(&data, 2, 2, 2, ArrayKind::Color);
        assert_eq!(&out[16..], &[188, 188, 188, 255]);
    }

    #[test]
    fn normal_mips_are_renormalized() {
        // (+0.6, 0, 0.8) and (-0.6, 0, 0.8) average to (0, 0, 0.8): must come back as (0, 0, 1).
        let n = |x: f32, z: f32| [((x + 1.0) * 127.5).round() as u8, 128, ((z + 1.0) * 127.5).round() as u8, 255];
        let data = [n(0.6, 0.8), n(-0.6, 0.8), n(0.6, 0.8), n(-0.6, 0.8)].concat();
        let out = generate_mipmaps(&data, 2, 2, 2, ArrayKind::Normal);
        assert_eq!(out[16 + 2], 255, "z should be 1.0 after renormalizing");
        assert!((out[16] as i32 - 128).abs() <= 1, "x should be ~0");
    }
}
