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
//! - **Embedded Shaders**: All WGSL shader files are embedded into the binary
//! - **Material Plugin**: Registers [`PlumeSplatMaterial`](crate::PlumeSplatMaterial)
//! - **Processing System**: Automatically converts [`PendingSplatMaterial`](crate::PendingSplatMaterial)
//!   into final materials when textures are loaded

use std::borrow::Cow;

use bevy::app::{App, Plugin, Update};
use bevy::log::warn;
use bevy::asset::{embedded_asset, Assets, Handle, RenderAssetUsages};
use bevy::ecs::entity::Entity;
use bevy::ecs::system::{Commands, Query, ResMut};
use bevy::image::{Image, ImageSampler, ImageSamplerDescriptor, ImageAddressMode, ImageFilterMode};
use bevy::pbr::{MeshMaterial3d, MaterialPlugin};
use bevy::render::render_resource::{
    Extent3d, TextureDataOrder, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};

use crate::builder::PendingSplatMaterial;
use crate::material::{PlumeSplatExtension, PlumeSplatMaterial};

/// Bevy plugin that enables PlumeSplat terrain splatting.
///
/// This plugin must be added to your Bevy app to use PlumeSplat materials.
/// It handles:
///
/// - Embedding and registering WGSL shader files
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

/// System that processes [`PendingSplatMaterial`] components.
///
/// When all textures for a pending material are loaded, this system:
/// 1. Combines individual textures into texture arrays
/// 2. Creates the final [`PlumeSplatMaterial`]
/// 3. Removes the [`PendingSplatMaterial`] component
/// 4. Adds [`MeshMaterial3d<PlumeSplatMaterial>`] to the entity
fn process_pending_materials(
    mut commands: Commands,
    query: Query<(Entity, &PendingSplatMaterial)>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<PlumeSplatMaterial>>,
) {
    for (entity, pending) in query.iter() {
        // Check if all textures are loaded
        if !pending.is_ready(&images) {
            continue;
        }

        // Combine textures into arrays
        let (albedo_array, normal_array, pbr_array) =
            combine_textures_into_arrays(pending, &mut images);

        // Create the extension
        let mut extension = PlumeSplatExtension::new(albedo_array);
        extension.settings = pending.settings;

        if let Some(normal) = normal_array {
            extension.normal_array = Some(normal);
        }
        if let Some(pbr) = pbr_array {
            extension.pbr_array = Some(pbr);
        }

        // Create the final material
        let material = PlumeSplatMaterial {
            base: pending.base_material.clone(),
            extension,
        };

        let material_handle = materials.add(material);

        // Update the entity: remove pending, add final material
        commands
            .entity(entity)
            .remove::<PendingSplatMaterial>()
            .insert(MeshMaterial3d(material_handle));
    }
}

/// Combines individual layer textures into texture arrays.
///
/// Returns (albedo_array, optional_normal_array, optional_pbr_array).
fn combine_textures_into_arrays(
    pending: &PendingSplatMaterial,
    images: &mut Assets<Image>,
) -> (Handle<Image>, Option<Handle<Image>>, Option<Handle<Image>>) {
    // Get dimensions from first albedo texture
    let first_albedo = images.get(&pending.layers[0].albedo).unwrap();
    let size = (
        first_albedo.texture_descriptor.size.width,
        first_albedo.texture_descriptor.size.height,
    );

    // Albedo is color data (sRGB). Normal and PBR are raw data and must stay linear.
    let albedo_array = create_texture_array(
        pending.layers.iter().map(|l| Some(&l.albedo)),
        size,
        [255, 255, 255, 255], // unused: every layer has an albedo
        TextureFormat::Rgba8UnormSrgb,
        images,
    );

    let normal_array = pending.has_normals().then(|| {
        create_texture_array(
            pending.layers.iter().map(|l| l.normal.as_ref()),
            size,
            [128, 128, 255, 255], // flat normal
            TextureFormat::Rgba8Unorm,
            images,
        )
    });

    let pbr_array = pending.has_pbr().then(|| {
        create_texture_array(
            pending.layers.iter().map(|l| l.pbr.as_ref()),
            size,
            [0, 128, 255, 128], // metallic=0, roughness=0.5, ao=1.0, height=0.5
            TextureFormat::Rgba8Unorm,
            images,
        )
    });

    (albedo_array, normal_array, pbr_array)
}

/// Creates a mipmapped texture array, one layer per handle.
///
/// Layers without a texture (`None`) are filled with `default_color`.
fn create_texture_array<'a>(
    layers: impl Iterator<Item = Option<&'a Handle<Image>>>,
    (width, height): (u32, u32),
    default_color: [u8; 4],
    format: TextureFormat,
    images: &mut Assets<Image>,
) -> Handle<Image> {
    let mip_count = calculate_mip_count(width, height);
    let default_data = default_color.repeat((width * height) as usize);

    let mut array_data = Vec::new();
    let mut num_layers = 0;
    for layer in layers {
        let data = match layer {
            None => Cow::Borrowed(default_data.as_slice()),
            Some(handle) => rgba8_bytes(images.get(handle).unwrap(), (width, height))
                .unwrap_or_else(|| {
                    warn!(
                        "plumesplat: layer {num_layers} texture {:?} must be {width}x{height}, \
                         8- or 16-bit RGBA, with CPU-side data; using a default color instead",
                        handle.path()
                    );
                    Cow::Borrowed(default_data.as_slice())
                }),
        };
        array_data.extend(generate_mipmaps(&data, width, height, mip_count));
        num_layers += 1;
    }

    images.add(Image {
        data: Some(array_data),
        texture_descriptor: TextureDescriptor {
            label: Some("splat_texture_array"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: num_layers,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
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
    })
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

/// Calculates the number of mip levels for a texture.
fn calculate_mip_count(width: u32, height: u32) -> u32 {
    (width.min(height) as f32).log2().floor() as u32 + 1
}

/// Generates mipmaps for a single texture layer.
fn generate_mipmaps(data: &[u8], width: u32, height: u32, mip_count: u32) -> Vec<u8> {
    let mut result = Vec::new();
    let mut current_data = data.to_vec();
    let mut current_width = width;
    let mut current_height = height;

    for mip in 0..mip_count {
        result.extend_from_slice(&current_data);

        if mip < mip_count - 1 {
            let next_width = (current_width / 2).max(1);
            let next_height = (current_height / 2).max(1);
            current_data = downsample_rgba(
                &current_data,
                current_width,
                current_height,
                next_width,
                next_height,
            );
            current_width = next_width;
            current_height = next_height;
        }
    }

    result
}

/// Downsamples an RGBA image using box filtering.
fn downsample_rgba(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Vec<u8> {
    let mut dst = vec![0u8; (dst_width * dst_height * 4) as usize];

    for dy in 0..dst_height {
        for dx in 0..dst_width {
            let sx = dx * 2;
            let sy = dy * 2;

            let mut r = 0u32;
            let mut g = 0u32;
            let mut b = 0u32;
            let mut a = 0u32;
            let mut count = 0u32;

            for oy in 0..2 {
                for ox in 0..2 {
                    let px = (sx + ox).min(src_width - 1);
                    let py = (sy + oy).min(src_height - 1);
                    let idx = ((py * src_width + px) * 4) as usize;

                    r += src[idx] as u32;
                    g += src[idx + 1] as u32;
                    b += src[idx + 2] as u32;
                    a += src[idx + 3] as u32;
                    count += 1;
                }
            }

            let dst_idx = ((dy * dst_width + dx) * 4) as usize;
            dst[dst_idx] = (r / count) as u8;
            dst[dst_idx + 1] = (g / count) as u8;
            dst[dst_idx + 2] = (b / count) as u8;
            dst[dst_idx + 3] = (a / count) as u8;
        }
    }

    dst
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
}
