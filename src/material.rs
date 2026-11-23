//! PlumeSplat material definition and configuration.
//!
//! This module provides the core material types for terrain splatting:
//!
//! - [`PlumeSplatMaterial`]: Type alias for the full extended material
//! - [`PlumeSplatExtension`]: The material extension with texture arrays and settings
//! - [`PlumeSplatSettings`]: Shader uniform configuration
//!
//! # Overview
//!
//! PlumeSplat extends Bevy's [`StandardMaterial`] to add texture array splatting
//! while retaining full PBR lighting support. This is accomplished using Bevy's
//! [`ExtendedMaterial`] system.
//!
//! # Example
//!
//! ```rust,no_run
//! use bevy::prelude::*;
//! use plumesplat::prelude::*;
//!
//! fn create_terrain_material(
//!     asset_server: &Res<AssetServer>,
//! ) -> PlumeSplatMaterial {
//!     let albedo = asset_server.load("textures/terrain_albedo.png");
//!     let normals = asset_server.load("textures/terrain_normal.png");
//!     let pbr = asset_server.load("textures/terrain_pbr.png");
//!
//!     PlumeSplatMaterial {
//!         base: StandardMaterial {
//!             perceptual_roughness: 0.8,
//!             metallic: 0.0,
//!             ..default()
//!         },
//!         extension: PlumeSplatExtension::new(albedo)
//!             .with_normal_array(normals)
//!             .with_pbr_array(pbr)
//!             .with_uv_scale(2.0)
//!             .with_triplanar_sharpness(4.0)
//!             .with_blend_offset(0.1)
//!             .with_blend_exponent(2.0),
//!     }
//! }
//! ```

use bevy::asset::{Asset, Handle};
use bevy::image::Image;
use bevy::math::Vec4;
use bevy::mesh::{Mesh, MeshVertexBufferLayoutRef};
use bevy::pbr::{
    ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
    StandardMaterial,
};
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use crate::vertex_attributes::{ATTRIBUTE_MATERIAL_INDICES, ATTRIBUTE_MATERIAL_WEIGHTS};

/// Type alias for the complete PlumeSplat material.
///
/// This combines Bevy's [`StandardMaterial`] (for PBR lighting) with
/// [`PlumeSplatExtension`] (for texture array splatting).
///
/// # Usage
///
/// ```rust,no_run
/// use bevy::prelude::*;
/// use plumesplat::prelude::*;
///
/// fn setup(mut materials: ResMut<Assets<PlumeSplatMaterial>>) {
///     let material = PlumeSplatMaterial {
///         base: StandardMaterial::default(),
///         extension: PlumeSplatExtension::default(),
///     };
///     let handle = materials.add(material);
/// }
/// ```
pub type PlumeSplatMaterial = ExtendedMaterial<StandardMaterial, PlumeSplatExtension>;

/// Material extension that adds texture array splatting to StandardMaterial.
///
/// This extension enables blending up to 256 different materials using texture arrays
/// and per-vertex material indices/weights. It supports:
///
/// - **Albedo textures**: Base color with optional height in alpha channel
/// - **Normal maps**: Per-material tangent-space normal maps
/// - **PBR textures**: Packed metallic, roughness, AO, and height maps
/// - **Triplanar mapping**: UV-less texturing for arbitrary geometry
/// - **Stochastic tiling**: Eliminates visible texture repetition
///
/// # Texture Array Format
///
/// Each handle must point to a 2D **array** image, one layer per material. The
/// [`SplatMaterialBuilder`](crate::SplatMaterialBuilder) builds these for you. To load a
/// vertically stacked PNG yourself, tell the loader to split it into layers (and to keep
/// normal / PBR data linear):
///
/// ```rust,no_run
/// use bevy::image::{ImageArrayLayout, ImageLoaderSettings};
/// use bevy::prelude::*;
///
/// fn load(asset_server: Res<AssetServer>) {
///     let albedo: Handle<Image> = asset_server
///         .load_builder()
///         .with_settings(|s: &mut ImageLoaderSettings| {
///             s.array_layout = Some(ImageArrayLayout::RowCount { rows: 4 });
///         })
///         .load("terrain_albedo_array.png");
///     let normals: Handle<Image> = asset_server
///         .load_builder()
///         .with_settings(|s: &mut ImageLoaderSettings| {
///             s.array_layout = Some(ImageArrayLayout::RowCount { rows: 4 });
///             s.is_srgb = false;
///         })
///         .load("terrain_normal_array.png");
/// }
/// ```
///
/// Images loaded this way have no mipmaps, so distant terrain will shimmer; prefer the
/// builder unless the arrays come with their own mips (e.g. KTX2).
///
/// Channel layout:
///
/// | Texture | Channel Layout | Description |
/// |---------|----------------|-------------|
/// | `albedo_array` | RGBA | RGB = color, A = height (optional) |
/// | `normal_array` | RGB | Tangent-space normals |
/// | `pbr_array` | RGBA | R = metallic, G = roughness, B = AO, A = height |
///
/// # Vertex Attributes
///
/// Meshes using this material must include custom vertex attributes:
///
/// - [`ATTRIBUTE_MATERIAL_INDICES`](crate::ATTRIBUTE_MATERIAL_INDICES): 4 material indices packed into u32
/// - [`ATTRIBUTE_MATERIAL_WEIGHTS`](crate::ATTRIBUTE_MATERIAL_WEIGHTS): 4 blend weights packed into u32
///
/// See [`MaterialVertex`](crate::MaterialVertex) for helpers to create these.
///
/// # Example
///
/// ```rust,no_run
/// use bevy::prelude::*;
/// use plumesplat::prelude::*;
///
/// fn create_extension(asset_server: &Res<AssetServer>) -> PlumeSplatExtension {
///     let albedo = asset_server.load("terrain_albedo.png");
///
///     PlumeSplatExtension::new(albedo)
///         .with_uv_scale(2.0)           // Texture tiling
///         .with_triplanar_sharpness(4.0) // Edge sharpness
///         .with_blend_offset(0.1)        // Sharper transitions
///         .with_blend_exponent(2.0)      // Even sharper
/// }
/// ```
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone, Default)]
#[bind_group_data(PlumeSplatKey)]
pub struct PlumeSplatExtension {
    /// Albedo (base color) texture array.
    ///
    /// Each layer is a different material's color texture. The alpha channel
    /// can optionally store height values for height-based blending.
    ///
    /// **Required** - This must be set for the material to render correctly.
    #[texture(100, dimension = "2d_array")]
    #[sampler(101, sampler_type = "filtering")]
    pub albedo_array: Handle<Image>,

    /// Normal map texture array.
    ///
    /// Each layer contains tangent-space normal maps for the corresponding
    /// material index. Each layer is projected with whiteout triplanar blending,
    /// then the world-space normals are blended by layer weight and renormalized.
    ///
    /// **Optional** - If not provided, geometry normals are used.
    #[texture(103, dimension = "2d_array")]
    #[sampler(104, sampler_type = "filtering")]
    pub normal_array: Option<Handle<Image>>,

    /// Packed PBR texture array.
    ///
    /// Each layer contains packed PBR properties for physically-based rendering:
    ///
    /// | Channel | Property | Range | Description |
    /// |---------|----------|-------|-------------|
    /// | R | Metallic | 0.0-1.0 | 0 = dielectric, 1 = metal |
    /// | G | Roughness | 0.0-1.0 | 0 = smooth/glossy, 1 = rough/matte |
    /// | B | Ambient Occlusion | 0.0-1.0 | 0 = fully occluded, 1 = no occlusion |
    /// | A | Height | 0.0-1.0 | Used for height-based blending |
    ///
    /// **Optional** - If not provided, metallic and roughness come from the base
    /// `StandardMaterial`, and height blending uses albedo alpha.
    #[texture(105, dimension = "2d_array")]
    #[sampler(106, sampler_type = "filtering")]
    pub pbr_array: Option<Handle<Image>>,

    /// Shader configuration settings.
    ///
    /// Controls UV scaling, blending behavior, and triplanar mapping parameters.
    /// See [`PlumeSplatSettings`] for details on each setting.
    #[uniform(102)]
    pub settings: PlumeSplatSettings,

    /// Per-layer overrides of global settings (currently stochastic rotation).
    ///
    /// Every layer inherits the global value from [`settings`](Self::settings)
    /// unless it has an override. The builder fills this from
    /// [`MaterialLayer::with_rotation`](crate::MaterialLayer::with_rotation);
    /// without the builder use [`with_layer_rotation`](Self::with_layer_rotation).
    #[uniform(107)]
    pub layer_settings: PlumeSplatLayerSettings,
}

/// Number of layers a material can address (material indices are `u8`).
const LAYER_COUNT: usize = 256;

/// Marker stored for "no override, use the global setting".
const INHERIT: f32 = -1.0;

/// Per-layer overrides of global [`PlumeSplatSettings`].
///
/// Each layer either **inherits** the global value (the default) or has its
/// own **override**. Because inheritance is resolved in the shader, changing
/// the global setting at runtime still affects every inheriting layer.
///
/// Currently holds the stochastic rotation override; more per-layer settings
/// will use the same inherit / override rule.
///
/// # Example
///
/// ```rust
/// use plumesplat::PlumeSplatLayerSettings;
///
/// let mut layers = PlumeSplatLayerSettings::default();
/// layers.set_rotation(2, Some(0.0)); // layer 2: never rotated
/// assert_eq!(layers.rotation(2), Some(0.0));
/// assert_eq!(layers.rotation(0), None); // layer 0: inherits the global value
/// ```
#[derive(Debug, Clone, PartialEq, ShaderType)]
pub struct PlumeSplatLayerSettings {
    // One entry per layer. x = stochastic rotation override (negative =
    // inherit); y, z, w reserved for future per-layer settings (also inherit).
    layers: [Vec4; LAYER_COUNT],
}

impl Default for PlumeSplatLayerSettings {
    fn default() -> Self {
        Self {
            layers: [Vec4::splat(INHERIT); LAYER_COUNT],
        }
    }
}

impl PlumeSplatLayerSettings {
    /// Returns the stochastic rotation override of `layer`, or `None` if it
    /// inherits [`PlumeSplatSettings::stochastic_rotation`].
    #[must_use]
    pub fn rotation(&self, layer: u8) -> Option<f32> {
        let value = self.layers[layer as usize].x;
        (value >= 0.0).then_some(value)
    }

    /// Sets the stochastic rotation override of `layer`.
    ///
    /// `Some(strength)` overrides the global value (clamped to 0.0-1.0;
    /// 0.0 = never rotated). `None` makes the layer inherit the global value again.
    pub fn set_rotation(&mut self, layer: u8, strength: Option<f32>) {
        self.layers[layer as usize].x = strength.map_or(INHERIT, |s| s.clamp(0.0, 1.0));
    }
}

/// Pipeline key: which optional arrays are bound, so the shader skips the missing ones.
#[doc(hidden)]
#[repr(C)]
#[derive(Debug, Eq, PartialEq, Hash, Copy, Clone)]
pub struct PlumeSplatKey {
    has_normal_array: bool,
    has_pbr_array: bool,
}

impl From<&PlumeSplatExtension> for PlumeSplatKey {
    fn from(ext: &PlumeSplatExtension) -> Self {
        Self {
            has_normal_array: ext.normal_array.is_some(),
            has_pbr_array: ext.pbr_array.is_some(),
        }
    }
}

/// Shader uniform settings for PlumeSplat rendering.
///
/// These settings control how textures are sampled and blended in the shader.
/// All settings have sensible defaults but can be tuned for different visual effects.
///
/// # Example
///
/// ```rust
/// use plumesplat::PlumeSplatSettings;
///
/// let settings = PlumeSplatSettings {
///     uv_scale: 2.0,              // 2x texture tiling
///     triplanar_sharpness: 8.0,   // Sharp triplanar edges
///     height_blend_sharpness: 0.5, // Enable height blending
///     blend_offset: 0.1,          // Sharper material transitions
///     blend_exponent: 2.0,        // Squared weights
///     far_scale: 0.11,            // Far sample ~9x larger
///     far_mix: 0.6,               // Large-scale variation strength
///     stochastic_rotation: 0.0,   // Random tile rotation (off)
///     lod_near: 0.0,              // Distance LOD fade start
///     lod_far: 0.0,               // Distance LOD fade end (0 = off)
/// };
/// ```
#[derive(Debug, Clone, Copy, ShaderType)]
pub struct PlumeSplatSettings {
    /// UV scale multiplier for texture sampling.
    ///
    /// Higher values result in more texture tiling (smaller apparent texture size).
    /// Lower values stretch the texture over a larger area.
    ///
    /// - **Default**: 1.0
    /// - **Range**: 0.1 to 10.0+ (no hard limit)
    /// - **Example**: 2.0 = textures tile twice as often
    pub uv_scale: f32,

    /// Triplanar mapping blend sharpness.
    ///
    /// Controls how sharply textures blend at geometry edges when using
    /// triplanar mapping. Higher values create sharper transitions between
    /// the X, Y, and Z projection planes.
    ///
    /// - **Default**: 4.0
    /// - **Range**: 1.0 (soft) to 16.0 (very sharp)
    pub triplanar_sharpness: f32,

    /// Height-based blend sharpness.
    ///
    /// When greater than 0, enables height-based material blending where
    /// materials with higher height values "poke through" at transitions.
    /// This creates more natural-looking material boundaries (e.g., rocks
    /// poking through grass).
    ///
    /// - **Default**: 0.0 (disabled)
    /// - **Range**: 0.0 (disabled) to 2.0+ (sharp height transitions)
    pub height_blend_sharpness: f32,

    /// Blend weight offset for sharper material transitions.
    ///
    /// Subtracts this value from all blend weights before normalization.
    /// This reduces the influence of materials with low weights, creating
    /// sharper boundaries between dominant materials.
    ///
    /// - **Default**: 0.0 (no offset)
    /// - **Range**: 0.0 to 0.5
    /// - **Warning**: Values above 0.5 may cause artifacts if all weights become zero
    pub blend_offset: f32,

    /// Blend weight exponent for sharper material transitions.
    ///
    /// Raises blend weights to this power before normalization. Higher values
    /// emphasize the dominant material and reduce blending at boundaries.
    ///
    /// - **Default**: 1.0 (linear blending)
    /// - **Range**: 1.0 (linear) to 8.0 (very sharp)
    /// - **Example**: 2.0 = quadratic falloff, 3.0 = cubic falloff
    pub blend_exponent: f32,

    /// Scale of the far (large-scale) albedo sample, relative to `uv_scale`.
    ///
    /// Multi-scale sampling samples each layer's albedo a second time at this
    /// smaller scale (so the texture appears larger) and uses it to brighten
    /// or darken the normal sample. This breaks up visible tiling at distance.
    ///
    /// - **Default**: 0.11 (the far texture is ~9x larger)
    /// - **Range**: 0.01 to 1.0. Avoid simple fractions like 0.5 or 0.25: the two
    ///   scales then tile in step and the repetition stays visible.
    pub far_scale: f32,

    /// Strength of the far albedo sample.
    ///
    /// - **Default**: 0.6
    /// - **Range**: 0.0 (off; the far sample is skipped entirely) to 1.0 (full)
    pub far_mix: f32,

    /// Random per-tile rotation in stochastic tiling.
    ///
    /// Each stochastic tile gets a random rotation on top of its random offset,
    /// which hides repetition much better for textures without a direction
    /// (grass, dirt, sand, gravel). Normal maps are rotated to match, so
    /// lighting stays correct. Costs a `sin`/`cos` per texture tap, no extra
    /// texture samples.
    ///
    /// This is the global value. Textures with a clear direction (layered rock,
    /// planks, baked directional shadows) look wrong when rotated, so turn it
    /// off for those layers with a per-layer override
    /// ([`MaterialLayer::with_rotation`](crate::MaterialLayer::with_rotation) or
    /// [`PlumeSplatExtension::with_layer_rotation`]).
    ///
    /// - **Default**: 0.0 (off)
    /// - **Range**: 0.0 (off) to 1.0 (any angle). 0.5 limits rotation to ±90°.
    pub stochastic_rotation: f32,

    /// Distance (world units, from the camera) where distance-based LOD starts
    /// to fade in. See [`lod_far`](Self::lod_far).
    ///
    /// - **Default**: 0.0
    pub lod_near: f32,

    /// Distance (world units, from the camera) where distance-based LOD is
    /// fully applied. Set to 0.0 to turn LOD off.
    ///
    /// Between `lod_near` and `lod_far` the effect fades in smoothly, so there
    /// is no popping. Beyond `lod_far`, normal maps are faded to the geometry
    /// normal and no longer sampled (cheaper, and less specular shimmer), and
    /// triplanar blending is sharpened so fewer projection planes are sampled.
    /// Stochastic tiling stays on at every distance.
    ///
    /// - **Default**: 0.0 (off)
    /// - **Range**: 0.0 (off), or greater than `lod_near`
    pub lod_far: f32,
}

impl Default for PlumeSplatSettings {
    fn default() -> Self {
        Self {
            uv_scale: 1.0,
            triplanar_sharpness: 4.0,
            height_blend_sharpness: 0.0,
            blend_offset: 0.0,
            blend_exponent: 1.0,
            far_scale: 0.11,
            far_mix: 0.6,
            stochastic_rotation: 0.0,
            lod_near: 0.0,
            lod_far: 0.0,
        }
    }
}

impl PlumeSplatSettings {
    /// Sets multi-scale albedo sampling, clamping both values to their ranges.
    /// Shared by the extension and builder `with_multi_scale` methods.
    pub(crate) fn set_multi_scale(&mut self, far_scale: f32, mix: f32) {
        self.far_scale = far_scale.clamp(0.01, 1.0);
        self.far_mix = mix.clamp(0.0, 1.0);
    }

    /// Sets distance-based LOD. `near` is clamped to >= 0 and `far` to at least
    /// `near + 0.01`, so the fade band is never empty. Shared by the extension
    /// and builder `with_distance_lod` methods.
    pub(crate) fn set_distance_lod(&mut self, near: f32, far: f32) {
        self.lod_near = near.max(0.0);
        self.lod_far = far.max(self.lod_near + 0.01);
    }
}

impl PlumeSplatExtension {
    /// Creates a new PlumeSplat extension with the given albedo texture array.
    ///
    /// # Arguments
    ///
    /// * `albedo_array` - Handle to the albedo 2D array image (see "Texture Array Format" above)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use bevy::prelude::*;
    /// use plumesplat::PlumeSplatExtension;
    ///
    /// fn setup(asset_server: Res<AssetServer>) {
    ///     let albedo = asset_server.load("terrain_albedo.png");
    ///     let extension = PlumeSplatExtension::new(albedo);
    /// }
    /// ```
    pub fn new(albedo_array: Handle<Image>) -> Self {
        Self {
            albedo_array,
            ..Default::default()
        }
    }

    /// Sets the UV scale for texture sampling.
    ///
    /// Higher values increase texture tiling density.
    ///
    /// # Arguments
    ///
    /// * `scale` - UV multiplier (default: 1.0)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use plumesplat::PlumeSplatExtension;
    /// # let albedo = bevy::asset::Handle::default();
    /// let ext = PlumeSplatExtension::new(albedo)
    ///     .with_uv_scale(2.0);  // 2x tiling
    /// ```
    pub fn with_uv_scale(mut self, scale: f32) -> Self {
        self.settings.uv_scale = scale;
        self
    }

    /// Sets the triplanar mapping blend sharpness.
    ///
    /// Controls how sharply the three projection planes blend at edges.
    ///
    /// # Arguments
    ///
    /// * `sharpness` - Blend exponent (default: 4.0, range: 1.0-16.0)
    pub fn with_triplanar_sharpness(mut self, sharpness: f32) -> Self {
        self.settings.triplanar_sharpness = sharpness;
        self
    }

    /// Enables height-based material blending.
    ///
    /// When enabled, materials with higher height values (from albedo alpha
    /// or PBR texture) will "poke through" at material transitions, creating
    /// more natural-looking boundaries.
    ///
    /// # Arguments
    ///
    /// * `sharpness` - Height blend sharpness (0.0 = disabled, default: 0.0)
    pub fn with_height_blending(mut self, sharpness: f32) -> Self {
        self.settings.height_blend_sharpness = sharpness;
        self
    }

    /// Sets the normal map texture array.
    ///
    /// # Arguments
    ///
    /// * `normal_array` - Handle to normal map texture array
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use bevy::prelude::*;
    /// use plumesplat::PlumeSplatExtension;
    ///
    /// fn setup(asset_server: Res<AssetServer>) {
    ///     let albedo = asset_server.load("terrain_albedo.png");
    ///     let normals = asset_server.load("terrain_normal.png");
    ///
    ///     let ext = PlumeSplatExtension::new(albedo)
    ///         .with_normal_array(normals);
    /// }
    /// ```
    pub fn with_normal_array(mut self, normal_array: Handle<Image>) -> Self {
        self.normal_array = Some(normal_array);
        self
    }

    /// Sets the packed PBR texture array.
    ///
    /// The PBR texture should have channels packed as:
    /// - R: Metallic (0-1)
    /// - G: Roughness (0-1)
    /// - B: Ambient Occlusion (0-1)
    /// - A: Height (0-1, for height-based blending)
    ///
    /// # Arguments
    ///
    /// * `pbr_array` - Handle to packed PBR texture array
    pub fn with_pbr_array(mut self, pbr_array: Handle<Image>) -> Self {
        self.pbr_array = Some(pbr_array);
        self
    }

    /// Sets the blend offset for material weight adjustment.
    ///
    /// Subtracts this value from all weights before normalization, causing
    /// materials with lower weights to have reduced influence. This creates
    /// sharper transitions between materials.
    ///
    /// # Arguments
    ///
    /// * `offset` - Weight offset (clamped to 0.0-0.5, default: 0.0)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use plumesplat::PlumeSplatExtension;
    /// # let albedo = bevy::asset::Handle::default();
    /// // Create sharper material boundaries
    /// let ext = PlumeSplatExtension::new(albedo)
    ///     .with_blend_offset(0.15);
    /// ```
    pub fn with_blend_offset(mut self, offset: f32) -> Self {
        self.settings.blend_offset = offset.clamp(0.0, 0.5);
        self
    }

    /// Sets the blend exponent for material weight adjustment.
    ///
    /// Raises weights to this power before normalization. Higher values
    /// create sharper transitions by emphasizing the dominant material.
    ///
    /// # Arguments
    ///
    /// * `exponent` - Weight exponent (clamped to 1.0-8.0, default: 1.0)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use plumesplat::PlumeSplatExtension;
    /// # let albedo = bevy::asset::Handle::default();
    /// // Use quadratic weight falloff for sharper boundaries
    /// let ext = PlumeSplatExtension::new(albedo)
    ///     .with_blend_exponent(2.0);
    /// ```
    pub fn with_blend_exponent(mut self, exponent: f32) -> Self {
        self.settings.blend_exponent = exponent.clamp(1.0, 8.0);
        self
    }

    /// Configures multi-scale albedo sampling.
    ///
    /// Each layer's albedo is sampled a second time at `far_scale` (relative to
    /// `uv_scale`) and used to modulate the normal sample, breaking up visible
    /// tiling at distance. Set `mix` to 0.0 to disable it and skip the extra samples.
    ///
    /// # Arguments
    ///
    /// * `far_scale` - Far sample scale (clamped to 0.01-1.0, default: 0.11)
    /// * `mix` - Strength (clamped to 0.0-1.0, default: 0.6)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use plumesplat::PlumeSplatExtension;
    /// # let albedo = bevy::asset::Handle::default();
    /// // Subtler, larger-scale variation
    /// let ext = PlumeSplatExtension::new(albedo)
    ///     .with_multi_scale(0.07, 0.4);
    /// ```
    pub fn with_multi_scale(mut self, far_scale: f32, mix: f32) -> Self {
        self.settings.set_multi_scale(far_scale, mix);
        self
    }

    /// Enables random per-tile rotation in stochastic tiling.
    ///
    /// Hides repetition much better for textures without a direction (grass,
    /// dirt, sand). This is the global value, used by every layer without an
    /// override; see [`with_layer_rotation`](Self::with_layer_rotation).
    ///
    /// # Arguments
    ///
    /// * `strength` - 0.0 (off, default) to 1.0 (any angle); clamped to that range
    ///
    /// # Example
    ///
    /// ```rust
    /// # use plumesplat::PlumeSplatExtension;
    /// # let albedo = bevy::asset::Handle::default();
    /// let ext = PlumeSplatExtension::new(albedo)
    ///     .with_stochastic_rotation(1.0);
    /// ```
    pub fn with_stochastic_rotation(mut self, strength: f32) -> Self {
        self.settings.stochastic_rotation = strength.clamp(0.0, 1.0);
        self
    }

    /// Overrides the stochastic rotation of one layer (by texture array index).
    ///
    /// The layer uses `strength` instead of the global
    /// [`stochastic_rotation`](PlumeSplatSettings::stochastic_rotation).
    /// Use 0.0 for textures with a clear direction. With the builder, use
    /// [`MaterialLayer::with_rotation`](crate::MaterialLayer::with_rotation) instead.
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index in the texture arrays
    /// * `strength` - 0.0 (never rotated) to 1.0 (any angle); clamped to that range
    ///
    /// # Example
    ///
    /// ```rust
    /// # use plumesplat::PlumeSplatExtension;
    /// # let albedo = bevy::asset::Handle::default();
    /// let ext = PlumeSplatExtension::new(albedo)
    ///     .with_stochastic_rotation(1.0) // every layer...
    ///     .with_layer_rotation(2, 0.0);  // ...except layer 2
    /// ```
    pub fn with_layer_rotation(mut self, layer: u8, strength: f32) -> Self {
        self.layer_settings.set_rotation(layer, Some(strength));
        self
    }

    /// Enables distance-based LOD: cheaper shading far from the camera.
    ///
    /// Between `near` and `far` (world units from the camera) the effect fades
    /// in smoothly. Beyond `far`, normal maps fade to the geometry normal and
    /// are no longer sampled, and triplanar blending is sharpened. Off by default.
    /// To turn it off again, set [`lod_far`](PlumeSplatSettings::lod_far) to 0.0.
    ///
    /// # Arguments
    ///
    /// * `near` - Fade start distance (clamped to >= 0.0)
    /// * `far` - Fade end distance (clamped to at least `near + 0.01`)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use plumesplat::PlumeSplatExtension;
    /// # let albedo = bevy::asset::Handle::default();
    /// let ext = PlumeSplatExtension::new(albedo)
    ///     .with_distance_lod(40.0, 80.0);
    /// ```
    pub fn with_distance_lod(mut self, near: f32, far: f32) -> Self {
        self.settings.set_distance_lod(near, far);
        self
    }
}

impl MaterialExtension for PlumeSplatExtension {
    fn vertex_shader() -> ShaderRef {
        "embedded://plumesplat/shaders/plumesplat_ext.wesl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://plumesplat/shaders/plumesplat_ext.wesl".into()
    }

    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Missing optional arrays are bound to a white fallback image, so the
        // shader must not sample them.
        if let Some(fragment) = descriptor.fragment.as_mut() {
            if key.bind_group_data.has_normal_array {
                fragment.shader_defs.push("HAS_NORMAL_ARRAY".into());
            }
            if key.bind_group_data.has_pbr_array {
                fragment.shader_defs.push("HAS_PBR_ARRAY".into());
            }
        }

        // Add the splat attributes (u32 each, 4 packed u8 values) to Bevy's mesh
        // vertex buffer at locations 10/11, clear of Bevy's built-in locations.
        // `get_layout` returns a clear "Mesh is missing requested attribute"
        // error for meshes without them (or without normals, which the shader needs).
        let splat = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            ATTRIBUTE_MATERIAL_INDICES.at_shader_location(10),
            ATTRIBUTE_MATERIAL_WEIGHTS.at_shader_location(11),
        ])?;
        descriptor.vertex.buffers[0]
            .attributes
            .extend(splat.attributes.into_iter().filter(|a| a.shader_location >= 10));

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_settings() {
        let settings = PlumeSplatSettings::default();
        assert_eq!(settings.uv_scale, 1.0);
        assert_eq!(settings.triplanar_sharpness, 4.0);
        assert_eq!(settings.height_blend_sharpness, 0.0);
        assert_eq!(settings.blend_offset, 0.0);
        assert_eq!(settings.blend_exponent, 1.0);
        assert_eq!(settings.far_scale, 0.11);
        assert_eq!(settings.far_mix, 0.6);
        assert_eq!(settings.stochastic_rotation, 0.0);
        assert_eq!(settings.lod_near, 0.0);
        assert_eq!(settings.lod_far, 0.0); // LOD off
    }

    #[test]
    fn test_distance_lod_clamping() {
        let ext = PlumeSplatExtension::default().with_distance_lod(40.0, 80.0);
        assert_eq!(ext.settings.lod_near, 40.0);
        assert_eq!(ext.settings.lod_far, 80.0);

        let ext = PlumeSplatExtension::default().with_distance_lod(-5.0, -1.0);
        assert_eq!(ext.settings.lod_near, 0.0);
        assert_eq!(ext.settings.lod_far, 0.01); // never off or empty once set

        let ext = PlumeSplatExtension::default().with_distance_lod(50.0, 20.0);
        assert_eq!(ext.settings.lod_near, 50.0);
        assert!(ext.settings.lod_far > 50.0);
    }

    #[test]
    fn test_layer_rotation_inherits_by_default() {
        let layers = PlumeSplatLayerSettings::default();
        assert!((0..=255).all(|i| layers.rotation(i).is_none()));
    }

    #[test]
    fn test_layer_rotation_override_and_reset() {
        let ext = PlumeSplatExtension::default()
            .with_layer_rotation(2, 0.0)
            .with_layer_rotation(255, 2.0); // clamped
        assert_eq!(ext.layer_settings.rotation(2), Some(0.0));
        assert_eq!(ext.layer_settings.rotation(255), Some(1.0));
        assert_eq!(ext.layer_settings.rotation(0), None);

        let mut layers = ext.layer_settings;
        layers.set_rotation(2, None);
        assert_eq!(layers.rotation(2), None);

        // Negative strengths clamp to 0.0 (an override), not to "inherit".
        layers.set_rotation(3, Some(-1.0));
        assert_eq!(layers.rotation(3), Some(0.0));
    }

    #[test]
    fn test_stochastic_rotation_clamping() {
        let ext = PlumeSplatExtension::default().with_stochastic_rotation(0.5);
        assert_eq!(ext.settings.stochastic_rotation, 0.5);

        let ext = PlumeSplatExtension::default().with_stochastic_rotation(2.0);
        assert_eq!(ext.settings.stochastic_rotation, 1.0);

        let ext = PlumeSplatExtension::default().with_stochastic_rotation(-1.0);
        assert_eq!(ext.settings.stochastic_rotation, 0.0);
    }

    #[test]
    fn test_multi_scale_clamping() {
        let ext = PlumeSplatExtension::default().with_multi_scale(0.07, 0.4);
        assert_eq!(ext.settings.far_scale, 0.07);
        assert_eq!(ext.settings.far_mix, 0.4);

        let ext = PlumeSplatExtension::default().with_multi_scale(0.0, -1.0);
        assert_eq!(ext.settings.far_scale, 0.01);
        assert_eq!(ext.settings.far_mix, 0.0);

        let ext = PlumeSplatExtension::default().with_multi_scale(5.0, 2.0);
        assert_eq!(ext.settings.far_scale, 1.0);
        assert_eq!(ext.settings.far_mix, 1.0);
    }

    #[test]
    fn test_blend_offset_clamping() {
        let ext = PlumeSplatExtension::default()
            .with_blend_offset(1.0); // Above max
        assert_eq!(ext.settings.blend_offset, 0.5);

        let ext = PlumeSplatExtension::default()
            .with_blend_offset(-0.5); // Below min
        assert_eq!(ext.settings.blend_offset, 0.0);
    }

    #[test]
    fn test_blend_exponent_clamping() {
        let ext = PlumeSplatExtension::default()
            .with_blend_exponent(10.0); // Above max
        assert_eq!(ext.settings.blend_exponent, 8.0);

        let ext = PlumeSplatExtension::default()
            .with_blend_exponent(0.5); // Below min
        assert_eq!(ext.settings.blend_exponent, 1.0);
    }

    #[test]
    fn test_builder_chain() {
        let ext = PlumeSplatExtension::default()
            .with_uv_scale(2.0)
            .with_triplanar_sharpness(8.0)
            .with_height_blending(0.5)
            .with_blend_offset(0.2)
            .with_blend_exponent(3.0);

        assert_eq!(ext.settings.uv_scale, 2.0);
        assert_eq!(ext.settings.triplanar_sharpness, 8.0);
        assert_eq!(ext.settings.height_blend_sharpness, 0.5);
        assert_eq!(ext.settings.blend_offset, 0.2);
        assert_eq!(ext.settings.blend_exponent, 3.0);
    }
}
