//! PlumeSplat material definition.
//!
//! This module defines the material extension that enables splatting up to 256 textures
//! using texture arrays and per-vertex material indices/weights.

use bevy::asset::{Asset, Handle};
use bevy::image::Image;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{
    ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
    StandardMaterial,
};
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
    VertexAttribute, VertexFormat,
};
use bevy::shader::ShaderRef;

use crate::vertex_attributes::{ATTRIBUTE_MATERIAL_INDICES, ATTRIBUTE_MATERIAL_WEIGHTS};

/// Type alias for the full PlumeSplat material (StandardMaterial + PlumeSplatExtension)
pub type PlumeSplatMaterial = ExtendedMaterial<StandardMaterial, PlumeSplatExtension>;

/// Material extension for terrain splatting with support for up to 256 textures.
///
/// This extension adds texture array splatting on top of Bevy's StandardMaterial,
/// giving you full PBR lighting support while allowing custom texture blending.
///
/// ## Texture Arrays
///
/// The extension expects texture arrays where each layer represents a different material:
/// - `albedo_array`: Base color textures (RGBA, A can store height for blending)
/// - `normal_array`: Normal map textures (RGB, optional)
/// - `pbr_array`: Packed PBR textures (R=Metallic, G=Roughness, B=AO, A=Height, optional)
///
/// ## Vertex Attributes
///
/// Meshes must include these custom vertex attributes:
/// - `ATTRIBUTE_MATERIAL_INDICES`: 4 material indices packed into u32
/// - `ATTRIBUTE_MATERIAL_WEIGHTS`: 4 blend weights packed into u32
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct PlumeSplatExtension {
    /// Albedo/base color texture array. Each layer is a different material.
    #[texture(100, dimension = "2d_array")]
    #[sampler(101, sampler_type = "filtering")]
    pub albedo_array: Handle<Image>,

    /// Normal map texture array. Each layer corresponds to the same material index as albedo.
    /// Optional - if not provided, geometry normals will be used.
    #[texture(103, dimension = "2d_array")]
    #[sampler(104, sampler_type = "filtering")]
    pub normal_array: Option<Handle<Image>>,

    /// Packed PBR texture array. Each layer contains:
    /// - R: Metallic (0.0 = dielectric, 1.0 = metal)
    /// - G: Roughness (0.0 = smooth, 1.0 = rough)
    /// - B: Ambient Occlusion (0.0 = occluded, 1.0 = not occluded)
    /// - A: Height (for height-based blending, optional)
    /// Optional - if not provided, default PBR values will be used.
    #[texture(105, dimension = "2d_array")]
    #[sampler(106, sampler_type = "filtering")]
    pub pbr_array: Option<Handle<Image>>,

    /// Material configuration uniforms.
    #[uniform(102)]
    pub settings: PlumeSplatSettings,
}

/// Settings for the PlumeSplat shader.
#[derive(Debug, Clone, Copy, ShaderType)]
pub struct PlumeSplatSettings {
    /// UV scale for triplanar mapping (higher = more tiling)
    pub uv_scale: f32,
    /// Triplanar blend sharpness (higher = sharper transitions at edges)
    pub triplanar_sharpness: f32,
    /// Height blend sharpness (0.0 to disable)
    pub height_blend_sharpness: f32,
    /// Blend offset - subtracts from weights before normalization.
    /// Higher values create sharper transitions by reducing influence of weaker materials.
    /// Range: 0.0 to 0.5 (max safe value before weights can all become zero)
    pub blend_offset: f32,
    /// Blend exponent - raises weights to this power before normalization.
    /// Higher values create sharper transitions (1.0 = linear, 2.0+ = sharper)
    /// Range: 1.0 to 8.0
    pub blend_exponent: f32,
}

impl Default for PlumeSplatSettings {
    fn default() -> Self {
        Self {
            uv_scale: 1.0,
            triplanar_sharpness: 4.0,
            height_blend_sharpness: 0.0,
            blend_offset: 0.0,
            blend_exponent: 1.0,
        }
    }
}

impl Default for PlumeSplatExtension {
    fn default() -> Self {
        Self {
            albedo_array: Handle::default(),
            normal_array: None,
            pbr_array: None,
            settings: PlumeSplatSettings::default(),
        }
    }
}

impl PlumeSplatExtension {
    /// Create a new PlumeSplat extension with the given texture array.
    pub fn new(albedo_array: Handle<Image>) -> Self {
        Self {
            albedo_array,
            ..Default::default()
        }
    }

    /// Set the UV scale for triplanar mapping.
    pub fn with_uv_scale(mut self, scale: f32) -> Self {
        self.settings.uv_scale = scale;
        self
    }

    /// Set the triplanar blend sharpness.
    pub fn with_triplanar_sharpness(mut self, sharpness: f32) -> Self {
        self.settings.triplanar_sharpness = sharpness;
        self
    }

    /// Enable height-based blending with specified sharpness.
    pub fn with_height_blending(mut self, sharpness: f32) -> Self {
        self.settings.height_blend_sharpness = sharpness;
        self
    }

    /// Set the normal map texture array.
    pub fn with_normal_array(mut self, normal_array: Handle<Image>) -> Self {
        self.normal_array = Some(normal_array);
        self
    }

    /// Set the packed PBR texture array (R=Metallic, G=Roughness, B=AO, A=Height).
    pub fn with_pbr_array(mut self, pbr_array: Handle<Image>) -> Self {
        self.pbr_array = Some(pbr_array);
        self
    }

    /// Set the blend offset for material weight adjustment.
    ///
    /// Subtracts this value from all weights before normalization, causing
    /// weaker materials to have less influence. Higher values create sharper
    /// transitions between materials.
    ///
    /// Range: 0.0 (no offset) to 0.5 (maximum safe offset)
    /// Default: 0.0
    pub fn with_blend_offset(mut self, offset: f32) -> Self {
        self.settings.blend_offset = offset.clamp(0.0, 0.5);
        self
    }

    /// Set the blend exponent for material weight adjustment.
    ///
    /// Raises weights to this power before normalization. Higher values
    /// create sharper transitions between materials by emphasizing
    /// the dominant material.
    ///
    /// Range: 1.0 (linear) to 8.0 (very sharp)
    /// Default: 1.0
    pub fn with_blend_exponent(mut self, exponent: f32) -> Self {
        self.settings.blend_exponent = exponent.clamp(1.0, 8.0);
        self
    }
}

impl MaterialExtension for PlumeSplatExtension {
    fn vertex_shader() -> ShaderRef {
        "embedded://plumesplat/shaders/plumesplat_ext.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://plumesplat/shaders/plumesplat_ext.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Add our custom vertex attributes
        // Find indices for our custom attributes in the layout
        if let Some(index) = layout
            .0
            .attribute_ids()
            .iter()
            .position(|id| *id == ATTRIBUTE_MATERIAL_INDICES.id)
        {
            let attr = &layout.0.layout().attributes[index];
            descriptor.vertex.buffers[0]
                .attributes
                .push(VertexAttribute {
                    format: VertexFormat::Uint32,
                    offset: attr.offset,
                    shader_location: 10, // Custom location for material indices
                });
        }

        if let Some(index) = layout
            .0
            .attribute_ids()
            .iter()
            .position(|id| *id == ATTRIBUTE_MATERIAL_WEIGHTS.id)
        {
            let attr = &layout.0.layout().attributes[index];
            descriptor.vertex.buffers[0]
                .attributes
                .push(VertexAttribute {
                    format: VertexFormat::Uint32,
                    offset: attr.offset,
                    shader_location: 11, // Custom location for material weights
                });
        }

        Ok(())
    }
}
