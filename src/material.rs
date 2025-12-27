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
/// - `albedo_array`: Base color textures (RGBA)
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
    #[sampler(101)]
    pub albedo_array: Handle<Image>,

    /// Material configuration uniforms.
    #[uniform(102)]
    pub settings: PlumeSplatSettings,
}

/// Settings for the PlumeSplat shader.
#[derive(Debug, Clone, Copy, Default, ShaderType)]
pub struct PlumeSplatSettings {
    /// UV scale for triplanar mapping (higher = more tiling)
    pub uv_scale: f32,
    /// Triplanar blend sharpness (higher = sharper transitions at edges)
    pub triplanar_sharpness: f32,
    /// Height blend sharpness (0.0 to disable)
    pub height_blend_sharpness: f32,
}

impl Default for PlumeSplatExtension {
    fn default() -> Self {
        Self {
            albedo_array: Handle::default(),
            settings: PlumeSplatSettings {
                uv_scale: 1.0,
                triplanar_sharpness: 4.0, // Higher for sharper transitions on steep slopes
                height_blend_sharpness: 0.0,
            },
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
