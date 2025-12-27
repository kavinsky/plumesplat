//! PlumeSplat material definition.
//!
//! This module defines the custom material that enables splatting up to 256 textures
//! using texture arrays and per-vertex material indices/weights.

use bevy::asset::{Asset, Handle};
use bevy::image::Image;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipelineKey};
use bevy::prelude::Mesh;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, CompareFunction, RenderPipelineDescriptor, ShaderType,
    SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use crate::vertex_attributes::{ATTRIBUTE_MATERIAL_INDICES, ATTRIBUTE_MATERIAL_WEIGHTS};

/// Configuration for the PlumeSplat material.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlumeSplatConfig {
    /// UV scale for triplanar mapping (higher = more tiling)
    pub uv_scale: f32,
    /// Sharpness of triplanar blending (higher = sharper transitions at edges)
    pub triplanar_sharpness: f32,
    /// Enable height-based blending between materials
    pub height_blend: bool,
    /// Height blend sharpness (higher = sharper transitions)
    pub height_blend_sharpness: f32,
}

impl PlumeSplatConfig {
    /// Default configuration with reasonable values.
    pub fn new() -> Self {
        Self {
            uv_scale: 1.0,
            triplanar_sharpness: 4.0,
            height_blend: false,
            height_blend_sharpness: 8.0,
        }
    }
}

/// Material for terrain splatting with support for up to 256 textures.
///
/// This material uses texture arrays to store multiple textures and samples them
/// based on per-vertex material indices and weights. It supports triplanar mapping
/// for UV-less texturing.
///
/// ## Texture Arrays
///
/// The material expects texture arrays where each layer represents a different material:
/// - `albedo_array`: Base color textures (RGBA)
/// - `normal_array`: Normal maps (optional)
/// - `arm_array`: Ambient occlusion, Roughness, Metallic packed (optional)
///
/// ## Vertex Attributes
///
/// Meshes must include these custom vertex attributes:
/// - `ATTRIBUTE_MATERIAL_INDICES`: 4 material indices packed into u32
/// - `ATTRIBUTE_MATERIAL_WEIGHTS`: 4 blend weights packed into u32
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct PlumeSplatMaterial {
    /// Albedo/base color texture array. Each layer is a different material.
    #[texture(0, dimension = "2d_array")]
    #[sampler(1)]
    pub albedo_array: Option<Handle<Image>>,

    /// Normal map texture array. Each layer corresponds to an albedo layer.
    #[texture(2, dimension = "2d_array")]
    #[sampler(3)]
    pub normal_array: Option<Handle<Image>>,

    /// ARM (Ambient Occlusion, Roughness, Metallic) texture array.
    /// R = AO, G = Roughness, B = Metallic
    #[texture(4, dimension = "2d_array")]
    #[sampler(5)]
    pub arm_array: Option<Handle<Image>>,

    /// Material configuration uniforms.
    #[uniform(6)]
    pub uniforms: PlumeSplatUniforms,
}

/// Uniform data for the PlumeSplat shader.
#[derive(Debug, Clone, Copy, Default, ShaderType)]
pub struct PlumeSplatUniforms {
    /// UV scale for triplanar mapping
    pub uv_scale: f32,
    /// Triplanar blend sharpness
    pub triplanar_sharpness: f32,
    /// Height blend sharpness (0.0 to disable)
    pub height_blend_sharpness: f32,
    /// Base roughness when no ARM texture
    pub base_roughness: f32,
    /// Base metallic when no ARM texture
    pub base_metallic: f32,
}

impl Default for PlumeSplatMaterial {
    fn default() -> Self {
        Self {
            albedo_array: None,
            normal_array: None,
            arm_array: None,
            uniforms: PlumeSplatUniforms {
                uv_scale: 1.0,
                triplanar_sharpness: 4.0,
                height_blend_sharpness: 0.0,
                base_roughness: 0.8,
                base_metallic: 0.0,
            },
        }
    }
}

impl PlumeSplatMaterial {
    /// Create a new PlumeSplat material with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the albedo texture array.
    pub fn with_albedo(mut self, albedo: Handle<Image>) -> Self {
        self.albedo_array = Some(albedo);
        self
    }

    /// Set the normal map texture array.
    pub fn with_normals(mut self, normals: Handle<Image>) -> Self {
        self.normal_array = Some(normals);
        self
    }

    /// Set the ARM (AO/Roughness/Metallic) texture array.
    pub fn with_arm(mut self, arm: Handle<Image>) -> Self {
        self.arm_array = Some(arm);
        self
    }

    /// Set the UV scale for triplanar mapping.
    pub fn with_uv_scale(mut self, scale: f32) -> Self {
        self.uniforms.uv_scale = scale;
        self
    }

    /// Set the triplanar blend sharpness.
    pub fn with_triplanar_sharpness(mut self, sharpness: f32) -> Self {
        self.uniforms.triplanar_sharpness = sharpness;
        self
    }

    /// Enable height-based blending with specified sharpness.
    pub fn with_height_blending(mut self, sharpness: f32) -> Self {
        self.uniforms.height_blend_sharpness = sharpness;
        self
    }

    /// Set base roughness (used when no ARM texture).
    pub fn with_roughness(mut self, roughness: f32) -> Self {
        self.uniforms.base_roughness = roughness;
        self
    }

    /// Set base metallic (used when no ARM texture).
    pub fn with_metallic(mut self, metallic: f32) -> Self {
        self.uniforms.base_metallic = metallic;
        self
    }
}

impl Material for PlumeSplatMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://plumesplat/shaders/plumesplat.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://plumesplat/shaders/plumesplat.wgsl".into()
    }

    fn specialize(
        _pipeline: &bevy::pbr::MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Add our custom vertex attributes to the vertex buffer layout
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(2),
            ATTRIBUTE_MATERIAL_INDICES.at_shader_location(3),
            ATTRIBUTE_MATERIAL_WEIGHTS.at_shader_location(4),
        ])?;

        descriptor.vertex.buffers = vec![vertex_layout];

        // Enable depth testing
        if let Some(ref mut depth_stencil) = descriptor.depth_stencil {
            depth_stencil.depth_compare = CompareFunction::GreaterEqual;
            depth_stencil.depth_write_enabled = true;
        }

        Ok(())
    }
}
