//! Vertex attributes for material splatting.
//!
//! This module provides custom vertex attributes for storing material indices and blend weights.
//! Unlike traditional splatting which is limited to 4 materials via RGBA weights, PlumeSplat
//! uses a two-attribute approach:
//!
//! - `ATTRIBUTE_MATERIAL_INDICES`: Stores up to 4 material indices (0-255 each) packed into a u32
//! - `ATTRIBUTE_MATERIAL_WEIGHTS`: Stores the blend weights for those 4 materials packed into a u32
//!
//! This allows blending between any 4 materials from a palette of 256.

use bevy::mesh::MeshVertexAttribute;
use bevy::render::render_resource::VertexFormat;

/// Custom vertex attribute storing 4 material indices packed into a u32.
///
/// Each byte represents one material index (0-255), allowing selection from a 256-texture array.
/// Format: `[index0, index1, index2, index3]` packed as `index0 | (index1 << 8) | (index2 << 16) | (index3 << 24)`
pub const ATTRIBUTE_MATERIAL_INDICES: MeshVertexAttribute =
    MeshVertexAttribute::new("MaterialIndices", 988540910, VertexFormat::Uint32);

/// Custom vertex attribute storing 4 blend weights packed into a u32.
///
/// Each byte represents a weight (0-255, normalized to 0.0-1.0 in shader).
/// Format: `[weight0, weight1, weight2, weight3]` packed as `w0 | (w1 << 8) | (w2 << 16) | (w3 << 24)`
///
/// Weights should sum to 255 for proper blending, but the shader will normalize them.
pub const ATTRIBUTE_MATERIAL_WEIGHTS: MeshVertexAttribute =
    MeshVertexAttribute::new("MaterialWeights", 988540911, VertexFormat::Uint32);

/// Represents per-vertex material data for splatting.
#[derive(Debug, Clone, Copy, Default)]
pub struct MaterialVertex {
    /// Up to 4 material indices (index into texture array)
    pub indices: [u8; 4],
    /// Blend weights for each material (should sum to ~255)
    pub weights: [u8; 4],
}

impl MaterialVertex {
    /// Create a new material vertex with specified indices and weights.
    pub fn new(indices: [u8; 4], weights: [u8; 4]) -> Self {
        Self { indices, weights }
    }

    /// Create a material vertex with a single material (no blending).
    pub fn single(index: u8) -> Self {
        Self {
            indices: [index, 0, 0, 0],
            weights: [255, 0, 0, 0],
        }
    }

    /// Create a material vertex blending two materials.
    pub fn blend2(index0: u8, index1: u8, weight0: f32) -> Self {
        let w0 = (weight0.clamp(0.0, 1.0) * 255.0) as u8;
        let w1 = 255 - w0;
        Self {
            indices: [index0, index1, 0, 0],
            weights: [w0, w1, 0, 0],
        }
    }

    /// Create a material vertex blending three materials.
    pub fn blend3(index0: u8, index1: u8, index2: u8, weights: [f32; 3]) -> Self {
        let total = weights[0] + weights[1] + weights[2];
        let normalized = if total > 0.0 {
            [weights[0] / total, weights[1] / total, weights[2] / total]
        } else {
            [1.0, 0.0, 0.0]
        };
        let w0 = (normalized[0] * 255.0) as u8;
        let w1 = (normalized[1] * 255.0) as u8;
        let w2 = 255u8.saturating_sub(w0).saturating_sub(w1);
        Self {
            indices: [index0, index1, index2, 0],
            weights: [w0, w1, w2, 0],
        }
    }

    /// Create a material vertex blending four materials.
    pub fn blend4(indices: [u8; 4], weights: [f32; 4]) -> Self {
        let total: f32 = weights.iter().sum();
        let normalized: [f32; 4] = if total > 0.0 {
            [
                weights[0] / total,
                weights[1] / total,
                weights[2] / total,
                weights[3] / total,
            ]
        } else {
            [1.0, 0.0, 0.0, 0.0]
        };

        let w0 = (normalized[0] * 255.0) as u8;
        let w1 = (normalized[1] * 255.0) as u8;
        let w2 = (normalized[2] * 255.0) as u8;
        let w3 = 255u8
            .saturating_sub(w0)
            .saturating_sub(w1)
            .saturating_sub(w2);

        Self {
            indices,
            weights: [w0, w1, w2, w3],
        }
    }

    /// Pack indices into a u32 for the vertex attribute.
    pub fn packed_indices(&self) -> u32 {
        (self.indices[0] as u32)
            | ((self.indices[1] as u32) << 8)
            | ((self.indices[2] as u32) << 16)
            | ((self.indices[3] as u32) << 24)
    }

    /// Pack weights into a u32 for the vertex attribute.
    pub fn packed_weights(&self) -> u32 {
        (self.weights[0] as u32)
            | ((self.weights[1] as u32) << 8)
            | ((self.weights[2] as u32) << 16)
            | ((self.weights[3] as u32) << 24)
    }
}

/// Encode material data for a collection of vertices into packed u32 vectors.
///
/// Returns `(indices, weights)` vectors suitable for `mesh.insert_attribute()`.
pub fn encode_material_data(vertices: &[MaterialVertex]) -> (Vec<u32>, Vec<u32>) {
    let indices: Vec<u32> = vertices.iter().map(|v| v.packed_indices()).collect();
    let weights: Vec<u32> = vertices.iter().map(|v| v.packed_weights()).collect();
    (indices, weights)
}

/// Helper to pack 4 u8 values into a u32.
#[inline]
pub fn pack_u8x4(values: [u8; 4]) -> u32 {
    (values[0] as u32)
        | ((values[1] as u32) << 8)
        | ((values[2] as u32) << 16)
        | ((values[3] as u32) << 24)
}

/// Helper to unpack a u32 into 4 u8 values.
#[inline]
pub fn unpack_u8x4(packed: u32) -> [u8; 4] {
    [
        (packed & 0xFF) as u8,
        ((packed >> 8) & 0xFF) as u8,
        ((packed >> 16) & 0xFF) as u8,
        ((packed >> 24) & 0xFF) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_material() {
        let v = MaterialVertex::single(42);
        assert_eq!(v.indices[0], 42);
        assert_eq!(v.weights[0], 255);
    }

    #[test]
    fn test_blend2() {
        let v = MaterialVertex::blend2(10, 20, 0.5);
        assert_eq!(v.indices[0], 10);
        assert_eq!(v.indices[1], 20);
        assert_eq!(v.weights[0] + v.weights[1], 255);
    }

    #[test]
    fn test_pack_unpack() {
        let values = [1, 2, 3, 4];
        let packed = pack_u8x4(values);
        let unpacked = unpack_u8x4(packed);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_material_vertex_packing() {
        let v = MaterialVertex::new([10, 20, 30, 40], [100, 50, 75, 30]);
        let packed_indices = v.packed_indices();
        let packed_weights = v.packed_weights();

        assert_eq!(unpack_u8x4(packed_indices), [10, 20, 30, 40]);
        assert_eq!(unpack_u8x4(packed_weights), [100, 50, 75, 30]);
    }
}
