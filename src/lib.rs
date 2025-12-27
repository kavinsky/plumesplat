//! # PlumeSplat
//!
//! Advanced terrain and mesh splatting for Bevy Engine with support for up to 256 materials.
//!
//! ## Features
//!
//! - **Texture Array Splatting**: Blend up to 256 materials in a single draw call
//! - **Per-Vertex Material Assignment**: Store up to 4 material indices per vertex
//! - **Triplanar Mapping**: UV-less texturing for arbitrary geometry
//! - **Height-Based Blending**: Optional height-map based transitions between materials
//! - **Full PBR Support**: Extends StandardMaterial for proper lighting integration
//!
//! ## Usage
//!
//! ```rust,no_run
//! use bevy::prelude::*;
//! use plumesplat::prelude::*;
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(DefaultPlugins)
//!         .add_plugins(PlumeSplatPlugin)
//!         .run();
//! }
//! ```

mod material;
mod plugin;
mod vertex_attributes;

pub mod prelude {
    pub use crate::material::{PlumeSplatExtension, PlumeSplatMaterial, PlumeSplatSettings};
    pub use crate::plugin::PlumeSplatPlugin;
    pub use crate::vertex_attributes::{
        ATTRIBUTE_MATERIAL_INDICES, ATTRIBUTE_MATERIAL_WEIGHTS, MaterialVertex,
        encode_material_data,
    };
}

pub use material::*;
pub use plugin::*;
pub use vertex_attributes::*;
