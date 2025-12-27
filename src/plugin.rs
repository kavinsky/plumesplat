//! Bevy plugin for PlumeSplat.

use bevy::app::{App, Plugin};
use bevy::asset::embedded_asset;
use bevy::pbr::MaterialPlugin;

use crate::material::PlumeSplatMaterial;

/// Plugin that registers the PlumeSplat material and shaders.
///
/// Add this to your Bevy app to enable terrain splatting:
///
/// ```rust,no_run
/// use bevy::prelude::*;
/// use plumesplat::prelude::*;
///
/// App::new()
///     .add_plugins(DefaultPlugins)
///     .add_plugins(PlumeSplatPlugin)
///     .run();
/// ```
pub struct PlumeSplatPlugin;

impl Plugin for PlumeSplatPlugin {
    fn build(&self, app: &mut App) {
        // Embed the shader files into the binary
        embedded_asset!(app, "shaders/plumesplat.wgsl");
        embedded_asset!(app, "shaders/triplanar.wgsl");
        embedded_asset!(app, "shaders/blending.wgsl");

        // Register the custom material
        app.add_plugins(MaterialPlugin::<PlumeSplatMaterial>::default());
    }
}
