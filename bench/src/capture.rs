//! `capture`: render every view in [`VIEWS`], record frame and GPU pass timings,
//! save a screenshot per view and write `metrics.json`, then exit.
//!
//! Views are measured in several interleaved rounds (A B C D A B C D ...) so slow
//! drifts such as GPU clock boost hit every view equally, and the spread between
//! rounds tells `compare` how much of a delta is just noise.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use bevy::diagnostic::DiagnosticsStore;
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::render::renderer::RenderAdapterInfo;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::window::{PresentMode, WindowResolution};
use plumesplat::prelude::*;
use serde::{Deserialize, Serialize};

use crate::scene::{self, BenchCamera, BenchTerrain, VIEWS};

pub struct CaptureArgs {
    pub out: PathBuf,
    pub label: String,
    pub width: u32,
    pub height: u32,
    pub frames: usize,
    pub rounds: usize,
    pub warmup: Duration,
    pub settle: Duration,
    pub views: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize, Default)]
pub struct Metrics {
    pub label: String,
    pub git: String,
    pub adapter: String,
    pub backend: String,
    pub resolution: [u32; 2],
    pub frames_per_round: usize,
    pub rounds: usize,
    /// Startup until the terrain has its final `PlumeSplatMaterial`
    /// (asset loading + texture array build).
    pub material_ready_ms: f64,
    pub views: Vec<ViewMetrics>,
}

#[derive(Serialize, Deserialize, Default)]
pub struct ViewMetrics {
    pub name: String,
    /// Wall-clock frame time (vsync off), all CPU+GPU work.
    pub frame_ms: Stats,
    /// GPU time per render pass from timestamp queries (empty if unsupported).
    pub gpu_ms: BTreeMap<String, Stats>,
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct Stats {
    pub mean: f64,
    pub median: f64,
    pub p95: f64,
    pub min: f64,
    pub samples: usize,
    /// Median of each round; their min..max range is the run's noise band.
    pub round_medians: Vec<f64>,
}

fn sorted_median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

impl Stats {
    fn from_rounds(rounds: Vec<Vec<f64>>) -> Self {
        let mut rounds: Vec<_> = rounds.into_iter().filter(|r| !r.is_empty()).collect();
        if rounds.is_empty() {
            return Self::default();
        }
        let round_medians = rounds.iter_mut().map(|r| sorted_median(r)).collect();
        let mut v: Vec<f64> = rounds.into_iter().flatten().collect();
        v.sort_by(f64::total_cmp);
        let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
        Self {
            mean: v.iter().sum::<f64>() / v.len() as f64,
            median: at(0.5),
            p95: at(0.95),
            min: v[0],
            samples: v.len(),
            round_medians,
        }
    }

    /// `(min, max)` of the per-round medians.
    pub fn band(&self) -> (f64, f64) {
        self.round_medians
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &m| (lo.min(m), hi.max(m)))
    }
}

enum Phase {
    Loading,
    Settling { until: Instant },
    Warmup { until: Instant },
    Sampling,
    Capturing,
    Next,
}

#[derive(Default)]
struct ViewAcc {
    frame: Vec<Vec<f64>>,
    gpu: BTreeMap<String, Vec<Vec<f64>>>,
}

#[derive(Resource)]
struct Bench {
    args: CaptureArgs,
    views: Vec<usize>,
    /// Index into `views` x `rounds`.
    step: usize,
    phase: Phase,
    started: Instant,
    frame_samples: Vec<f64>,
    gpu_samples: BTreeMap<String, Vec<f64>>,
    gpu_last_seen: BTreeMap<String, Instant>,
    acc: Vec<ViewAcc>,
    metrics: Metrics,
}

impl Bench {
    fn view_slot(&self) -> usize {
        self.step % self.views.len()
    }
    fn round(&self) -> usize {
        self.step / self.views.len()
    }
}

pub fn run(args: CaptureArgs) {
    let views: Vec<usize> = match &args.views {
        None => (0..VIEWS.len()).collect(),
        Some(names) => names
            .iter()
            .map(|n| {
                VIEWS.iter().position(|v| v.name == n).unwrap_or_else(|| {
                    let all: Vec<_> = VIEWS.iter().map(|v| v.name).collect();
                    panic!("unknown view `{n}`, available: {all:?}")
                })
            })
            .collect(),
    };
    std::fs::create_dir_all(&args.out).expect("create output dir");

    let (w, h) = (args.width, args.height);
    let metrics = Metrics {
        label: args.label.clone(),
        git: git_describe(),
        resolution: [w, h],
        frames_per_round: args.frames,
        rounds: args.rounds,
        ..default()
    };

    App::new()
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: format!("plumesplat-bench [{}]", args.label),
                        resolution: WindowResolution::new(w, h).with_scale_factor_override(1.0),
                        resizable: false,
                        present_mode: PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    // Share the crate's assets instead of keeping a copy in bench/.
                    file_path: concat!(env!("CARGO_MANIFEST_DIR"), "/../assets").into(),
                    ..default()
                }),
            RenderDiagnosticsPlugin,
            PlumeSplatPlugin,
        ))
        .insert_resource(Bench {
            acc: views.iter().map(|_| ViewAcc::default()).collect(),
            args,
            step: 0,
            views,
            phase: Phase::Loading,
            started: Instant::now(),
            frame_samples: Vec::new(),
            gpu_samples: BTreeMap::new(),
            gpu_last_seen: BTreeMap::new(),
            metrics,
        })
        .add_systems(Startup, scene::setup)
        .add_systems(Update, drive)
        .run();
}

#[allow(clippy::too_many_arguments)] // Bevy system params.
fn drive(
    mut commands: Commands,
    mut bench: ResMut<Bench>,
    time: Res<Time<Real>>,
    store: Res<DiagnosticsStore>,
    adapter: Option<Res<RenderAdapterInfo>>,
    ready: Query<(), (With<BenchTerrain>, With<MeshMaterial3d<PlumeSplatMaterial>>)>,
    mut camera: Query<&mut Transform, With<BenchCamera>>,
    mut exit: MessageWriter<AppExit>,
) {
    let now = Instant::now();
    let b = &mut *bench;
    match b.phase {
        Phase::Loading => {
            if ready.is_empty() {
                if now - b.started > Duration::from_secs(120) {
                    error!("terrain material never became ready (missing textures?)");
                    exit.write(AppExit::error());
                }
                return;
            }
            b.metrics.material_ready_ms = (now - b.started).as_secs_f64() * 1000.0;
            if let Some(info) = adapter {
                b.metrics.adapter = info.name.clone();
                b.metrics.backend = format!("{:?}", info.backend);
            }
            info!("material ready after {:.0} ms", b.metrics.material_ready_ms);
            // Let pipelines compile, textures upload and GPU clocks ramp up.
            b.phase = Phase::Settling { until: now + b.args.settle };
        }
        Phase::Settling { until } => {
            if now >= until {
                start_view(b, &mut camera, now);
            }
        }
        Phase::Warmup { until } => {
            if now >= until {
                b.frame_samples.clear();
                b.gpu_samples.clear();
                b.gpu_last_seen = gpu_measurements(&store).map(|(k, t, _)| (k, t)).collect();
                b.phase = Phase::Sampling;
            }
        }
        Phase::Sampling => {
            b.frame_samples.push(time.delta_secs_f64() * 1000.0);
            // GPU timings arrive a few frames late; only take measurements we haven't seen.
            for (pass, t, value) in gpu_measurements(&store) {
                if b.gpu_last_seen.get(&pass) != Some(&t) {
                    b.gpu_last_seen.insert(pass.clone(), t);
                    b.gpu_samples.entry(pass).or_default().push(value);
                }
            }
            if b.frame_samples.len() < b.args.frames {
                return;
            }
            let slot = b.view_slot();
            let view = &VIEWS[b.views[slot]];
            let mut frames = std::mem::take(&mut b.frame_samples);
            info!("round {} {:>10}: frame median {:.3} ms", b.round() + 1, view.name, sorted_median(&mut frames));
            let acc = &mut b.acc[slot];
            acc.frame.push(frames);
            for (pass, samples) in std::mem::take(&mut b.gpu_samples) {
                acc.gpu.entry(pass).or_default().push(samples);
            }

            if b.round() > 0 {
                b.step += 1;
                b.phase = Phase::Next;
                return;
            }
            // Screenshots are deterministic, so one per view (first round) is enough.
            let path = b.args.out.join(format!("{}.png", view.name));
            commands
                .spawn(Screenshot::primary_window())
                .observe(move |ev: On<ScreenshotCaptured>, mut bench: ResMut<Bench>| {
                    match ev.image.clone().try_into_dynamic() {
                        Ok(img) => img.to_rgb8().save(&path).expect("save screenshot"),
                        Err(e) => error!("screenshot conversion failed: {e:?}"),
                    }
                    bench.step += 1;
                    bench.phase = Phase::Next;
                });
            b.phase = Phase::Capturing;
        }
        Phase::Capturing => {}
        Phase::Next => {
            if b.step < b.views.len() * b.args.rounds {
                start_view(b, &mut camera, now);
                return;
            }
            for (slot, acc) in std::mem::take(&mut b.acc).into_iter().enumerate() {
                b.metrics.views.push(ViewMetrics {
                    name: VIEWS[b.views[slot]].name.into(),
                    frame_ms: Stats::from_rounds(acc.frame),
                    gpu_ms: acc.gpu.into_iter().map(|(k, v)| (k, Stats::from_rounds(v))).collect(),
                });
            }
            let json = serde_json::to_string_pretty(&b.metrics).unwrap();
            std::fs::write(b.args.out.join("metrics.json"), json).expect("write metrics.json");
            info!("wrote {}", b.args.out.display());
            exit.write(AppExit::Success);
        }
    }
}

fn start_view(b: &mut Bench, camera: &mut Query<&mut Transform, With<BenchCamera>>, now: Instant) {
    let view = &VIEWS[b.views[b.view_slot()]];
    for mut t in camera.iter_mut() {
        *t = Transform::from_translation(view.eye).looking_at(view.target, Vec3::Y);
    }
    b.phase = Phase::Warmup { until: now + b.args.warmup };
}

/// `(pass name, measurement time, ms)` for every GPU pass timing in the store.
fn gpu_measurements(store: &DiagnosticsStore) -> impl Iterator<Item = (String, Instant, f64)> + '_ {
    store.iter().filter_map(|d| {
        let path = d.path().as_str();
        let pass = path.strip_prefix("render/")?.strip_suffix("/elapsed_gpu")?;
        let m = d.measurement()?;
        Some((pass.to_string(), m.time, m.value))
    })
}

fn git_describe() -> String {
    let run = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let rev = run(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into());
    // Only the library sources matter; `bench/` itself may be injected into old checkouts.
    let dirty = run(&["status", "--porcelain", "--", "../src"]).is_some_and(|s| !s.is_empty());
    if dirty { format!("{rev}-dirty") } else { rev }
}
