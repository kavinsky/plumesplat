//! plumesplat visual + performance regression harness.
//!
//! ```text
//! plumesplat-bench capture <label> [--out DIR] [--size WxH] [--frames N] [--rounds N]
//!                                  [--warmup SECS] [--settle SECS] [--views a,b]
//! plumesplat-bench compare <base> <candidate> [--out DIR] [--threshold 0-255]
//! ```
//! `<label>`/`<base>`/`<candidate>` are directories under `bench/results/`
//! (or any path). See bench/README.md.

mod capture;
mod compare;
mod scene;

use std::path::{Path, PathBuf};
use std::time::Duration;

const USAGE: &str = "usage:
  plumesplat-bench capture <label> [--out DIR] [--size WxH] [--frames N] [--rounds N] [--warmup SECS] [--settle SECS] [--views a,b]
  plumesplat-bench compare <base> <candidate> [--out DIR] [--threshold 0-255]";

fn results_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("results")
}

/// A bare name means `bench/results/<name>`; anything with a separator is a path.
fn resolve(s: &str) -> PathBuf {
    if s.contains(['/', '\\']) { PathBuf::from(s) } else { results_dir().join(s) }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (positional, flags) = split(&args);
    let flag = |name: &str| flags.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str());
    let num = |name: &str, default: f64| {
        flag(name).map_or(default, |v| v.parse().unwrap_or_else(|_| die(&format!("--{name}: not a number: {v}"))))
    };

    match positional.as_slice() {
        ["capture", label] => {
            let (width, height) = flag("size").map_or((1600, 900), |s| {
                s.split_once('x')
                    .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
                    .unwrap_or_else(|| die("--size must look like 1600x900"))
            });
            capture::run(capture::CaptureArgs {
                out: flag("out").map_or_else(|| resolve(label), PathBuf::from),
                label: label.to_string(),
                width,
                height,
                frames: (num("frames", 300.0) as usize).max(1),
                rounds: (num("rounds", 3.0) as usize).max(1),
                warmup: Duration::from_secs_f64(num("warmup", 0.5)),
                settle: Duration::from_secs_f64(num("settle", 5.0)),
                views: flag("views").map(|v| v.split(',').map(str::to_string).collect()),
            });
        }
        ["compare", base, cand] => {
            let (base, cand) = (resolve(base), resolve(cand));
            let default_out = results_dir().join("compare").join(format!(
                "{}_vs_{}",
                base.file_name().unwrap().to_string_lossy(),
                cand.file_name().unwrap().to_string_lossy()
            ));
            let out = flag("out").map_or(default_out, PathBuf::from);
            let report = compare::run(compare::CompareArgs {
                base: &base,
                cand: &cand,
                out: &out,
                // Renders are bit-identical run to run, so anything above rounding is a real change.
                threshold: num("threshold", 2.0) as u8,
            });
            println!("{report}");
            println!("report: {}", out.join("report.md").display());
        }
        _ => die(USAGE),
    }
}

fn split(args: &[String]) -> (Vec<&str>, Vec<(String, String)>) {
    let mut pos = Vec::new();
    let mut flags = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if let Some(name) = a.strip_prefix("--") {
            let value = it.next().unwrap_or_else(|| die(&format!("--{name} needs a value")));
            flags.push((name.to_string(), value.clone()));
        } else {
            pos.push(a.as_str());
        }
    }
    (pos, flags)
}

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2)
}
