//! `compare`: diff two capture runs. Writes per-view diff heatmaps and
//! side-by-side strips plus a markdown report with image metrics and timing deltas.

use std::fmt::Write;
use std::path::Path;

use image::{Rgb, RgbImage};

use crate::capture::{Metrics, Stats};

// Note: with a single round (`--rounds 1`) the range collapses to one value,
// so any difference is reported as faster/slower. Use >= 3 rounds.

pub struct CompareArgs<'a> {
    pub base: &'a Path,
    pub cand: &'a Path,
    pub out: &'a Path,
    /// Per-channel difference (0-255) above which a pixel counts as "changed".
    pub threshold: u8,
}

struct ImageDiff {
    mae: f64,
    rmse: f64,
    psnr: f64,
    max: u8,
    changed_pct: f64,
}

pub fn run(args: CompareArgs) -> String {
    std::fs::create_dir_all(args.out).expect("create output dir");
    let base = load_metrics(args.base);
    let cand = load_metrics(args.cand);

    let mut md = String::new();
    let _ = writeln!(md, "# plumesplat bench: `{}` vs `{}`\n", base.label, cand.label);
    let _ = writeln!(md, "| | base | candidate |\n|---|---|---|");
    let _ = writeln!(md, "| git | {} | {} |", base.git, cand.git);
    let _ = writeln!(md, "| adapter | {} ({}) | {} ({}) |", base.adapter, base.backend, cand.adapter, cand.backend);
    let _ = writeln!(md, "| resolution | {:?} | {:?} |", base.resolution, cand.resolution);
    let _ = writeln!(
        md,
        "| material ready | {:.0} ms | {:.0} ms ({}) |\n",
        base.material_ready_ms,
        cand.material_ready_ms,
        pct(base.material_ready_ms, cand.material_ready_ms)
    );
    if base.adapter != cand.adapter || base.resolution != cand.resolution {
        md.push_str("> **Warning:** runs used different GPUs or resolutions; timings are not comparable.\n\n");
    }

    md.push_str("## Images\n\n");
    let _ = writeln!(
        md,
        "Changed = any channel differs by more than {}/255.\n\n| view | changed px | MAE | RMSE | PSNR (dB) | max |\n|---|---|---|---|---|---|",
        args.threshold
    );
    let mut images_md = String::new();
    for view in &cand.views {
        let name = &view.name;
        let (Some(a), Some(b)) = (load_png(&args.base.join(format!("{name}.png"))), load_png(&args.cand.join(format!("{name}.png")))) else {
            let _ = writeln!(md, "| {name} | missing screenshot | | | | |");
            continue;
        };
        if a.dimensions() != b.dimensions() {
            let _ = writeln!(md, "| {name} | size mismatch {:?} vs {:?} | | | | |", a.dimensions(), b.dimensions());
            continue;
        }
        let (d, heat) = diff(&a, &b, args.threshold);
        heat.save(args.out.join(format!("{name}_diff.png"))).expect("save diff");
        side_by_side(&a, &b, &heat).save(args.out.join(format!("{name}_sbs.png"))).expect("save sbs");
        let psnr = if d.psnr.is_finite() { format!("{:.2}", d.psnr) } else { "identical".into() };
        let _ = writeln!(
            md,
            "| {name} | {:.3}% | {:.3} | {:.3} | {psnr} | {} |",
            d.changed_pct, d.mae, d.rmse, d.max
        );
        let _ = writeln!(images_md, "### {name}\n\nbase | candidate | diff (x4)\n\n![{name}]({name}_sbs.png)\n");
    }

    md.push_str("\n## Timings\n\n");
    let _ = write!(
        md,
        "Median over all sampled frames; lower is better. Δ is candidate vs base. \
         The range is min..max of the per-round medians. GPU pass times (timestamp queries) \
         are the reliable signal: a verdict needs non-overlapping ranges and |Δ| ≥ {:.0}%. \
         Wall-clock frame time also includes CPU and present jitter, so it is info only. \
         GPU passes under 0.05 ms are hidden.\n\n",
        MIN_EFFECT * 100.0
    );
    md.push_str("| view | metric | base ms (range) | candidate ms (range) | Δ | verdict |\n|---|---|---|---|---|---|\n");
    for cv in &cand.views {
        let Some(bv) = base.views.iter().find(|v| v.name == cv.name) else { continue };
        row(&mut md, &cv.name, "frame (wall clock)", &bv.frame_ms, &cv.frame_ms, false);
        if cv.gpu_ms.is_empty() {
            let _ = writeln!(md, "| {} | gpu | n/a (no timestamp queries on this backend) | | | |", cv.name);
        }
        for (pass, cs) in &cv.gpu_ms {
            match bv.gpu_ms.get(pass) {
                Some(bs) if bs.median.max(cs.median) >= 0.05 => {
                    row(&mut md, &cv.name, &format!("gpu `{pass}`"), bs, cs, true)
                }
                _ => {}
            }
        }
    }
    md.push('\n');
    md.push_str("## Side by side\n\n");
    md.push_str(&images_md);

    std::fs::write(args.out.join("report.md"), &md).expect("write report.md");
    md
}

/// Smallest relative change worth a verdict, even when round ranges don't overlap.
const MIN_EFFECT: f64 = 0.02;

fn row(md: &mut String, view: &str, metric: &str, b: &Stats, c: &Stats, judge: bool) {
    let ((blo, bhi), (clo, chi)) = (b.band(), c.band());
    let big = b.median > 0.0 && ((c.median - b.median) / b.median).abs() >= MIN_EFFECT;
    let verdict = if !judge {
        "info only"
    } else if big && chi < blo {
        "faster"
    } else if big && clo > bhi {
        "**slower**"
    } else {
        "~ same"
    };
    let _ = writeln!(
        md,
        "| {view} | {metric} | {:.3} ({blo:.3}..{bhi:.3}) | {:.3} ({clo:.3}..{chi:.3}) | {} | {verdict} |",
        b.median,
        c.median,
        pct(b.median, c.median)
    );
}

fn pct(base: f64, cand: f64) -> String {
    if base <= 0.0 {
        return "n/a".into();
    }
    format!("{:+.1}%", (cand - base) / base * 100.0)
}

fn load_metrics(dir: &Path) -> Metrics {
    let path = dir.join("metrics.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn load_png(path: &Path) -> Option<RgbImage> {
    image::open(path).ok().map(|i| i.to_rgb8())
}

/// Per-pixel stats plus a heatmap (black = same, brighter/redder = larger diff, x4 gain).
fn diff(a: &RgbImage, b: &RgbImage, threshold: u8) -> (ImageDiff, RgbImage) {
    let (w, h) = a.dimensions();
    let mut heat = RgbImage::new(w, h);
    let (mut sum_abs, mut sum_sq, mut max, mut changed) = (0u64, 0u64, 0u8, 0u64);
    for ((pa, pb), out) in a.pixels().zip(b.pixels()).zip(heat.pixels_mut()) {
        let mut px_max = 0u8;
        for c in 0..3 {
            let d = pa[c].abs_diff(pb[c]);
            sum_abs += d as u64;
            sum_sq += (d as u64) * (d as u64);
            px_max = px_max.max(d);
        }
        max = max.max(px_max);
        if px_max > threshold {
            changed += 1;
        }
        let v = (px_max as u32 * 4).min(255) as u8;
        *out = Rgb([v, v / 3, 0]);
    }
    let n = (w * h * 3) as f64;
    let mse = sum_sq as f64 / n;
    let d = ImageDiff {
        mae: sum_abs as f64 / n,
        rmse: mse.sqrt(),
        psnr: if mse == 0.0 { f64::INFINITY } else { 10.0 * (255.0f64 * 255.0 / mse).log10() },
        max,
        changed_pct: changed as f64 / (w * h) as f64 * 100.0,
    };
    (d, heat)
}

fn side_by_side(a: &RgbImage, b: &RgbImage, heat: &RgbImage) -> RgbImage {
    let (w, h) = a.dimensions();
    let mut out = RgbImage::new(w * 3, h);
    for (i, img) in [a, b, heat].into_iter().enumerate() {
        image::imageops::replace(&mut out, img, (w * i as u32) as i64, 0);
    }
    out
}
