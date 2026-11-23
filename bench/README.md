# plumesplat-bench

A tool for checking how a change affects rendering and speed. It renders a
fixed scene, saves screenshots and timings, and compares two such runs.
It is a separate workspace crate with `publish = false`. The root `Cargo.toml`
also excludes `bench/`, so none of it ships with `plumesplat`.

## Quick start

```powershell
# Compare a ref (here master) against your working tree, including uncommitted changes
./bench/compare-refs.ps1 -Base master

# Compare two refs, passing extra options through to capture
./bench/compare-refs.ps1 -Base HEAD~1 -Candidate HEAD -- --rounds 5 --size 1920x1080
```

Manual workflow (for example, to compare a settings change in the same build):

```powershell
cargo run --release -p plumesplat-bench -- capture before
# ...edit src/...
cargo run --release -p plumesplat-bench -- capture after
cargo run --release -p plumesplat-bench -- compare before after
```

Output is written to `bench/results/`, which is gitignored:

```
results/<label>/        <view>.png, metrics.json
results/compare/<a>_vs_<b>/
    report.md           image metrics, timing table, side-by-side images
    <view>_diff.png     heatmap: black = identical, brighter = bigger difference (x4 gain)
    <view>_sbs.png      base | candidate | diff
```

Always use `--release`. Debug builds measure the wrong thing.

## What it measures

| | |
|---|---|
| **Images** | One screenshot per view. Identical code renders bit-identical images, so any nonzero MAE means the change did something. "Changed px" counts pixels where a channel differs by more than `--threshold` (default 2/255). |
| **GPU pass time** | Per-pass GPU time from timestamp queries (`RenderDiagnosticsPlugin`). The terrain shader cost shows up in `main_opaque_pass_3d`. This is the number to trust. Timestamp queries need Vulkan or DX12. |
| **Frame time** | Wall-clock time with vsync off. It includes CPU and present jitter, so the report marks it "info only". |
| **Material ready** | Time from startup until the terrain has its final material: asset loading plus building the texture arrays on the CPU. It is a single sample, so treat it as rough. |

Views are defined in `src/scene.rs`:

- `overview`: the same framing as the basic example.
- `closeup`: ground fills the screen. This is the worst case for shader cost.
- `grazing`: a low angle toward the horizon, to see mips and distant tiling.
- `topdown`: looking straight down, to see repetition and tile seams.

To add a view, add an entry to `VIEWS`. To test a new feature, change the
builder call in `scene::setup`. Both sides of a `compare-refs` run use the
current `bench/` sources, so any ref you compare must compile against
`scene.rs`.

## Reading timing results

Views are measured in interleaved rounds (A B C D, A B C D, ...). The report
shows the median and the min..max range of the per-round medians. A pass is
marked **faster** or **slower** only when the two ranges don't overlap and
the change is at least 2%. Anything else is "~ same".

GPU clocks can settle at a different level from one run to the next, and the
ranges within a single run don't capture that. As a result, a pass that the
change could not have affected can still show a difference of about 10%. Before
trusting a timing result:

- Use `--rounds 5` or more and a larger `--size` so that GPU work dominates.
- Run the comparison twice, or swap base and candidate, and check that the
  result has the same sign both times.
- Check a pass the change should not affect (for example, `clustering`). If it
  moved too, the difference comes from the run, not from your change.
- On NVIDIA, you can fix the clocks while benchmarking with
  `nvidia-smi --lock-gpu-clocks=<MHz>` (requires admin; undo with `-rgc`).
- Close other GPU-heavy programs, and keep the window visible and unminimized.

## CLI

```
plumesplat-bench capture <label> [--out DIR] [--size WxH] [--frames N] [--rounds N]
                                 [--warmup SECS] [--settle SECS] [--views a,b]
plumesplat-bench compare <base> <candidate> [--out DIR] [--threshold 0-255]
```

Defaults: 1600x900, 300 frames per view per round, 3 rounds, 0.5 s warmup per
view, and 5 s settle after loading. A bare label means `bench/results/<label>`.
A value containing `/` or `\` is used as a path.
