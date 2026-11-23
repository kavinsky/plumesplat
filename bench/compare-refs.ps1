<#
.SYNOPSIS
  Capture + compare plumesplat between a git ref and the working tree (or another ref).

.EXAMPLE
  ./bench/compare-refs.ps1 -Base master
  ./bench/compare-refs.ps1 -Base HEAD~1 -Candidate HEAD -- --rounds 5 --size 1920x1080

  The current bench/ sources are used for BOTH sides (copied into a temporary
  worktree for refs), so the scene and measurement code are identical. The ref
  must therefore have a library API compatible with bench/src/scene.rs.
#>
# No positional binding: otherwise, with only -Base given, the first forwarded
# capture arg (e.g. --rounds) would bind to -Candidate.
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory)] [string] $Base,
    # Empty = current working tree (including uncommitted changes).
    [string] $Candidate = "",
    # Extra args forwarded to `capture`, e.g. --rounds 5 --views closeup
    [Parameter(ValueFromRemainingArguments)] [string[]] $CaptureArgs = @()
)
$ErrorActionPreference = "Stop"
$repo = (git rev-parse --show-toplevel).Trim()
$results = Join-Path $repo "bench/results"

function Invoke-Capture([string] $Ref, [string] $Label) {
    $out = Join-Path $results $Label
    if (-not $Ref) {
        cargo run --release -p plumesplat-bench --manifest-path "$repo/Cargo.toml" -- capture $Label --out $out @CaptureArgs
        if ($LASTEXITCODE) { throw "capture failed for working tree" }
        return
    }
    $sha = (git -C $repo rev-parse --short $Ref).Trim()
    $wt = Join-Path $repo "target/bench-worktrees/$sha"
    if (Test-Path $wt) { git -C $repo worktree remove --force $wt | Out-Null }
    git -C $repo worktree add --detach $wt $sha | Out-Null
    try {
        # Inject the current harness so both sides are measured the same way.
        $wtBench = Join-Path $wt "bench"
        if (Test-Path "$wtBench/src") { Remove-Item -Recurse -Force "$wtBench/src" }
        New-Item -ItemType Directory -Force $wtBench | Out-Null
        Copy-Item "$repo/bench/Cargo.toml" $wtBench
        Copy-Item -Recurse "$repo/bench/src" $wtBench
        $toml = Join-Path $wt "Cargo.toml"
        if (-not (Select-String -Path $toml -Pattern '^\[workspace\]' -Quiet)) {
            Add-Content $toml "`n[workspace]`nmembers = [`"bench`"]"
        }
        # Share the main target dir so Bevy isn't rebuilt from scratch.
        $env:CARGO_TARGET_DIR = Join-Path $repo "target"
        cargo run --release -p plumesplat-bench --manifest-path $toml -- capture $Label --out $out @CaptureArgs
        if ($LASTEXITCODE) { throw "capture failed for $Ref ($sha)" }
    }
    finally {
        Remove-Item Env:CARGO_TARGET_DIR -ErrorAction SilentlyContinue
        git -C $repo worktree remove --force $wt | Out-Null
    }
}

function Get-Label([string] $Ref) {
    if (-not $Ref) { return "worktree" }
    "ref-" + (git -C $repo rev-parse --short $Ref).Trim()
}

$baseLabel = Get-Label $Base
$candLabel = Get-Label $Candidate
Invoke-Capture $Base $baseLabel
Invoke-Capture $Candidate $candLabel
cargo run --release -q -p plumesplat-bench --manifest-path "$repo/Cargo.toml" -- compare $baseLabel $candLabel
