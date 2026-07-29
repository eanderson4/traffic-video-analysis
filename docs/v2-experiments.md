# tva v2 experiments — Phase A

Testbed: the EP-03 hero clip
(`math-vs-vibes/promo/ep-03/phantom-jam-short/work/hero-wave/tva`,
3840×2160 @ 25 fps, 426 frames, elevated carriageway, drone translates hard
in the first and final second). Baseline artifacts are the current v1 run.

Shared metrics (compute for baseline AND each experiment):
- **M1 static residual**: median + p90 world-px reprojection residual of
  static-vehicle landmarks, per frame; report mid-clip vs last 2 s.
- **M2 scale stability**: local Jacobian scale of the frame→world map at the
  jam-road centroid, per frame; report max/median ratio (v1 blows up to
  10-14× at the end; target < 2×).
- **M3 dot offset**: median frame-px distance between mapped world estimate
  and detected bbox center at frames 300/380/415/424 (v1: 1.5/14/69/65 px).
- **M4 speed plausibility** (Punzo): share of |jerk| > 15 m/s³ (car-length
  scale: 1 car_len ≈ 4.5 m); ∫v-vs-displacement drift per track; stopped
  cars read < 0.1 car_len/s.
- **M5 stop-onset count + wave fit**: number of stop events surviving QA;
  Theil-Sen wave speed through (t, station) onsets with CI; must land in
  10-25 km/h backward using the car-length scale prior.
- **M6 visual QA**: speed-profile sheet (raw vs smoothed), overlay stills at
  t = 0/9/15/16.8 s.

## Experiment R1 — direct-to-reference registration (branch `regv2`)

Replace chained stabilization with reference-frame registration:
- Blur-score frames (variance of Laplacian on vehicle-masked regions); pick
  sharpest mid-clip frame as reference; keyframes every ~2 s (blur-gated).
- Frame→keyframe→reference matching, depth ≤ 2. ORB (4k) or SIFT features,
  vehicle-bbox exclusion masks (all detections, dilated 15%), MAGSAC++.
- First try `stabilo` (pip; MUST NOT alter torch — check, install --no-deps
  if needed); fall back to a ~150-line in-repo implementation with the same
  defaults (ORB, SNN 0.9, MAGSAC++, 0.5× downscale for matching, full-res H).
- Optional deck polygon mask (hand-drawn once, JSON in work dir) restricting
  features to the elevated carriageway.
- Output: same `homographies.json` schema (+ `"method": "ref-frame"`), so
  world/render run unchanged. Keep the static-vehicle anchor pass available
  but expect it to become a no-op improvement.

Success: M1 last-2s ≤ 2× mid-clip; M2 max/median < 2; M3 ≤ 5 px at all four
probe frames; plate.png at least as sharp as v1.

## Experiment K1 — velocity-fusion kinematics (branch `kinv2`)

Add LK-flow velocity measurements and fuse:
- Per track per frame-pair: Shi-Tomasi corners inside bbox eroded 20-25%,
  pyramidal LK to next frame, endpoints through their own frames'
  homographies, median world displacement / dt → z_v (vx, vy). Mahalanobis
  gate against the filter prediction; drop degenerate flow (few corners,
  high spread).
- Kalman: state [x y vx vy ax ay] (or 2× 1-D [p v a]), measurements z_p
  (current adaptive R_p) + z_v (own R_v). RTS backward pass. Montanino-Punzo
  prefilter before the filter (spike cuts, impossible jumps).
- Stop onset: keep hysteresis; localize onset via Mexican-hat wavelet-energy
  peak on the fused speed series, sub-frame interpolated.
- Validation extra: on manually-static parked cars, z_v median must read
  < 0.05 car_len/s — direct proof the flow measurement is unbiased.

Success: M4 jerk share ≈ 0 with stopped-car speeds pinned at 0; speed curves
at clip ends bounded and monotone-plausible (no oscillation) EVEN ON v1
homographies; M5 wave speed in band with tighter CI than baseline.

## Sequencing

R1 and K1 are independent (K1 must prove robustness on v1's bad
homographies; rerunning K1 on R1's output afterward is the integration
test). Each runs on its own git branch in its own worktree, own copy of the
work dir. Integration: merge both, run full pipeline, compare all metrics,
then re-render the hero overlay.

## QA harness (shared, lands with whichever branch merges first)

`python3 -m tva qa --work <dir>` → prints M1-M5 as a table, writes
`qa/report.json`. Both experiments add their metrics here rather than ad-hoc
prints, so baseline-vs-experiment comparison is one command.
