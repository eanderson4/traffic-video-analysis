# R1 + K1 integration: K1 velocity fusion on R1 ref-frame homographies

Branch `v2-integration` (= main + regv2 + kinv2 + integration fixes),
testbed = fresh copy of the EP-03 hero clip work dir (`testdata/hero`,
3840x2160 @ 25 fps, 426 frames). Pipeline as run:

    register (SG-smoothed ref-frame Hs, ref = frame 140, 13 keyframes)
    -> world (position-only) -> anchors -> flow -> world (fused)
    -> qa / r1_metrics / speedqa / render

deck.json reused from the R1 worktree (valid: identical tracks.json gives
the identical reference frame + keyframe set). torch 2.9.1 / torchvision
0.24.1 / fsspec 2024.12.0 verified untouched before and after.

## Metric table

v1 and integrated columns recomputed here with the merged QA code
(baseline reproduces both reports exactly); R1-only and K1-only from
their reports. R1-only shows the register+anchors column where it
differs. K1-only ran on v1 homographies, so its M1-M3 are the baseline's.
Cross-column M1 absolutes are approximate (v1 = frame-0 plane, R1/integ =
frame-140 plane, scale ratio ~1.2; landmark sets also differ per run).

| metric | v1 baseline | R1-only | K1-only | integrated | target |
|---|---|---|---|---|---|
| M1 mid med/p90 (world px) | 0.88/4.64 | 2.57/12.2 (1.75/7.8) | = v1 | 1.24/10.1 | — |
| M1 last-2s med/p90 | 4.02/30.7 | 1.99/10.4 (0.93/2.7) | = v1 | **0.77/1.77** | — |
| **M1 last2s/mid (med)** | 4.58 | 0.77 (0.53) | = v1 | **0.62** | <= 2 |
| **M2 jac max/median** | 1.42 | 1.51 | = v1 | **1.53** | < 2 |
| **M3 f300/380/415/424 (px)** | 1.5/14.3/69/65 | 4.8/1.3/2.3/1.0 (3.9/1.6/2.0/1.2) | = v1 | **2.7/0.8/1.0/1.8** | <= 5 all |
| M4 jerk share | 0.114 | — | **0.0073** | 0.0148 | ~0 |
| M4 stopped-run med (cl/s) | 0.059 | — | **0.041** | 0.048 | < 0.1 |
| M4 stopped runs < 0.1 | 74.4% | — | **88.8%** | 85.6% | high |
| M4 integral drift med/p90 | .0002/.0039 | — | .0002/.0025 | **.0001/.0033** | ~0 |
| parked z_v raw mag med (cl/s) | — | — | **0.098** | 0.297 | < 0.05 |
| parked z_v corrected mag med | — | — | 0.100 | **0.063** | — |
| parked z_v signed bias med | — | — | 0.030 | **0.019** | < 0.05 |
| M5 stop events | 38 | — | 38 | 48 | ~38 |
| M5 wave, wavelet onsets (km/h) | 15.2 [5.7,21.0] | — | **15.7 [13.2,21.6]** | 14.1 [6.3,53.8] (n=5) | 10-25 |
| M5 wave, hysteresis onsets | = wavelet | — | 19.3 [13.3,25.4] | 15.2 [6.9,20.1] (n=7) | 10-25 |
| M5 field-slope cross-check | 12.2 | — | 14.1 | 8.8 | agrees |
| M6 speed profiles / stills | end blowup | — | bounded | **bounded, clean** | — |

(cl/s = car_len/s; car_len prior 4.5 m; integrated car_len = 49.6 WORLD
px vs 55.6 image px — see fix 3.)

Verdict vs solo branches:
- **vs R1 (M1-M3): integrated >= R1 everywhere** — M3 all four probes
  <= 2.7 px (R1's worst was 4.8), M1 last-2s p90 1.77 px, ratio 0.62.
- **vs K1 (M4): near parity** — stopped-run speeds and parked-car bias
  better than K1; jerk share 2x K1's (0.015 vs 0.007, both ~100x below
  baseline). Cause: residual white registration jitter (below).
- **vs K1 (M5): point estimate reproduced, CI degraded** — 14-15 km/h
  backward across both onset variants (K1 15.7, baseline 15.2), but the
  bootstrap CI is wider than K1's ([6.9,20.1] hysteresis vs [13.2,21.6])
  because only 5-7 onsets lie on the front and their scatter is larger.
  Honest fail on "tighter CI than K1"; still tighter than baseline.

## What conflicted and how it was resolved

Textual: only `tva/__main__.py` (both branches edited the subcommand
list) — kept `register` and `flow`, plus kinv2's dedicated speedqa/qa
parsers.

The real conflicts were semantic. Running K1 unmodified on R1's
homographies was WORSE than either solo branch: parked z_v 0.34 cl/s
(3.5x K1), stopped runs regressed, and the M5 wave fit collapsed
entirely (field slope 0.0, out of band). Four causes, all of the form
"a K1 defense calibrated on chained homographies misfires on ref-frame
ones":

1. **White differential jitter (the biggest finding).** z_v cancels
   chain error because consecutive frames SHARE it. R1 estimates every
   frame independently (ORB edge + per-frame LK polish), so consecutive
   Hs carry uncorrelated ~0.6 px jitter that z_v differences and
   multiplies by fps: static-car common-mode slip measured 16.0 px/s
   with lag-1 autocorrelation -0.30 (differenced white noise), vs
   4.1 px/s at +0.62 (smooth drift) on v1. R1's registration is more
   ACCURATE but differentially NOISIER — the property K1 depends on.
   Fix at the registration layer: Savitzky-Golay(7,2) smoothing of the
   H parameter sequence in `register.py` (slip 16.8 -> 9.7 px/s, landmark
   residual unchanged; a plain Gaussian lags the fast end move).
2. **ZvBias pooling averaged away the signal it corrects.** K1 pools
   static-vehicle slip samples over +/-2 frames — right for smooth
   drift, exactly wrong for per-frame white jitter. On ref-frame Hs the
   fit is now per frame (pool=0; 44 static samples/frame median on this
   clip), and the R_p "coherent slip" inflation uses one frame (dt)
   instead of 0.5 s (white jitter does not integrate into z_p error;
   the 0.5 s assumption was de-weighting positions ~8 px everywhere).
3. **car_len was image px, thresholds/conversions consume world px.**
   On v1 the frame-0 reference plane coincides with the image plane, so
   it silently worked. On R1's frame-140 plane every car_len-derived
   quantity (v_stop/v_move hysteresis, Edie slow threshold, jerk bound,
   px/s -> km/h) was off ~15-20%. This alone broke M5: v_move too high
   made the upstream creep pocket read "slow" early, welding it onto the
   main jam component and flattening the front boundary. car_len is now
   measured in world px via the per-observation Jacobian (no-op on v1).
4. **z_v Jacobian division on trusted geometry.** K1 divides z_v by the
   local Jacobian because on a corrupt chain ALL corruption enters
   through J_f. On ref-frame homographies J_f is real perspective; the
   division handed the filter image-scale velocities to fuse against
   world-scale positions (~20% systematic mismatch). `load_flow` now
   uses z_v raw (world px/s) when method == "ref-frame", scaling only
   the image-px noise floor by J_f. The chained path is unchanged.

## Corrections made redundant / ownership decisions

- **v1 plate mosaic drift refinement: OFF on ref-frame** (default
  auto-detects method; R1 report's requested flip, implemented).
- **z_v J_f normalization: OFF on ref-frame** (see 4) — it was the
  corrupt-chain defense; R1 makes it a bias.
- **ZvBias is NOT redundant** — the anticipated double-correction with
  R1's per-frame trust/polish didn't materialize as double-subtraction;
  instead each layer owns a different component: register/SG owns
  temporal smoothness, anchors owns smooth absolute drift, ZvBias
  (per-frame) owns the residual per-frame-pair common mode (field MAD
  18.3 -> 3.1 px/s across the fixes) and stays the R_p/R_v trust signal.
- **anchors stays smoothed.** Per-frame (unsmoothed) anchor corrections
  were tested as an alternative jitter owner and rejected: ~24-48 bbox
  landmarks/frame can't out-resolve the jitter (static slip went 16.8 ->
  19.8 px/s) — while LK flow corners can. `anchors(ws, smooth=)` keeps
  the experiment reachable. With R1+anchors the anchors pass is a small
  polish (residual 2.64 -> 2.14 px), as R1 predicted; R1's flow indeed
  expects it on top and the integrated pipeline runs it.

## Remaining issues (genuine, not integration bugs)

- **Raw z_v noise floor is ~3x higher on ref-frame Hs** (0.30 vs 0.098
  cl/s parked raw magnitude) even after SG smoothing. The slip field
  removes the common mode (corrected 0.063, signed bias 0.019 — both
  better than K1), but the incoherent residual (~8 px/s per car) is
  spatially structured homography noise that a per-frame affine can't
  fully capture. It shows up as the 2x jerk share and as stop-onset
  timing scatter. A registration-layer fix would need the per-frame
  polish to be jointly (temporally) estimated, not per-frame.
- **M5 front extraction is fragile at this n.** The wavelet-onset front
  fit rests on 5 events (MAD 40 px); the largest-slow-component boundary
  fragments under small speed perturbations (a 1-column time-closing of
  the slow mask reconnects it and moves the field slope 8.8 -> ~11.6
  km/h, but was NOT adopted — it also shifts K1's committed answer, i.e.
  it changes the measurand rather than robustifying it). All estimator
  variants agree on 12-17 km/h backward; the honest statement is a wide
  CI, not a pass.
- **M1 cross-column comparability** is limited: landmark sets are
  derived per run and the reference planes differ in scale (~1.2x).
  Ratios and frame-px M3 are the trustworthy comparisons (both excellent
  here).
- The stop-event count rose to 48 (38 solo): more creep re-stops survive
  QA on the quieter registration; event de-duplication for M5 was not
  revisited.

## Artifacts (testdata/hero, gitignored)

- `homographies.json` (final: ref-frame + SG + anchors),
  `homographies-refframe.json` (pre-anchors snapshot),
  `homographies-v1.json`, `world_tracks-v1.json` (baseline snapshots)
- `qa/report-{baseline,final}.json`, `qa/r1-metrics.jsonl` (all runs,
  labels v1-baseline .. integ-final)
- `qa/speed-profiles-integ-final.png`, `qa/still-f{060,300,420}.jpg`,
  `overlay-speed.mp4` (rendered from the fused run; markers glued to
  detections, verified on stills incl. the last second)
