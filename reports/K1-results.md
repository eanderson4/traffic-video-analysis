# K1 — velocity-fusion kinematics: results

Branch `kinv2`, evaluated on the EP-03 hero clip (`testdata/hero`, 3840x2160
@ 25 fps, 426 frames) **using the v1 partially-corrupted homographies** —
proving speed robustness against bad registration was the point.

Pipeline: `tva flow` (LK z_v measurement + cache, ~50 s) → `tva world`
(fused CA-Kalman + RTS, MP prefilter, wavelet onsets, ~15 s) → `tva qa`.
Baseline = v1 position-only artifacts (`world_tracks.baseline.json`),
scored with the *same* QA code (`qa/report-baseline.json` vs
`qa/report-k1.json`).

## Metrics vs baseline

| metric | baseline (v1) | K1 fused | target |
|---|---|---|---|
| M4 jerk share (\|jerk\| > 15 m/s³ eq.) | 0.114 | **0.0073** | ~0 |
| M4 stopped-run median speed (car_len/s) | 0.059 | **0.041** | — |
| M4 stopped runs < 0.1 car_len/s | 74.4 % | **88.8 %** | pinned at 0 |
| M4 integral-consistency drift (median / p90) | 0.0002 / 0.0039 | 0.0002 / **0.0025** | ~0 |
| parked-car z_v signed bias (car_len/s) | n/a | **0.030 → PASS** (< 0.05) | < 0.05 |
| parked-car z_v magnitude median (car_len/s) | 0.098 | 0.100 | (noise floor, see notes) |
| M5 stop events | 38 | **38** | ~baseline count |
| M5 wave speed (Theil-Sen, onsets on front) | 15.2 km/h | **15.7 km/h** | 10–25 band |
| M5 bootstrap 95 % CI | [5.7, 21.0] (width 15.3) | **[13.2, 21.6] (width 8.4)** | tighter than baseline |
| M5 front-fit residual MAD | 99.9 px | **13.7 px** | — |
| M5 field-slope cross-check | 12.2 km/h | 14.1 km/h | agrees |

Wave direction: backward (against the traffic direction of the event
tracks) in both runs. Scale prior: car_len ≈ 55.6 px ≈ 4.5 m. The
hysteresis-onset variant gives 19.3 km/h CI [13.3, 25.4] (n=10 on front);
the wavelet onsets give the tighter, more consistent fit (n=6, MAD 13.7 px).

## Speed profiles at the corrupted clip ends (M6-style visual)

`testdata/hero/qa/speed-profiles-baseline.png` vs
`testdata/hero/qa/speed-profiles-k1.png`, tracks 9/33/35/36/45/49
(inspected, not just generated):

- Baseline: every track ramps to **200–250 px/s (≈ 60–70 km/h) in the last
  1.5 s** — pure registration blowup — and oscillates 0→90→30 px/s in the
  first second while the drone translates.
- K1: last 1.5 s stays **bounded ≤ ~70 px/s and monotone-plausible** (a
  gentle rise consistent with the upstream jam pockets visibly releasing
  after t≈13 s); first second starts smooth with no oscillation; stopped
  segments sit at 0–2 px/s; track 33's deceleration into its 7 s stop rides
  the z_v envelope and hits zero exactly at the onset.

## What it took (the actual findings)

1. **z_v cancels accumulated chain drift, not per-frame slip.** The
   world displacement factors as J_f · (relative-homography-compensated
   image displacement), so all corruption enters via (a) the local Jacobian
   scale J_f and (b) the per-frame chain increment error. Fixes:
   normalize z_v by J_f (`load_flow`), and measure the slip.
2. **Ground background is the wrong slip anchor.** Measuring per-frame slip
   on vehicle-masked background corners made parked z_v *worse* (0.085 →
   0.130 car_len/s): the background is the ground plane below the elevated
   deck, and under drone translation its parallax differs from the deck's.
   The bg bias is still cached in flow.json but deliberately unused.
3. **Static vehicles are the right anchor, and the slip is a field, not a
   vector.** A global per-frame bias vector removed only ~15 % of static
   z_v; a per-frame **affine slip field** fit over static vehicles
   (`ZvBias`, pooled ±2 frames, robust re-fit) works, and doubles as a
   trust signal: field magnitude × 0.5 s inflates R_p (coherent drift =
   box centers lying), field residual MAD inflates R_v (LK/registration
   breakdown, e.g. motion blur at the ends). Jerk share 0.31 → 0.033 from
   these two, → 0.0073 after lowering white-jerk intensity to 0.35 car_len
   (affordable now that crispness comes from z_v, not loose process noise).
4. **A single line through all stop onsets is meaningless here.** The clip
   holds several jam pockets at once plus internal creep re-stops; residual
   MAD of the naive all-onsets Theil-Sen was 350–500 px with sign flips
   between runs. The estimator that works (and matches Ji et al.'s wave
   formalism): largest connected slow component of the Edie-binned
   (t, station) field → its upstream boundary is the stop-wave front →
   Theil-Sen + bootstrap through the onsets lying on that front.
   Station axis = robust PCA of the event cloud (the spine-track centerline
   covered only ~1/3 of the road and clamped stations at its ends).

## Honest failure notes

- **Parked z_v magnitude criterion fails as literally specified** (median
  |z_v| 0.10 vs 0.05 car_len/s). After slip-field correction the *signed*
  per-track bias passes (0.030), i.e. the measurement is unbiased; the
  magnitude number is the per-measurement noise floor (~0.2 image px/frame
  at 25 fps, set by LK sub-pixel matching + relative-homography residual at
  EST_WIDTH=1920). Passing the magnitude form needs quieter registration
  (R1) or sub-pixel corner refinement in `flow.py`, not fusion changes.
  Also: v1 "static" flags include gridlock creepers, so some of that
  magnitude is real motion.
- **Jerk share is 0.7 %, not 0.** Violations concentrate at t = 13–16 s
  where the drone translation is hardest; mid-clip is ~0.01–0.02. The
  remaining wiggle is the filter negotiating z_p drift vs gated z_v.
- **Only 6 wavelet onsets lie on the wave front**, so the M5 CI rests on a
  small n (bootstrap accordingly wide, [13.2, 21.6] km/h). The independent
  field-boundary slope (14.1 km/h) and hysteresis-onset fit (19.3 km/h)
  bracket it; all in band.
- **11 % of stopped runs read > 0.1 car_len/s — much of that is real.**
  In those runs z_v itself reads 0.38 car_len/s median: gridlock creep,
  not estimator noise. "Pinned at 0" holds for genuinely stationary cars.
- **z_v measurements themselves still blow up in the final second** (LK on
  motion-blurred 4K + Jacobian distortion). They are rejected by the
  Mahalanobis gate + MAD-inflated R_v rather than fixed; end-of-clip speeds
  therefore lean on the CA prior + surviving measurements. Bounded and
  plausible, but not evidence-rich — R1's registration rewrite remains the
  structural fix.
- `render` untouched: renders stay glued to detections; smoothed states
  drive color/labels only, so published positions diverging from corrupt
  z_p at the ends is acceptable by design.
