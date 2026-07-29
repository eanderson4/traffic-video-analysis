# tva v2 plan

Distilled from the four research reports in `research/` (2026-07-29):
`end-to-end-systems.md`, `camera-registration.md`, `detection-segmentation.md`,
`tracking-kinematics.md`. Each claim below is cited there.

## Verdicts on v1

What the literature validates:
- World-plane-first, track stabilization (transform coordinates, never warp
  pixels) — exactly the Geo-trax design.
- CA-Kalman + RTS smoothing with v,a taken from filter states — literally the
  highD/inD recipe; our adaptive measurement noise was published independently
  as "adaptive EM-RTS" (Measurement, 2026).
- Hysteresis stop classification, space-time diagrams, static road/lane layer
  computed once on the background plate — all standard practice.

What the literature says we got wrong:
- **Chaining homographies is the anti-pattern.** Nobody chains. highD,
  Geo-trax/stabilo, CitySim all register every frame directly to one
  reference frame (graph depth ≤ 2). Our drift + end-of-clip projective
  blowup is the predictable consequence.
- **We anchored on vehicles instead of masking them out.** stabilo's core
  trick is the inverse of ours: dilate all detected vehicle boxes ~15% and
  *exclude* them from registration features. In a jam, most Shi-Tomasi
  corners sit on (possibly creeping) cars — poison.
- **We registered to the wrong plane.** The elevated deck is where every
  measured vehicle lives; features should come from the deck only (site
  polygon mask). Geo-trax masks tall structures for exactly this reason.
- **Speed noise is a measurement-model problem, not only a smoothing
  problem.** Bbox-center differencing has a known better alternative:
  per-vehicle LK optical flow as a direct velocity measurement fused in the
  Kalman (standard in UAV navigation and fixed-camera speed work; absent
  from the drone-dataset world — a genuine differentiator for us).

## v2 architecture

### Registration (rewrite of stabilize/plate/refine)
1. Reference frame = sharpest mid-clip frame (blur-scored). Small keyframe
   set (~every 2 s, blur-gated); every frame matches to nearest keyframe,
   keyframes match to reference. No chains longer than 2 edges.
2. Feature masks: all detected vehicle bboxes dilated 15% excluded; optional
   per-site deck polygon (annotator) restricts features to the measurement
   plane. Ground/buildings excluded by the same polygon.
3. Estimator: `cv2.findHomography(..., cv2.USAC_MAGSAC)` with tight
   threshold. ORB/SIFT descriptors matched globally (survives fast motion,
   unlike LK chaining). Evaluate `pip install stabilo` before writing our
   own (MIT-ish, battle-tested in Geo-trax; supports masks + reference-frame
   trajectory stabilization out of the box).
4. Per-frame QA emitted by the module itself: static-landmark reprojection
   residual, local-scale (Jacobian) map, blur score, block-wise homography
   spatial variance as a parallax/projective alarm. Frames failing gates get
   flagged; downstream weights measurements accordingly.
5. Later tier (structural fix, separate phase): pycolmap sequential SfM on
   masked frames → per-frame poses → deck plane fit → ray-plane
   intersection per detection. Kills parallax and projective blowup exactly.
   Relief-displacement correction (levelX gen-2: offset ∝ height/altitude,
   radial from principal point) is the cheap interim for vehicle-height bias.

### Kinematics (extend kinematics.py)
1. Two measurements per track per frame into one CA Kalman + RTS:
   - z_p: box center through the frame homography (current), adaptive R_p
     (keep Jacobian scaling + residual reweighting; add detection-confidence
     scaling, NSA-style).
   - z_v: median world-plane displacement of pyramidal-LK corners inside the
     bbox eroded ~20-25%, endpoints mapped through their own frames'
     homographies (auto-cancels drone motion). Own R_v, Mahalanobis-gated.
     Reads identically 0 for stopped cars — crisp stop onsets by
     construction.
2. Pre-filter before smoothing (Montanino-Punzo): impossible-point removal,
   speed/accel spike cuts.
3. Stop onset: hysteresis unchanged (it matches Ji et al.'s critical-speed
   wave-front definition); onset localized by Mexican-hat wavelet-energy
   peak on the speed series (Zheng 2011) with sub-frame interpolation.
4. Tracker: BoT-SORT config, ReID off, GMC off (homography is better GMC),
   track_buffer ~2 s; offline tracklet stitching gated by world distance +
   time gap, RTS interpolates the joins.

### Wave measurement (new module)
1. Primary: Theil-Sen (or RANSAC) line through stop-onset (t, station)
   points per wave; bootstrap CI. Slope = wave speed.
2. Cross-checks: Radon/autocorrelation slope of the Edie-binned speed field;
   Newell pairwise (τ, δ) fits on same-lane pairs.
3. Sanity band: 10-25 km/h backward (Sugiyama ~20, I-24 10-20, ASM prior
   15). Outside the band → suspect scale before believing it.

### Scale (new, three-tier API, tier recorded in metadata)
a. Reference-frame → orthophoto/satellite georegistration (2-4 clicked
   correspondences; ~25 cm absolute, <1% scale).
b. Altitude + intrinsics GSD (needs flight metadata).
c. Scene priors: lane width 3.66 m; fleet-mean car length ~4.5 m over ≥50
   classified cars. Priors double as QA on tiers (a)/(b).

### QA (new module, per-run report — "a module, not a vibe")
- Punzo 2011: jerk share (|jerk| > 15 m/s³ ≈ 0), ∫v-vs-x consistency drift,
  spectral noise floor.
- MiTra jump screens: lon steps > 2 m/frame, lat > 0.5 m/frame.
- Stabilization residual per frame (static landmarks); fragmentation rate.
- Macroscopic: wave speed in sanity band.
- Speed-accuracy realism: Berghaus-validated ceiling is ±0.45 m/s, ±0.3
  m/s² — don't claim better.

### Detection (later phase; current VisDrone weights stay for now)
- Path: YOLO11-OBB (DOTA-pretrained) fine-tuned on DroneVehicle + UAV-OBB +
  ~300 self-labeled frames (SAM2-assisted OBB from existing tracks), with
  directional-blur augmentation. OBB gives steadier centers + free heading
  (~15-20% trajectory-consistency gain, lower speed RMSE — Riehl 2025,
  TSTP 2025).
- Jitter micro-benchmark first (no published comparison exists): box center
  vs OBB center vs SAM2 mask centroid vs weighted-NMS, scored with the
  arXiv 1611.06467 stability metric.
- SAHI: audit only (gains concentrate <30 px; our cars are 40-100 px).

## Phasing

- **Phase A (now):** registration v2 (direct-to-reference + masks +
  MAGSAC++) and kinematics v2 (velocity fusion + prefilter + wavelet
  onsets) + QA module. Hero clip is the testbed; success criteria in
  `v2-experiments.md`.
- **Phase B:** wave-speed estimators, scale tiers, spacetime/overlay polish
  for the episode short.
- **Phase C:** OBB fine-tune, jitter micro-benchmark, pycolmap ray-plane
  tier, satellite georegistration.

## Constraints

- torch stays pinned at 2.9.1 / torchvision 0.24.1 (shared env with the
  diarization stack). Any pip install must not touch torch/torchvision/
  fsspec — verify with `pip show torch` after installs; use `--no-deps` and
  vendor code if a package insists.
- Renders must stay glued to detections (never draw smoothed estimates as
  marker positions) — smoothing drives color/state/labels only.
