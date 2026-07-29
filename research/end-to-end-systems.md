# How existing systems extract vehicle trajectories from aerial/drone video

Research survey for the `tva` library (2026-07-29). ~29 sources consulted; condensed
receipts in `research/notes/`. Our baseline: per-frame chained homography to frame-0
(Shi-Tomasi + LK + RANSAC), VisDrone-YOLOv8, ByteTrack, Kalman/RTS speed smoothing,
stop-onset events, space-time diagrams. Pain points: chained-homography drift, parallax
on elevated roads, speed precision.

---

## 1. pNEUMA (Athens drone swarm) and pNEUMA Vision

**What it is.** 10 drones over 1.3 km² of downtown Athens, 5 days, ~500,000 trajectories
at 25 fps. The reference urban drone-trajectory dataset.
- Site: https://open-traffic.epfl.ch/ (data: https://zenodo.org/records/10491409)
- Paper: Barmpounakis & Geroliminis (2020), *TR-C*, "On the new era of urban traffic
  monitoring with massive drone data: The pNEUMA large-scale field experiment."

**Pipeline.** Crucially, pNEUMA did **not** open-source extraction: trajectory extraction
was outsourced to **DataFromSky** (RCE Systems s.r.o., Brno) — stated on
https://open-traffic.epfl.ch/index.php/about/. DataFromSky's approach (per their papers
and the MiTra description below): video stabilization + georegistration against
real-world coordinates, DL detection, **Multiple Hypothesis Tracking + Kalman filtering**.
Output CSVs: per-vehicle rows with lat/lon, speed (km/h), lon/lat acceleration at 25 Hz.

**Accuracy lineage.** The methodological accuracy basis is Barmpounakis, Vlahogianni,
Golias & Babinec (2019), *Transportation Letters* 11(6), "How accurate are small drones
for measuring microscopic traffic parameters?" — key finding: **accuracy is dominated by
video stabilization and the georeferencing procedure**, not the detector. Earlier:
Babinec & Apeltauer (2016), *IJTST* 5(3), on position-estimation accuracy from low-flying
UAVs (same conclusion: distortion correction + calibration are the crucial phases).

**pNEUMA Vision** (Kim, Anagnostopoulos, Barmpounakis, Geroliminis 2023, *TR-C* 147,
DOI 10.1016/j.trc.2022.103966; data https://zenodo.org/records/7426506; toolbox
https://github.com/shgold/pNEUMA-Vision-toolbox) adds the imagery back: every 10th frame
with image-coordinate vehicle positions and **azimuths** (used to build rotated bounding
boxes). Its second contribution is directly relevant to us: **regularized anomaly
detection to denoise vision-extracted trajectories**, distinguishing stationary vs
non-stationary errors (non-stationary errors dominate), and showing visually-restricted
trajectories (edge of frame, occlusion) are the ones that become anomalies. They also use
density-map estimation (segmentation-based counting) and show density-map space-time
diagrams identify congested queues better than detection-based counting.

**Open reimplementations / ecosystem.**
- https://github.com/shgold/pNEUMA-Vision-toolbox — bbox reconstruction from position+azimuth.
- https://github.com/JoachimLandtmeters/pNEUMA_mastersproject — map-matching pNEUMA
  trajectories to OSM network, macroscopic quantities (loop-detector emulation).
- https://github.com/EPFL-ENAC/pNEUMA — interactive web visualization (flow/density/speed).
- No open reimplementation of the extraction itself exists — that gap is what Geroliminis'
  lab later filled with Geo-trax (Section 4).

## 2. The levelX family: highD / inD / rounD / exiD

Created by RWTH Aachen (ika/fka), commercialized as leveLXData (https://levelxdata.com/).
These are the gold-standard trajectory datasets for AV validation.

**highD** (Krajewski, Bock, Kloeker, Eckstein, ITSC 2018, https://arxiv.org/abs/1810.05642,
https://www.highd-dataset.com/): 60 recordings, 6 German highway sites, ~420 m per site,
110,000 vehicles, 4K@25fps.
Pipeline details (see `notes/emergentmind-highd-pipeline-manual.md`):
- **Positioning**: DJI Phantom 4 Pro hovering **next to** the highway (not overhead) to
  reduce perspective distortion; ~10x10 cm ground pixel; sunny windless conditions only.
- **Stabilization**: every frame's background registered **directly to the first frame**
  (anchor registration, not chained), OpenCV; first frame rotated so lane markings are
  axis-aligned (makes lon/lat decomposition trivial).
- **Detection**: U-Net semantic segmentation → pixel clusters → boxes (not a box detector).
- **Tracking**: simple distance-based matching (clean because detection is near-perfect).
- **Smoothing**: **Rauch-Tung-Striebel smoother with constant-acceleration model** for
  position/speed/acceleration in both axes.
- **Reported accuracy**: ~99% detection rate, ~2% FP, mean box-midpoint position error
  **< 3 cm** vs manual labels; commonly cited as "typically < 10 cm" trajectory accuracy.
- **Postprocessing we don't do**: per-track maneuver labels (free driving / following /
  critical / lane change), surrounding-vehicle graph (leader/follower + adjacent lanes),
  and safety surrogates **DHW/THW/TTC precomputed per frame**; lane-change
  parameterization (quadratic lon, 5th-order lat polynomials).

**inD** (Bock et al., IV 2020, https://arxiv.org/abs/1911.07602): same recipe at urban
intersections, adds VRUs (pedestrians/cyclists), 11,500 road users. Explicitly "methods
similar to highD"; positioning error typically < 10 cm claimed.

**rounD / exiD** (Krajewski et al. ITSC 2020; Moers et al. IV 2022): roundabouts and
motorway ramps. exiD provides georeferenced position, heading, lon/lat velocity and
acceleration + lane assignment against HD maps (OpenDRIVE). The second-generation levelX
pipeline is described in "Vehicle Position Estimation with Aerial Imagery from Unmanned
Aerial Vehicles" (Kruber et al., IV 2020, arXiv:2004.08206): DL detection + **relief
displacement correction** — because the drone is not an orthographic camera, a vehicle's
roof is displaced radially outward from the image principal point proportional to
(vehicle height / flight altitude); they correct the box center back to the road plane.
**This is exactly our elevated-carriageway parallax problem** (elevation of the road adds
a constant-ish offset; vehicle height adds a per-class offset).

## 3. CitySim (UCF SST Lab)

Zheng, Abdel-Aty, Yue et al. (2022→TRR 2023), https://arxiv.org/abs/2208.11036, repo
https://github.com/UCF-SST-Lab/UCF-SST-CitySim1-Dataset. 1,140 min of 4K@30fps at 12
locations (freeway, weaving, intersections). Explicit **five-step pipeline** (full text
extracted, see scratch notes):
1. **Video stabilization**: SIFT features from first & last frames → stabilize sampled
   random frames → build a **vehicle-free median background ("accumulated weighted
   frame")** → register *every* frame to that background via homography. CSRT tracker as
   fallback matcher when SIFT fails (e.g. swaying vegetation at high altitude).
2. **Object filtering**: Gaussian-mixture background subtraction to mask out static
   false-positive sources (road markings etc.).
3. **Multi-video stitching**: histogram color matching across drones + SIFT on overlap
   regions + blending (longer corridors from multiple drones).
4. **Detection & tracking**: Mask R-CNN instance masks → **rotated bounding boxes aligned
   with heading** (they demonstrate axis-aligned boxes inflate vehicle size on curves/turns);
   CSRT correlation tracker interpolates through missed detections.
5. **Enhanced error filtering**: a manual **"data fixing tool"** for human correction of
   erroneous boxes (QA-by-editor stage).
Extras we lack: 3D base maps and signal timings per site (digital-twin assets).

## 4. Geo-trax / Songdo Traffic (EPFL + KAIST) — the closest thing to what we're building

**The single most relevant open-source system.** Fonod, Cho, Yeo, Geroliminis (2025),
*TR-C* 178, DOI 10.1016/j.trc.2025.105205, https://arxiv.org/abs/2411.02136.
- Pipeline repo: https://github.com/rfonod/geo-trax (YOLO + 6 selectable trackers
  incl. ByteTrack/BoT-SORT/OC-SORT, stabilization, georeferencing, ~103 commits, active).
- Stabilization library: https://github.com/rfonod/stabilo — plus tuner `stabilo-optimize`.
- Data: Songdo Traffic (~700k trajectories, 20 intersections, 10 DJI Mavic 3 at 150 m,
  4K@29.97fps) and Songdo Vision (5,419 annotated frames, ~300k instances) on Zenodo.

Pipeline detail (full notes: `notes/arxiv-org-html-2411-02136-9bd78a24d8.md`):
- **Detection**: YOLOv8s at 1920x1920 input, trained on 8 public aerial datasets + Songdo
  Vision. 0.951 mAP@50 (cars 0.992); box-center error **2.21 ± 1.99 px = 12.2 ± 10.9 cm**
  at 150 m altitude. Ablation: dropping the domain-specific training set costs
  **-28% mAP@50** — VisDrone-style generic weights alone are measurably worse.
- **Tracking**: BoT-SORT with camera-motion compensation, 30-frame track buffer; track
  class label = argmax of **summed per-frame confidence per class** (fixes flickering
  class labels).
- **Stabilization (their answer to our drift problem)**: every frame registered
  **directly to a chosen reference frame** (anchored, never chained), with **detected
  vehicle bounding boxes (dilated 15%) used as exclusion masks** so moving traffic never
  contaminates the registration. ORB (2,000 kps/frame, 4,000 on reference), brute-force
  matcher SNN ratio 0.9, **MAGSAC++** (not vanilla RANSAC), frames downscaled 0.5x. They
  benchmark stabilization **ground-truth-free** with a proposed MIoU metric (mean IoU of
  re-projected boxes) across 2,900 perturbation trials. Stabilo exposes 6 classical
  (ORB/SIFT/RootSIFT/BRISK/KAZE/AKAZE) + 5 learned (XFeat, DISK, DeDoDe, KeyNet, LoFTR)
  detectors and LightGlue matching, CUDA-accelerated, as both CLI and library.
- **Georeferencing**: 3-step chain: reference frame → per-site "master frame" →
  RTK-GCP-calibrated **orthophoto** (13 QR-code GCPs measured with EMLID RS2+ RTK-GNSS;
  orthophoto stitched from a second drone's photos) → WGS84/local CRS via affine.
  RootSIFT + MAGSAC++; reprojection error 1.32 ± 0.97 px on 8k×8k orthophoto cutouts.
- **Kinematics**: linear interpolation of gaps, then **Gaussian smoothing whose kernel
  was tuned against RTK-GNSS ground truth from an instrumented AV** driving in the scene;
  raw positions released unsmoothed.
- **Dimensions**: per-track vehicle length/width = first quartile of stabilized box sizes,
  filtered to frames where azimuth deviates < 15° from motion direction.
- **Validation**: instrumented AV with RTK-GNSS compared against drone-derived
  trajectories ("very high consistency"); manual lane/section segmentation for
  lane-resolved output.

## 5. Other recent open datasets/pipelines worth knowing

- **DRIFT** (KAIST/Soonchunhyang 2025, https://arxiv.org/abs/2504.11019, repo
  **https://github.com/AIxMobility/The-DRIFT** — extraction scripts included): 9
  intersections along 2.6 km, one hovering drone per intersection at **250 m** (above
  Korean limit with waiver), YOLOv11m with **oriented bounding boxes** (mAP@50 0.994) +
  ByteTrack, ORB + BF + MAGSAC++ stabilization with a scale/rotation/translation
  "GeoAlign" fallback, orthophoto (Agisoft Metashape + EMLID GCPs) for world coordinates,
  per-lane polygon ROIs, leader/follower IDs in the CSV. QA: **fragmentation rate on a 5%
  random sample with binomial CI → 96.98% trajectory continuity**.
- **MiTra** (TU Dresden 2025, *Scientific Data* 12:1174, PMC12241643; data DOI
  10.25532/OPARA-881; https://github.com/ankitiitm/MiTra): 6 DJI Mini 2 over 900 m of
  Milan urban freeway incl. congested stop-and-go — the closest published analog to our
  phantom-jam clips. Extraction by DataFromSky (MHT + Kalman). QA numbers to copy:
  **global positioning error 25.3 cm** measured as distance error over **13 lamp posts**
  (fixed landmarks) vs extracted coordinates; plausibility screening counting
  **longitudinal jumps > 2 m/frame and lateral jumps > 0.5 m/frame** (914 lateral jumps
  across 107M timestamps); stitching lateral shift ~0.5 m flagged where drones flew 130 m
  to the side (EASA) — i.e. **parallax from oblique viewing quantified and disclosed**.
  Congested-flow validation: **stop-and-go wave speeds 4-4.6 m/s (~15-17 km/h)** from
  space-time diagrams and Edie's method (100 m × 20 s cells) — matches classical theory.
- **Berghaus et al. 2024** (RWTH, *Comm. Transp. Res.* 4:100133, open access,
  DOI 10.1016/j.commtr.2024.100133): 8,648 trajectories, 1,200 m German highway with
  off-ramp + congestion, post-trained YOLOv5, and — key for elevated roads — projection
  onto the road surface via **full 3D camera calibration** rather than a flat homography.
  Validation against **induction loops + a smartphone IMU/GPS inside one vehicle in the
  data**: speed deviation **0.45 m/s**, acceleration deviation **0.3 m/s²**. That's the
  realistic speed-precision ceiling for our kind of setup.
- **SWIFTraj** (2026, https://arxiv.org/abs/2602.22563): UAV-swarm freeway+urban
  trajectories with cross-video stitching (longest continuous track 4.5 km); QA via
  **internal consistency analysis of position vs speed** (integrate reported speed,
  compare with reported positions).
- **HIGH-SIM** (Shi et al., *Comm. Transp. Res.* 2021): helicopter-based highway
  trajectories; introduced trajectory "consistency analysis" comparisons vs NGSIM —
  the NGSIM cautionary tale (Coifman & Li 2017 found large systematic NGSIM errors) is
  the reason all modern datasets publish QA sections.
- **AutomatedTrajectoryDataExtraction** (https://github.com/mdshafaque56/AutomatedTrajectoryDataExtraction):
  small YOLOv8 pipeline whose scale estimation uses **lane geometry or median car length
  → pixels-per-meter** — the cheap-prior approach, useful as fallback.

## 6. Commercial tools

- **DataFromSky / TrafficSurvey** (https://datafromsky.com/trafficsurvey/): the market
  leader; extracted pNEUMA and MiTra. Post-recording cloud analysis of drone or fixed
  camera video (~2.9 EUR/hr), proprietary viewer, MHT + Kalman tracking, interactive
  gate/geo-registration workflow, claims 98-100% accuracy. Independent audit: Ali et al.
  2024 (https://arxiv.org/abs/2404.17212) — with true bird's-eye 90° footage, count
  errors are low and **space-mean-speed MAPE < 4%**; probe-vehicle speeds vs GPS gave
  Pearson r = 0.94, paired t-test p = 0.42 (no significant difference). But at oblique
  angles (60°, 50 m) class-count errors explode (bus/HCV confusion up to -197%). Their
  own research page (https://datafromsky.com/research-papers/) points to the
  stabilization/georeferencing-dominates-accuracy papers.
- **GoodVision Video Insights** (https://goodvisionlive.com/, Transoft ecosystem):
  cloud traffic analytics from fixed cameras and drones, "min 95% accuracy" marketing,
  turn-count/OD focus rather than research-grade trajectories. No published methodology.
- **leveLXData** (Section 2) sells custom drone trajectory campaigns with the highD-family
  pipeline.

Takeaway: commercial tools win on tooling (interactive review viewers, geo-registration
UX), not on disclosed algorithms. Nobody sells the pipeline; the open ones are Geo-trax,
The-DRIFT, and CitySim (partially).

## 7. How systems attach metric scale (cross-cutting)

Ranked by accuracy, as used in the wild:
1. **RTK-GCPs + orthophoto registration** (Geo-trax, DRIFT): survey a handful of ground
   control points once (RTK-GNSS, ~2 cm), build an orthophoto, register the video's
   reference frame to it. Gives absolute WGS84 + metric scale + cross-flight consistency.
2. **Georegistration to existing basemap/satellite imagery** (DataFromSky workflow, MiTra
   UTM output; pNEUMA): pick correspondences between the reference frame and a
   georeferenced map. Cheapest absolute-scale method; accuracy set by basemap (~10-30 cm).
3. **Full 3D camera calibration** (Berghaus 2024): intrinsics + pose from known 3D scene
   geometry, project detections onto the road surface. Handles non-planar/elevated scenes.
4. **Flight-altitude GSD** (nominal pixel size from altitude + intrinsics, highD's
   ~10 cm/px sanity anchor). Vulnerable to barometric altitude error.
5. **Scene priors**: lane width, lane-marking dash/gap cycle, median car length
   (AutomatedTrajectoryDataExtraction; common in one-off academic papers). Fine for ±5%.
No serious system relies on (5) alone; all the gold-standard datasets use (1)-(3).

## 8. How systems verify/QA trajectories (cross-cutting)

- **Manual-label detection audits**: highD (99% detection, <3 cm center error vs manual),
  Songdo Vision test split (center error in cm), DRIFT test split.
- **Independent-sensor cross-validation**: instrumented AV with RTK-GNSS (Geo-trax),
  induction loops + in-traffic smartphone (Berghaus: 0.45 m/s speed, 0.3 m/s² accel),
  probe-vehicle GPS + paired t-test (Ali/DFS), RTK probe runs (Barmpounakis 2019).
- **Fixed-landmark reprojection**: MiTra's 13 lamp posts → 25.3 cm global error; Geo-trax
  orthophoto reprojection error in px.
- **Ground-truth-free stabilization metrics**: stabilo's MIoU of re-projected boxes under
  synthetic perturbations (2,900 trials/config).
- **Kinematic plausibility screens**: MiTra jump counts (>2 m/frame lon, >0.5 m/frame lat),
  pNEUMA Vision anomaly detection (stationary vs non-stationary error decomposition).
- **Internal consistency**: integrate speeds and compare to positions (SWIFTraj, HIGH-SIM,
  Punzo's NGSIM consistency work).
- **Track-level statistics**: DRIFT fragmentation rate with confidence interval on a
  random 5% sample (96.98% continuity).
- **Macroscopic sanity**: FD/Edie's method and stop-and-go wave speeds ~4-4.6 m/s
  (MiTra); queue visibility in space-time diagrams (pNEUMA Vision density maps).
- **Human-in-the-loop repair**: CitySim's data-fixing tool; DataFromSky's manual
  cross-check ("no ID errors reported" for MiTra).

---

## What we should copy (actionable, mapped to our pain points)

**Stabilization / drift (highest priority)**
1. **Stop chaining homographies.** Every serious system anchors: register each frame
   directly to a reference frame (highD, Geo-trax/stabilo, CitySim's median background).
   Chained products accumulate drift by construction. For a 17 s clip this is a drop-in
   change with our existing LK/feature machinery.
2. **Mask out vehicles during registration** using our own YOLO detections dilated ~15%
   (stabilo's core trick). In a traffic jam, most of our Shi-Tomasi corners are currently
   on stopped cars — poison for the ground-plane homography. This alone probably explains
   a chunk of our drift.
3. **Swap RANSAC for MAGSAC++** (`cv2.findHomography(..., cv2.USAC_MAGSAC)`) with a tight
   threshold; used by Geo-trax and DRIFT. Free accuracy.
4. **Consider depending on `stabilo` outright** (pip-installable, MIT-family license —
   verify; CLI + `Stabilizer` class, learned features like LoFTR/XFeat available for
   low-texture asphalt). At minimum steal its defaults: ORB 2k/4k keypoints, SNN 0.9,
   MAGSAC++, 0.5x downscale, reference-frame anchoring.
5. **Adopt a ground-truth-free stabilization metric** (stabilo's MIoU idea): re-project
   static-region boxes/points through our transform and report residuals per clip, so we
   can quantify stabilization quality per video without manual labels.
6. If SIFT/ORB registration is flaky on a clip, CitySim's fallback is worth noting:
   build a **vehicle-free median background** from stabilized sample frames and register
   against that, with a correlation tracker (CSRT) as backup matcher.

**Parallax / elevated carriageway**
7. Implement **relief-displacement correction** (Kruber et al. 2020 / levelX gen-2):
   correct each box center toward the principal point by factor ≈ h_object/H_flight along
   the radial direction — with per-class vehicle-height priors (car ~1.5 m, van ~2 m,
   truck ~3.5 m) plus the carriageway's elevation above the reference ground plane. This
   converts our systematic parallax bias into a correctable term instead of noise.
8. Where geometry is genuinely non-planar, Berghaus 2024 shows the right answer is a
   **3D camera calibration + projection onto the road surface** rather than a single
   plane homography. A cheap version for us: fit the homography using features on the
   elevated deck only (mask everything off-deck), so the world plane is the deck itself.
9. highD's operational trick also applies at capture time: hover **beside** the road at
   max legal altitude rather than obliquely over it to minimize perspective distortion,
   and note that oblique viewing measurably degrades everything (Ali 2024: oblique DFS
   count errors up to ~200%; MiTra's 0.5 m lateral bias from 130 m side offset).

**Speed precision**
10. Keep RTS-with-constant-acceleration (that *is* the highD recipe) but **tune the
    smoothing against ground truth once**: drive a GPS/RTK-logging car (or phone, per
    Berghaus) through a recorded scene and pick smoothing parameters minimizing speed
    error — Geo-trax tuned its Gaussian kernel exactly this way. Realistic target:
    **±0.45 m/s speed, ±0.3 m/s² acceleration** (Berghaus); don't promise better.
11. Publish/keep **raw positions unsmoothed** alongside smoothed kinematics (Geo-trax,
    MiTra convention) so smoothing can be re-tuned later.
12. Use **rotated/oriented boxes** (CitySim, DRIFT, pNEUMA Vision azimuths) — YOLOv8/11
    OBB heads exist in ultralytics. Axis-aligned boxes inflate size and wobble the center
    on curved or angled roads, which leaks straight into lateral speed noise.
13. Vehicle dimensions: Geo-trax's estimator — first quartile of stabilized box sizes
    over frames where box azimuth ≈ motion heading (±15°) — is simple and robust; useful
    for us as both output and a per-track sanity feature.

**Metric scale**
14. Offer a **three-tier scale API**: (a) georegistration of the reference frame to a
    basemap/orthophoto (2-4 clicked correspondences on satellite imagery → absolute
    coordinates, the DataFromSky workflow); (b) altitude+intrinsics GSD; (c) scene priors
    (lane width / dash cycle / median car length). Record which tier produced the scale in
    output metadata. For one-off 17 s clips, (a) with Esri/Google imagery is cheap and
    matches what pNEUMA/MiTra effectively do (~25 cm absolute is achievable).

**QA (make it a module, not a vibe)**
15. Implement a `qa` report per run with the field's standard checks:
    - jump screen: counts of lon steps > 2 m/frame and lat steps > 0.5 m/frame (MiTra);
    - internal consistency: integrate smoothed speed vs measured displacement (SWIFTraj);
    - fragmentation/continuity rate over tracks with a CI (DRIFT);
    - stabilization residual metric per clip (rec. 5);
    - landmark check: user marks 2+ fixed landmarks, we report reprojection drift over
      time (MiTra lamp-post method) — this directly measures residual homography drift;
    - macroscopic sanity: shockwave speed from our space-time diagram should land in
      **4-4.6 m/s (15-17 km/h)** for stop-and-go waves (MiTra; classical theory). Great
      built-in check for the phantom-jam use case.
16. Add a lightweight **manual review/repair path** (CitySim's "data fixing tool",
    DataFromSky viewer): even an export-to-CSV + overlay-video loop where a human can
    delete/merge track IDs covers the top failure modes.

**Detection**
17. Expect and close the domain gap: Geo-trax lost **28% mAP@50** without domain-specific
    training data. Fine-tune our detector on the open aerial sets that now exist —
    **Songdo Vision** (~300k instances, COCO/YOLO formats), **pNEUMA Vision**, **DRIFT**
    annotations (OBB polygons) — rather than relying on VisDrone weights alone. Their
    train configs are in the geo-trax repo (`train/`).
18. Post-hoc track classification: assign each track's class by **argmax of summed
    per-frame confidences** (Geo-trax) instead of per-frame or majority vote.

**Repos to actually read code from**
- https://github.com/rfonod/geo-trax (`detect_track_stabilize.py`, georeferencing, QA)
- https://github.com/rfonod/stabilo (+ `stabilo-optimize`)
- https://github.com/AIxMobility/The-DRIFT (`extraction/`, `model/`, `vis/`)
- https://github.com/shgold/pNEUMA-Vision-toolbox (rotated bbox from azimuth)
- https://github.com/UCF-SST-Lab/UCF-SST-CitySim1-Dataset (formats, wiki, 3D assets)
- https://github.com/ankitiitm/MiTra (QA/plausibility scripts)
- https://github.com/JoachimLandtmeters/pNEUMA_mastersproject (network-level aggregation)

## Source list (consulted)

1. https://open-traffic.epfl.ch/ (+ /about/, /downloads/)
2. Barmpounakis & Geroliminis 2020, TR-C (pNEUMA) — via site/secondary
3. Kim et al. 2023, TR-C 147:103966 (pNEUMA Vision)
4. https://zenodo.org/records/7426506 (pNEUMA Vision data)
5. https://github.com/shgold/pNEUMA-Vision-toolbox
6. https://github.com/JoachimLandtmeters/pNEUMA_mastersproject
7. https://github.com/EPFL-ENAC/pNEUMA
8. Krajewski et al. 2018, arXiv:1810.05642 (highD) + https://www.emergentmind.com/topics/highd
9. Bock et al. 2020, arXiv:1911.07602 (inD)
10. levelxdata.com (highD/inD/rounD/exiD product pages)
11. Kruber et al. 2020 "Vehicle Position Estimation with Aerial Imagery from UAVs"
12. Zheng et al. 2022, arXiv:2208.11036 (CitySim, full PDF) + UCF-SST-Lab repo
13. Fonod et al. 2025, arXiv:2411.02136 / TR-C 178:105205 (Geo-trax/Songdo, full text)
14. https://github.com/rfonod/geo-trax
15. https://github.com/rfonod/stabilo (+ Zenodo records)
16. Songdo Traffic / Songdo Vision (Zenodo/infoscience)
17. Lee et al. 2025, arXiv:2504.11019 (DRIFT, full text) + AIxMobility/The-DRIFT
18. Chaudhari, Treiber, Okhrin 2025, Sci Data 12:1174 (MiTra, full text, PMC12241643)
19. Berghaus et al. 2024, Comm. Transp. Res. 4:100133 (sciopen full record)
20. Barmpounakis et al. 2019, Transportation Letters 11(6) (drone accuracy)
21. Babinec & Apeltauer 2016, IJTST 5(3)
22. https://datafromsky.com/ (+ TrafficSurvey, research-papers, accuracy news)
23. Ali et al. 2024, arXiv:2404.17212 (DFS veracity audit, full text)
24. https://goodvisionlive.com/ (product pages)
25. https://github.com/mdshafaque56/AutomatedTrajectoryDataExtraction
26. SWIFTraj, arXiv:2602.22563
27. HIGH-SIM, Shi et al., Comm. Transp. Res. 2021 (via secondary)
28. Coifman & Li 2017 (NGSIM critique) — context via search
29. MDPI Remote Sensing 17(3):407 (vehicle-height projection correction, 2025)
