# Detection & segmentation for aerial traffic video — research notes

Date: 2026-07-29. Context: 4K nadir/oblique drone clips of highways; current stack = VisDrone-trained YOLOv8x (`mshamrai/yolov8x-visdrone`) at imgsz 1920, world-plane tracking for speed. Pain points: bbox-center jitter → speed noise, motion-blur bias on movers, no heading from axis-aligned boxes, want road-surface + lane-line understanding.

---

## 1. Detectors and weights for aerial vehicles

### Oriented bounding boxes (OBB) — the headline finding

OBB is not just "nicer boxes"; there is now direct evidence it improves trajectory/speed quality:

- **Riehl et al. 2025 (Sci Rep / ETH Zurich), "Consistent vehicle trajectory extraction from aerial recordings using oriented object detection"** — benchmarks **18 detectors** (HBB vs OBB) on real aerial video, full pipeline with Hungarian matching + extended Kalman + **Rauch-Tung-Striebel smoothing**. Using OBB + angular information gave **~15% better internal consistency and ~20% better platoon consistency** of extracted trajectories, and better lane-coordinate reconstruction. Open-source pipeline: https://github.com/DerKevinRiehl/trajectory_analysis · paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12304479/
- **TSTP 2025 (Springer), "Towards Accurate Vehicle Speed Estimation in Aerial Views"** — directly compares YOLOv8 AABB vs OBB paired with multiple trackers for speed estimation, evaluated on speed RMSE and SNR: **"trackers combined with OBB-enabled detection exhibited lower signal fluctuations than those based on AABB localization"**, and tracker choice significantly impacts speed quality. https://link.springer.com/chapter/10.1007/978-3-032-14094-4_8
- **CitySim** (UCF, 1140 min of drone video, 12 sites) extracts **rotated boxes via Mask R-CNN masks** specifically because axis-aligned boxes corrupt position/heading; 5-step pipeline = stabilization → filtering → stitching → detect+track → enhanced error filtering. https://github.com/UCF-SST-Lab/UCF-SST-CitySim1-Dataset · https://arxiv.org/abs/2208.11036

Why AABB centers jitter on highways: an axis-aligned box around a rotated/diagonal vehicle is the AABB of the OBB, so its size and center move with tiny orientation/extent changes; OBB center is the geometric vehicle center and the long-axis angle is a free heading measurement (mod 180°, disambiguate with track velocity).

### Pretrained OBB weights

- **Ultralytics YOLO11-OBB / YOLO26-OBB** — official weights pretrained on **DOTAv1** (`yolo11n/s/m/l/x-obb.pt`, `yolo26*-obb.pt`), imgsz 1024. DOTA classes include `small-vehicle` and `large-vehicle`. YOLO26-OBB adds an angle loss that fixes boundary-discontinuity (the 179°↔0° flip that causes heading jitter) and ProgLoss+STAL for tiny objects. Docs: https://docs.ultralytics.com/tasks/obb/ · DOTA guide: https://docs.ultralytics.com/datasets/obb/dota-v2/
  - Caveat: DOTA is largely satellite/high-altitude imagery (0.1–1 m GSD). Your 4K drone frames at 2–4 cm GSD are a domain shift — expect to **fine-tune**, not use zero-shot.
- **mmrotate** (OpenMMLab) — rotated Faster R-CNN, Oriented R-CNN, RoI Transformer, RTMDet-R; large DOTA model zoo (up to ~78.9 mAP DOTA single-scale). https://github.com/open-mmlab/mmrotate · RTMDet paper (SOTA rotated real-time): https://arxiv.org/abs/2212.07784
- **Oriented-Det** (pip-installable, Apache-2.0, DOTA baselines without custom CUDA; weights at `dl4eo/oriented-det-pretrained` on HF). https://deeplearning.earth/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/

### OBB fine-tuning datasets (drone-altitude vehicles)

- **DroneVehicle** — 28,439 RGB+IR image pairs, **953k OBB annotations**, 5 classes (car, truck, bus, van, freight car), urban roads/parking. The biggest drone-vehicle OBB corpus. https://github.com/VisDrone/DroneVehicle · https://arxiv.org/abs/2003.02437
- **UAV-OBB (2026)** — 1,617 images at 1920×1080, 46,807 OBB annotations, 6 classes, **already in YOLOv8-OBB label format**, plus evaluation MP4s. Purpose-built for rotation-aware detection/tracking/traffic-flow. https://www.sciencedirect.com/science/article/pii/S2352340926002635 · https://pmc.ncbi.nlm.nih.gov/articles/PMC13092195/
- **DOTA v1/v1.5/v2** — 1.7M OBB annotations, 18 classes; use as pretrain, not target domain. https://docs.ultralytics.com/datasets/obb/dota-v2/

### Axis-aligned aerial weights (current family)

- `mshamrai/yolov8x-visdrone` (current choice) and siblings n/s/m: https://huggingface.co/mshamrai/yolov8s-visdrone
- **dronefreak VisDrone Detection Model Zoo** (updated 2026-07) — multiple YOLO sizes fine-tuned on VisDrone, loadable via `hf_hub_download(repo_id=..., filename="best.pt")`: https://huggingface.co/collections/dronefreak/visdrone-detection-model-zoo (e.g. https://huggingface.co/dronefreak/visdrone-yolov8m). Same account has a **UAVid semantic-segmentation zoo** (road-class aerial segmentation).
- ENOT-accelerated variant: https://huggingface.co/ENOT-AutoDL/yolov8s_visdrone
- **UAVDT**: benchmark code + configs (mmdetection-based) at https://github.com/KostadinovShalon/UAVDetectionTrackingBenchmark ; dataset mirror https://github.com/dataset-ninja/uavdt. UAVDT is car-centric highway/urban drone video — good val set, but few ready-made modern weights; VisDrone weights generalize reasonably.

Verdict on Q1: yes — OBB centers/headings materially improve tracking and speed estimation (two independent quantified studies + CitySim's design choice). Best path is DOTA-pretrained YOLO11/26-OBB **fine-tuned on DroneVehicle + UAV-OBB (+ a few hundred self-labeled frames)** rather than any zero-shot weight.

---

## 2. Small-object practice: SAHI, TTA, resolution, temporal consistency

- **SAHI** (Akyon et al., ICIP 2022): slicing at inference adds **+6.8 / +5.1 / +5.3 AP** (FCOS/VFNet/TOOD) on VisDrone+xView **on top of aerial-trained baselines**; adding slicing-aided *fine-tuning* brings cumulative **+12.7 / +13.4 / +14.5 AP**. https://arxiv.org/abs/2202.06934 · https://github.com/obss/sahi
  - Interpretation for us: our "naive tiling didn't help" result was a **weights problem, not a tiling problem** — a COCO model can't detect top-down cars at any scale. SAHI over VisDrone/OBB weights is a different experiment. But note SAHI's gains concentrate on objects <~30 px; at 4K nadir with cars 40–100 px, full-frame imgsz 1920 may already saturate recall. Measure before adopting (SAHI also multiplies inference cost and can add tile-boundary duplicate/NMS jitter on cars crossing seams).
- **ASAHI** — adaptive slice count/size to cut SAHI's redundant compute: https://arxiv.org/abs/2604.19233
- **TTA** (scale/flip): small AP gains, ~3× cost; fine for offline passes, but TTA-averaged boxes change the jitter statistics run-to-run — keep detector deterministic for speed measurement, use TTA only for recall audits.
- **Temporal consistency / jitter reduction** (this matters more than AP for speed precision):
  - **"On The Stability of Video Detection and Tracking"** (Zhang & Wang) — defines a **detection-stability metric** (center-position, scale/ratio, temporal continuity, measured trajectory-centric) and shows accuracy and stability are *weakly correlated*; **weighted NMS** (confidence-weighted box averaging instead of winner-take-all) improves both. Adopt this metric to score our detectors. https://arxiv.org/abs/1611.06467
  - **Seq-NMS** — re-scores/links boxes across frames; recall-oriented, offline-friendly. https://arxiv.org/abs/1602.08465
  - **StrongSORT** — two directly usable pieces: **NSA Kalman** (measurement-noise covariance scaled by detection confidence, so blurry/low-conf boxes pull the state less) and **GSI** (Gaussian-process smoothing interpolation over whole trajectories, explicitly built because "raw tracked results generally include noisy jitter"). https://arxiv.org/abs/2202.13514
  - **UncertaintyTrack** — uses detector localization uncertainty in association/filtering: https://arxiv.org/abs/2402.12303
  - Offline pipelines (ours included) should use a **forward-backward smoother (RTS)** rather than a causal Kalman — this is what Riehl et al. and the highD-family pipelines do; the drone-speed literature converges on multi-frame differencing + moving-average/sliding-window smoothing (e.g. https://juyeuav.com/monocular-vision-based-speed-detection-for-unmanned-drone-videos/ , https://www.mdpi.com/1999-4893/17/12/558).
  - Ground truth for validating speed error: the Shanghai intersection dataset pairs drone video with a **cm-level GNSS probe vehicle** — the model for how to validate our own pipeline (one instrumented car pass). https://pmc.ncbi.nlm.nih.gov/articles/PMC11404071/

---

## 3. Precision localization (sub-pixel centers)

- **Center-heatmap detectors**: CenterNet/CenterTrack represent objects as center points; peak + local offset regression gives sub-pixel centers by construction, and CenterTrack adds a learned inter-frame center offset (jitter-suppressing by design). https://arxiv.org/abs/2004.01177 · https://github.com/xingyizhou/CenterTrack. No aerial-vehicle pretrained weights of note — would need training on VisDrone/DroneVehicle; treat as a research branch, not a drop-in.
- **Sub-pixel heatmap decoding is a solved technique** in the keypoint world: soft-argmax / continuous heatmap regression removes the quantization error of discrete argmax (Bulat et al., BMVC 2021: https://www.adrianbulat.com/downloads/BMVC2021/Subpixel_Heatmap_Regression_for_Facial_Landmark_Localization.pdf ; "Learning to Make Keypoints Sub-Pixel Accurate": https://arxiv.org/abs/2407.11668). If we ever train a center/keypoint head, use sub-pixel decoding.
- **Mask centroids**: masks average hundreds of boundary pixels, so their centroids are empirically steadier than box centers. Used in practice: CitySim (Mask R-CNN → rotated box), Seg2Track++ (SAM2 masks, **Mask Centroid Distance** association + confidence-aware costs, KITTI MOTS: https://arxiv.org/abs/2606.03875), "Robust Visual Tracking by Segmentation" (https://arxiv.org/abs/2203.11191). SAM2 prompted per-track (boxes as prompts on keyframes, memory propagation between) is the cheapest route to mask centroids for us.
- **Feature-point alternative**: trajectory papers with oblique views track a **bumper point** instead of the centroid to dodge parallax/occlusion (https://pmc.ncbi.nlm.nih.gov/articles/PMC12412117/). For nadir this matters less; for oblique clips it's the right target point.
- Quantified jitter comparisons across box-center vs mask-centroid vs OBB-center are **not published anywhere I found** — the closest are the stability metric (arXiv 1611.06467) and the OBB-vs-AABB speed RMSE/SNR chapter. This is a genuinely open micro-benchmark we can run (and publish) on our own clips.

---

## 4. Road surface and lane extraction from aerial imagery

- **Datasets/models for aerial lane markings**:
  - **DLR SkyScapes** — 13 cm GSD aerial imagery, 31 classes incl. **12 lane-marking sub-classes**; the reference dataset for aerial HD-map segmentation. https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-skyscapes · https://github.com/tianzq/SkyScapes
  - **Aerial LaneNet** (Azimi et al., TGRS 2019) — wavelet-enhanced FCN for aerial lane-marking segmentation (the classic baseline; cited via SkyScapes repo).
  - **"Advancements in Road Lane Mapping"** (arXiv 2410.05717) — fine-tunes and compares **12 semantic-segmentation architectures** for aerial lane extraction; practical guide to what to fine-tune. https://arxiv.org/abs/2410.05717
  - **Vision-Based Highway Lane Extraction from UAV Imagery** (Electronics 2025) — 3-stage: enhanced STDC segmentation with boundary-aware loss → geometric constraints → lane modeling; **94.1% F1 / 93.8% IoU** at 50–150 m altitude. Closest published system to our use case. https://www.mdpi.com/2079-9292/14/17/3554
  - **Improved DeepLabV3+ for UAV highway lane lines** (https://www.mdpi.com/2071-1050/17/16/7317) and **UAV-Laneline3K** dataset, ~3000 UAV images incl. shadows/merges (https://www.mdpi.com/2072-4292/18/11/1820).
  - Aerial road-class segmentation weights: **dronefreak UAVid segmentation zoo** on HF (see §1).
- **SAM-family for roads**:
  - **SAM-Road** (CVPRW 2024) — fine-tunes SAM's encoder to output road/intersection probability masks and extracts a **vectorized road graph** (vertices via NMS on the mask); large-scale, fast. https://arxiv.org/abs/2403.16051 · https://github.com/htcr/sam_road
  - **GeoSAM** — SAM fine-tuned with auto-generated multi-modal prompts for mobility infrastructure, addressing SAM's weakness on thin aerial features. https://arxiv.org/abs/2311.11319
  - Vanilla SAM/SAM2 zero-shot struggles on thin lane paint (features blend into background) — consistent with our finding that **top-hat paint extraction on a stabilized plate** is needed alongside SAM masks. Our prototype (SAM2 prompted from trajectory centerlines + morphological top-hat) matches what the literature does with heavier training.
- **Keyframe segmentation + propagation** — validated pattern:
  - The scene is static in a stabilized/registered frame, so **segment once (or at ~1 frame/s), propagate by homography** — exactly what trajectory pipelines do: CitySim stabilizes video before everything; the stationary/nonstationary-UAV speed paper registers every frame to a **georeferenced orthomosaic** via feature matching, then all per-frame results live in one map frame (https://www.mdpi.com/1999-4893/17/12/558).
  - For propagating *object* masks through video, SAM2's memory module is the mechanism; **SAM2Long** (training-free memory-tree fix for long-video error accumulation: https://arxiv.org/abs/2410.16268) and **SAMURAI** (motion-aware memory selection for fast movers: https://arxiv.org/abs/2411.11922) are the two drop-in upgrades if we run per-vehicle SAM2 masks on long clips.
  - Recommendation: road surface + lanes are **static-layer** products — compute on the median/background plate once per scene (no per-frame segmentation at all), map into frames via the stabilization homography. Reserve SAM2 video mode for per-vehicle masks.

---

## 5. Motion blur

- **Capture side beats compute side.** Blur length in pixels ≈ (relative speed × exposure time) / GSD. At 3 cm GSD, a 30 m/s vehicle smears **~1 px at 1/1000 s but ~8 px at 1/125 s**. Drone-mapping guidance converges on fast shutters (≥1/500–1/1000, cap ~1/2000 before e-shutter kicks in on DJI): https://www.hammermissions.com/post/preventing-motion-blur-in-drone-photogrammetry-flights · https://help.inflights.com/en/articles/8646894-dji-mavic-3e-rtk-exposure-setting-for-drone-mapping-shutter-priority-mode. Action: fly shutter-priority, **log shutter/ISO from DJI SRT/EXIF sidecars**, and record exposure in clip metadata so blur length per vehicle is computable from its own tracked speed.
- **Deblur-before-detect: mostly not worth it** for our offline pipeline. Deblurring networks are heavy, and detection-oriented studies note gains are fragile and depend on the deblur model matching the blur (survey: https://arxiv.org/abs/2401.05055 ; RT-Deblur, a rare real-time detector-oriented attempt: https://link.springer.com/article/10.1007/s00371-023-02991-y ; a UAV small-object paper explicitly rejects deblur preprocessing as compute-heavy and unreliable, preferring blur-robust training: https://link.springer.com/article/10.1007/s40747-024-01676-w). Better: **fine-tune with synthetic motion-blur augmentation** (directional blur along track heading) so the detector's centers stay calibrated on movers.
- **Blur-aware confidence**: detectors already emit lower confidence on blurred movers — exploit it instead of fixing it. **NSA Kalman** (StrongSORT) scales measurement noise by confidence, and **UncertaintyTrack** uses per-box localization covariance; both convert "blurry detection" into "low-weight measurement" automatically. Note the physics: for constant velocity, blur smear is symmetric around the true center, so the bias is second-order (from the detector's edge response, not the smear itself) — which is why smoothing + confidence-weighting recovers most of it without deblurring.

---

## Recommended detection stack

**Target architecture:**

1. **Detector**: YOLO11x-OBB (or YOLO26-OBB when stable in ultralytics) pretrained on DOTAv1, **fine-tuned on DroneVehicle + UAV-OBB + ~300 self-labeled 4K frames** (label with SAM2-assisted OBB fitting from our existing tracks), with directional motion-blur augmentation. Inference at imgsz 1536–1920 full-frame; SAHI slicing only if the recall audit demands it.
2. **Measurement**: OBB center + heading (mod-180 resolved by track velocity) as the Kalman measurement; per-detection confidence feeds **NSA-style adaptive measurement noise**.
3. **Tracker/smoother**: BoT-SORT/StrongSORT-style association, then offline **RTS (forward-backward) smoothing** in world-plane coordinates; GSI-style trajectory interpolation over gaps. Speeds always differentiated from the smoothed world-plane trajectory, never from raw centers.
4. **Static scene layer**: background/median plate per stabilized scene → road mask (SAM-Road or SkyScapes/UAVid-fine-tuned DeepLabV3+/STDC) + lane paint (top-hat + geometric constraints, per the 94% F1 UAV highway-lane paper) computed **once**, propagated by the stabilization homography. No per-frame segmentation.
5. **Validation**: adopt the center-position stability metric from arXiv 1611.06467; one instrumented-car GNSS pass per site for absolute speed truth (Shanghai-dataset pattern).

**Experiment shortlist, ranked by expected impact on speed-measurement precision:**

| # | Experiment | Cost | Expected impact |
|---|-----------|------|-----------------|
| 1 | RTS smoother + NSA Kalman (confidence-weighted noise) on existing YOLOv8x-VisDrone tracks | Low — no retraining | High. Literature says most speed noise dies in the smoothing layer (StrongSORT GSI, Riehl RTS); immediate win. |
| 2 | OBB fine-tune (YOLO11x-obb on DroneVehicle + UAV-OBB + own frames), OBB-center tracking | Medium — 1–2 GPU-days + labeling | High. ~15–20% trajectory-consistency gain + free heading (Riehl 2025); lower speed RMSE/SNR vs AABB (TSTP 2025). Unlocks lane-relative kinematics. |
| 3 | Jitter micro-benchmark: box center vs OBB center vs SAM2 mask centroid vs weighted-NMS box, scored with the arXiv 1611.06467 stability metric on 3 clips | Low | Medium-high, and it de-risks #2/#5; result is publishable (no such comparison exists). |
| 4 | Directional motion-blur augmentation in fine-tune + shutter logging from SRT metadata (fly ≥1/1000 s) | Low | Medium. Removes the mover-vs-parked calibration bias at the source; cheaper and more reliable than deblur nets. |
| 5 | SAM2 mask-centroid measurement for a subset of tracks (SAM2Long/SAMURAI memory if clips are long) | Medium — SAM2 compute per track | Medium. Steadier centers where boxes are noisy; use as offline refinement pass, not the main path. |
| 6 | SAHI (1280 tiles, 0.2 overlap) over VisDrone/OBB weights — recall + stability audit vs full-frame 1920 | Low | Low-medium at 4K nadir (cars are 40–100 px); mainly matters for high-altitude/wide shots and motorcycles. |
| 7 | Static road/lane layer: SAM-Road + SkyScapes-fine-tuned marking head on the background plate, homography propagation | Medium | Indirect but strategic: lane-relative coordinates make speed/lane-change analytics robust and enable per-lane aggregation. |

Skip for now: deblurring networks (fragile, expensive), CenterTrack retraining (revisit only if OBB centers still jitter), TTA in the measurement path (nondeterministic jitter).
