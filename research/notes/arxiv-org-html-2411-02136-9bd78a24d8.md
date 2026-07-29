# notes: https://arxiv.org/html/2411.02136

### https://arxiv.org/html/2411.02136

## Source-attributed Factual Notes

**Source**: ArXiv preprint (2411.02136) of accepted manuscript for *Transportation Research Part C: Emerging Technologies*, DOI: `10.1016/j.trc.2025.105205`.  
**Publication Status**: Accepted version, pre-typesetting. Final version published 2025.

---

### Publication Info
- **Authors & Emails**: Robert Fonod (`robert.fonod@ieee.org`), Haechan Cho (`gkqkemwh@kaist.ac.kr`), Hwasoo Yeo (`hwasoo@kaist.ac.kr`), Nikolas Geroliminis (`nikolas.geroliminis@epfl.ch`).
- **Affiliations**: 
  - EPFL – School of Architecture, Civil and Environmental Engineering, Lausanne CH-1015, Switzerland.
  - KAIST – Department of Civil and Environmental Engineering, Daejeon 34141, South Korea.
- **Dataset DOIs**: Songdo Traffic dataset [28], Songdo Vision dataset [29]. Code repositories [30,31,32].

---

### Experiment Details (Songdo, South Korea, Oct 4–7, 2022)
- **Location**: Songdo International Business District, South Korea.
- **Date**: October 4–7, 2022. October 3 (planned first day) excluded due to strong winds, rain, fog.
- **Equipment**: 10 commercial-off-the-shelf DJI Mavic 3 drones, operated by certified pilots.
- **Coverage**: 20 busy intersections (labels A to U, excluding D which denotes drones). See Fig. 1 of source.
- **Flight parameters**:
  - Altitude: 150 meters (some 140 m to mitigate collision risk).
  - Perspective: bird’s-eye view (BEV).
  - Video: 4K UHD (3,840 × 2,160 pixels) at 29.97 fps.
  - Data collection: 10 daily flight sessions (5 morning AM1–AM5, 5 afternoon PM1–PM5), each ~30 min, cycle time 40 min.
  - Total flights: 400 flights over 4 days → 12 TB raw video data.
- **Ground Control Points (GCPs)**: 13 RTK-calibrated GCPs, each with 1 × 1 m QR code, placed across study area. Calibrated using EMLID RS2+ multi-band RTK-GNSS receiver.
- **Orthophoto**: Created with DJI Mavic 2 Pro at 75 m altitude, capturing overlapping 6,000 × 4,000 px images. Resolution: 209,181 × 181,692 pixels, file size >30 GB. Stitched using Dromii’s Drone as a Service platform.

---

### Datasets
#### Songdo Traffic Dataset [28]
- **Size**: ~700,000 unique vehicle trajectories.
- **Frequency**: 29.97 points per second.
- **Metadata per trajectory**: Vehicle ID, type, dimension estimates, IS0-formatted timestamp per point, instantaneous speed & acceleration, road section & lane number, visibility flag.
- **Coordinate systems**: Global geographic (EPSG:4326, WGS84), local Cartesian (EPSG:5186), orthophoto coordinates.

#### Songdo Vision Dataset [29]
- **Size**: 5,419 human-annotated video frames. Split: 80% training, 20% test (preserved public split).
- **Total annotations**: Nearly 300,000 vehicle instances across 4 classes: cars (including vans & light-duty), buses, trucks, motorcycles. Also pedestrians & bicycles annotated in training but not used at inference.
- **Annotation effort**: ~300 cumulative hours using Microsoft Azure ML Studio. Three iterative rounds of validation.
- **Class distribution (training/test ratios)**:
  - Car: 195,539 / 49,508 (20.2% test)
  - Bus: 7,030 / 1,759 (20.0% test)
  - Truck: 11,779 / 3,052 (20.6% test)
  - Motorcycle: 2,963 / 805 (21.4% test)
  - Total vehicles in Songdo Vision: 217,311 train + 55,124 test.
- **Annotation formats**: COCO, YOLO, Pascal VOC.

---

### Methodology Highlights

#### Object Detection
- **Detector**: Anchor-free YOLOv8 (scale 's'), input resolution 1,920 × 1,920 px.
- **Training**: Multi-stage: first on BASE dataset (8 public + Songdo Vision), then fine-tuned on FINE dataset (subset of high-quality images). See Table 1 in source for dataset list (CARPK, PUCPR+, CyCAR, UAVDT, HARPY, RAI4VD, UIT-ADrone, VisDrone, plus Songdo Vision).
- **Hyperparameters** (best): confidence threshold τd=0.25, NMS IoU threshold ηIoU=0.7.
- **Data augmentation**: Random scaling ±50%, translation ±10%, horizontal flip 0.5, hue/sat/bright perturbations (0.015/0.7/0.4), mosaic augmentation (prob=1.0), Albumentations (gaussian & median blur, grayscale, CLAHE each prob=0.01).
- **Training**: SGD with lr=0.01, momentum=0.937, weight decay=0.0005, batch size=8, early stopping patience=50 epochs, AMP enabled. Loss weights: box 7.5, class 0.5, DFL 1.5.

#### Object Tracking
- **Algorithm**: BoT-SORT [80] with embedded EMC (sparse OF). Track buffer T_buff=30 frames.
- **Association thresholds**: New track confidence >0.6, high-confidence match >0.5, low-confidence match >0.1, matching threshold 0.8.
- **Class label refinement**: After tracking, each vehicle’s class is determined by argmax of summed confidence scores across all frames.

#### Track Stabilization
- **Approach**: Homography-based registration of each frame to reference frame (first frame of video). Exclusion masks derived from detected bounding boxes (enlarged ϵ=0.15) prevent keypoint detection on vehicles.
- **Feature detector**: ORB (max keypoints: K=2,000 per frame, K_ref=4,000 for reference).
- **Matcher**: Brute-force with SNN ratio θ_SNN=0.9.
- **Robust estimator**: MAGSAC++ (τ=0.999999, max iterations Γ=5,000, epipolar threshold η=2).
- **Downscaling**: ρ=0.5 (reduce 4K frames to ~1,920 × 1,080).
- **Performance metrics**: Proposed MIoU (mean IoU over re-projected bounding boxes) in addition to HEA. Implementation available as `stabilo` library [31]; tuning tool `stabilo-optimize` [32].
- **Validation**: Used 29 unique aerial scenes, 100 trials per scene (photometric + homographic distortions) → 2,900 evaluations per combination.

#### Georeferencing
- **Process**: 3-step: (1) reference frame → master frame (intermediary, chosen per intersection for consistency), (2) master frame → orthophoto cut-out, (3) orthophoto coordinates → geographic (affine from metadata).
- **Key detail**: Master frame approach ensures uniform transformation accuracy across multiple drone viewpoints.
- **Georeferencing parameters**:
  - Detector: RSIFT (max keypoints 250,000, small epsilon 1e-8).
  - Matcher: Brute-force with SNN ratio 0.55.
  - Robust estimator: MAGSAC++ (τ=0.999999, Γ=10,000, η=3).
  - Orthophoto cut-out resolution: chosen 8,000 × 8,000 px based on ablation study (reprojection error ~1.323 ± 0.970 px, mean inliers 198.25, min 82).
- **Road segmentation**: Manual segmentation of each cut-out (50 working hours). Sections labeled "N_G" (N=intersection, G=direction/group), lanes numbered from innermost outward.

#### Vehicle Dimension Estimation
- **Method**: Uses both un-stabilized and stabilized bounding boxes; filters by azimuth (maximum deviation θ̄=15°) and ratio criteria; computes first quartile of accepted BB sizes. See Appendix A of source.

#### Speed & Acceleration
- **Computation**: Linear interpolation to fill gaps, Gaussian smoothing tuned against AV data (Section 4.2). No interpolation or smoothing on raw trajectory data.

---

### Performance Metrics

#### Detection Performance (on Songdo Vision test set, including Songdo Vision train set)
| Class      | Precision | Recall | mAP@50 | mAP@50-95 |
|------------|-----------|--------|--------|-----------|
| All        | 0.911     | 0.935  | 0.951  | 0.711     |
| Car        | 0.979     | 0.981  | 0.992  | 0.835     |
| Bus        | 0.952     | 0.977  | 0.988  | 0.826     |
| Truck      | 0.887     | 0.916  | 0.935  | 0.722     |
| Motorcycle | 0.827     | 0.866  | 0.888  | 0.463     |

- **Mean Euclidean error for BB centers**: 2.21 ± 1.99 px (12.2 ± 10.9 cm) on test set.

#### Ablation (without Songdo Vision training data):
- mAP@50 dropped by 28.15%, mAP@50-95 by 41.35%. Motorcycle mAP@50 dropped by 78.16%, mAP@50-95 by 135.03%.

#### Stabilization Validation:
- Proposed MIoU metric and HEA. For selected parameters (ORB, downscale 0.5, SNN 0.9) see Fig. 7 in source – achieved high MIoU/HEA competitive with state-of-the-art.

#### Georeferencing Accuracy (Ablation Study):
- Best orthophoto cut-out resolution: 8,000 × 8,000 px → mean reprojection error 1.323 ± 0.970 px. Minimum inliers 82 (above threshold ~30).

---

### Validation with Autonomous Vehicle (AV)
- **AV data source**: Instrumented autonomous vehicle provided by SCIGC (Stanford Center at Incheon Global Campus), equipped with high-precision RTK-GNSS.
- **Event**: Collision occurred during battery swap (not recorded), but resulting traffic disruptions documented. AV trajectories compared with drone-derived trajectories – "very high consistency" between the two measurement methods.

---

### Resources & Open Source
- **Datasets**: Songdo Traffic [28] and Songdo Vision [29] – publicly released.
- **Code**: 
  - Extraction pipeline [30]
  - Stabilization library `stabilo` [31]
  - Optimization tool `stabilo-optimize` [32]
- All URLs not fully reproduced here but indicated as public.

---

### Relevant Leads
- **Organizations**: EPFL, KAIST, SCIGC (Stanford Center at Incheon Global Campus), Dromii (DaaS platform).
- **Datasets**: Songdo Traffic, Songdo Vision, CARPK, PUCPR+, CyCAR, UAVDT, HARPY, RAI4VD, UIT-ADrone, VisDrone.
- **Software**: `stabilo` [31], `stabilo-optimize` [32] – likely on GitHub under KAIST/EPFL.
- **Commercial platform**: Microsoft Azure ML Studio (annotation), Dromii’s DaaS (orthophoto stitching).
- **Hardware**: DJI Mavic 3, DJI Mavic 2 Pro, EMLID RS2+ RTK-GNSS.
- **Journal**: Transportation Research Part C: Emerging Technologies (final version available at DOI above).
- **Contact emails** of authors: `robert.fonod@ieee.org`, `gkqkemwh@kaist.ac.kr`, `hwasoo@kaist.ac.kr`, `nikolas.geroliminis@epfl.ch`.
