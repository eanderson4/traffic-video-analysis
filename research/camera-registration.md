# Camera-motion estimation / registration for aerial traffic video

Research notes for the `tva` v2 registration stack. 2026-07-29.

Context — what v1 does and where it breaks:

- v1: per-frame homographies to frame-0's ground plane (Shi-Tomasi + pyramidal LK + RANSAC), chained frame-to-frame, patched by mosaic-drift re-registration and affine corrections anchored on static vehicles.
- **F1 (off-plane parallax):** the carriageway is elevated; drone translation causes parallax between the deck plane and the ground plane; anchoring to one plane misregisters the other.
- **F2 (projective blowup):** fast translate/tilt at clip start/end makes the chained homography strongly projective; local scale amplifies 10-14x and world-plane speeds explode.
- **F3 (motion blur):** blur during fast moves biases feature and bbox positions.

---

## 1. How the field actually handles camera motion

The consistent pattern across drone-trajectory datasets: **nobody chains frame-to-frame homographies over a whole clip.** Everyone registers each frame (or its tracks) directly to a fixed reference frame, masks out vehicles before matching, and georeferences via a master frame + orthophoto. Full SfM/VO is essentially absent from the traffic-dataset literature — because their drones hover quasi-stationary. Our translating/tilting drone is the harder, less-covered case.

- **pNEUMA (EPFL, Athens swarm)** — extraction outsourced to DataFromSky (proprietary). Kim et al. 2023 ("Visual extensions and anomaly detection in the pNEUMA experiment", TR-C) retrofit stabilization by matching the central region of each frame to a reference; they document that residual camera motion is a real error source in the published trajectories (noise, perspective distortion, human-induced errors). https://www.sciencedirect.com/science/article/pii/S0968090X22003795 , https://open-traffic.epfl.ch/
- **Geo-trax / Songdo dataset (Fonod, Cho, Yeo, Geroliminis 2025, TR-C 178:105205)** — the current best-documented open pipeline, and the closest role model. Key design choices:
  - **Track stabilization, not video stabilization**: they register frames but only transform the *track coordinates*, never resample pixels — detection/tracking runs on raw frames, avoiding warp-induced blur and wasted compute.
  - **Detected vehicle bounding boxes are exclusion masks during registration** (a cleaner version of our "provably-static vehicles" hack, inverted: exclude everything that might move rather than anchor on things proven static).
  - **Tall objects (buildings) are also masked out "to prevent parallax errors and ensure reliable planar homography"** — i.e., they hit our F1 and solved it by masking off-plane structure, which is exactly the per-region-plane discipline we lack.
  - **Master-frame + orthophoto georeferencing**: every session registers to one master frame per site; the master frame is georeferenced once against an RTK-GNSS/GCP-corrected orthophoto. No chaining.
  - Registration recipe: SIFT → RootSIFT → BFMatcher kNN + Lowe ratio → `findHomography` RANSAC.
  - Paper: https://arxiv.org/abs/2411.02136 ; code: https://github.com/rfonod/geo-trax
- **stabilo (Fonod)** — the stabilization core of Geo-trax factored into a maintained pip-installable Python library (`pip install stabilo`): stabilizes frames *or tracked-object trajectories* to a chosen reference frame, robust homography or affine, user-supplied exclusion masks, benchmarked by a companion Stabilo-Optimize project. Directly reusable for us. https://github.com/rfonod/stabilo
- **highD (RWTH Aachen, 2018)** — hovering drone over straight German highway; they do detection in image space and rely on nearly-static camera + simple stabilization; the geometry is trivially planar. Not a source of techniques for moving cameras. https://arxiv.org/abs/1810.05642
- **Berghaus et al. 2024 (off-ramp dataset, Data in Brief)** — explicit pipeline stages: video stabilization, camera calibration (lens distortion), detection; confirms stabilization+calibration as the standard preprocessing pair. https://www.sciencedirect.com/science/article/pii/S2772424724000167
- **CitySim (UCF, 2023)** — five-step pipeline starting with video stabilization, then filtering/stitching/tracking; georeferencing method not fully disclosed (criticized by Fonod et al. for that). https://arxiv.org/abs/2208.11036
- **Multi-UAV cooperative trajectory extraction (2025)** — detection → tracking → coordinate conversion → cross-drone trajectory matching; again reference-frame registration per drone. https://www.sciencedirect.com/science/article/pii/S1110016825008610

**Takeaway:** the field's answer to Q1 is "homography, but direct-to-reference with motion masks and an orthophoto-anchored master frame." Chaining is the anomaly (ours). None of them handle a genuinely translating camera over non-planar structure — for that we must borrow from SfM/stabilization literature (§2-4).

## 2. Non-planar scenes (our elevated-carriageway problem)

Options, cheap → expensive:

1. **Pick one plane and mask everything else (Geo-trax discipline).** Since all measured vehicles live on the elevated deck, the *deck* is the right registration plane. Restrict features to the carriageway surface (road segmentation mask or a hand-drawn polygon per site) and treat ground-level features as outliers, not anchors. This converts F1's whack-a-mole into a deliberate choice: register to the deck plane, measure on the deck plane. Cost: ~zero; it's a masking change.
2. **Multiple homographies / per-region planes.** Estimate one homography per planar region (deck vs ground), assign features by region mask, use the deck homography for measurement. Classical piecewise-planar registration; robust and simple when regions are known a priori (they are — the viaduct doesn't move).
3. **Mesh / spatially-variant warps** — APAP "as-projective-as-possible" moving-DLT (Zaragoza et al., CVPR 2013, TPAMI) fits a smoothly varying local homography field; handles parallax for *stitching* but the warp is not physically meaningful, so it's fine for alignment, dangerous for *measurement* (locally distorts scale). Use only as an alignment aid, never as the metric map. https://cs.adelaide.edu.au/~tjchin/apap/files/tpami_mdlt_lowres.pdf , Python re-impl: https://github.com/EadCat/APAP-Image-Stitching
4. **Plane + parallax decomposition.** Register the dominant plane; residual motion of off-plane points is a 1-parameter epipolar field (parallax magnitude ∝ height/depth). Classic theory: Triggs, "Plane + Parallax, Tensors and Factorization" (ECCV 2000, https://lear.inrialpes.fr/people/triggs/pubs/Triggs-eccv00.pdf); survey treatment in "Moving Objects Detection with a Moving Camera" (https://arxiv.org/pdf/2001.05238). Gives us a principled residual model and a way to *estimate deck height* relative to ground from the video itself. Medium cost, no new tools.
5. **Full camera pose + known 3D plane (the real fix).** Recover per-frame camera pose (SfM on the static scene, vehicles masked), define the deck as a 3D plane fitted to reconstructed deck points, then map each detection by **ray-plane intersection**. This is exact under any camera motion — no projective blowup, no parallax ambiguity, both F1 and F2 die simultaneously. With minutes of offline compute available, this is entirely practical for a 400-frame clip (§3).

**Takeaway:** (1)+(2) is the cheap fix; (5) is the correct fix. (4) is a nice diagnostic layer (measure how big the deck/ground offset actually is).

## 3. Long-sequence drift: keyframes, BA, and which tools fit 400 frames of 4K

Mosaicing literature settled this long ago: keyframes with ~30% overlap + pairwise homography links + global adjustment + loop closure beats chaining (e.g., "Building Aerial Mosaics" I/II for visual MTI, https://www.researchgate.net/publication/254951372_Building_aerial_mosaics_for_visual_MTI ; McLauchlan's sequential-BA mosaicing, https://bmva-archive.org.uk/bmvc/2000/papers/p62.pdf ; "Drift-Free Real-Time Sequential Mosaicing"). Our chain+mosaic hack is a degenerate version of this with only sequential edges.

Tool assessment for a ~400-frame 4K clip (all Python-usable):

| Tool | What it gives | Maturity / fit |
|---|---|---|
| **pycolmap** (https://colmap.github.io/pycolmap/, wheels on PyPI, now the official binding — standalone repo deprecated *into* COLMAP mainline) | Full SfM: poses + sparse points + BA, sequential matching mode for video | Mature, actively maintained, GPU optional. 400 frames at 4K (downscale features to ~2K) reconstructs in minutes. **Best fit for the offline pose track.** Feed it vehicle masks. |
| **hloc** (cvg/Hierarchical-Localization) + **LightGlue/SuperPoint** (https://github.com/cvg/LightGlue, ICCV 2023) | Learned matching frontend for COLMAP; PnP localization of new images against an existing model | Mature, standard research stack. LightGlue matches survive blur/low-texture far better than Shi-Tomasi+LK. Also the tool for frame→orthophoto matching (§6). |
| **OpenSfM** (Mapillary) | Full SfM with GPS priors, GCPs | Works, but development has slowed since the Meta acquisition; pycolmap is the safer bet. |
| **GTSAM** (`pip install gtsam`) | Factor-graph optimization: pose graph over keyframes, homography-chain relaxation, loop-closure edges | Very mature. Right tool if we keep a homography/similarity pose graph instead of full SfM: keyframe nodes, sequential + re-detection edges, optimize globally. |
| **DPVO / DROID-SLAM** (https://github.com/princeton-vl/DPVO, NeurIPS 2023; https://arxiv.org/abs/2408.01654 for DPV-SLAM) | Learned monocular VO/SLAM; DPVO = sparse, fast, low memory, exports COLMAP-format reconstructions | Research-grade but widely used; needs GPU, needs downscaled input (~512-960px). Good as a robust *initializer* when feature SfM struggles (blur segments), not as the metric backbone. |
| **DUSt3R / MASt3R / VGGT** (https://vgg-t.github.io/, CVPR 2025) | Feed-forward poses + dense pointmaps from unposed images | Cutting-edge; evaluation on aerial photogrammetry blocks exists (https://arxiv.org/abs/2507.14798) with decent pose but sub-photogrammetric accuracy, high VRAM at scale. Watch, don't adopt yet. |
| **pyceres** (cvg) | Custom BA / refinement problems in Python | Mature companion to pycolmap for bespoke costs (e.g., plane-constrained BA). |

**Takeaway:** for offline v2, pycolmap sequential SfM with vehicle masks is squarely within budget (minutes) and replaces chain+mosaic+anchor hacks with one principled optimization. GTSAM is the lighter alternative if we stay in homography-land: keyframe graph + global relaxation instead of a chain.

## 4. What video stabilization research contributes

- **Bundled Camera Paths (Liu et al., SIGGRAPH 2013)** — mesh of local homographies, per-cell camera "paths" smoothed jointly; explicitly built because "single homography cannot model parallax and rolling shutter." https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/Stabilization_SIGGRAPH13.pdf
- **MeshFlow (Liu et al., ECCV 2016)** — sparse motion field on mesh vertices, median-filtered, smoothed per vertex-profile; minimal-latency online. Python re-impl: https://github.com/sudheerachary/Mesh-Flow-Video-Stabilization
- Transfer verdict: their *diagnosis* transfers (single homography fails under parallax + RS — exactly F1/F2); their *models* mostly don't. Stabilizers optimize for smoothness and are content-free about metric truth — a mesh warp that looks stable can still locally rescale by percent-level amounts, which is fatal for speed measurement. Two things worth stealing:
  1. **Vertex-profile smoothing as QA signal**: fit MeshFlow-style local motion, and use *spatial variance of the local homographies across the frame* as a per-frame "planarity/parallax alarm" — exactly the frames where F2 explodes.
  2. **The path/warp split**: stabilizers separate "estimated camera path" from "rendered warp." Our analog is Geo-trax's track stabilization: estimate registration, transform only track coordinates, never warp pixels.

## 5. Rolling shutter and motion blur (measurement, not aesthetics)

- **Rolling shutter.** Drone cameras (DJI mainline) are rolling-shutter; readout ~10-30 ms/frame. During fast pan/translate, rows are captured at different times → skew that biases a fitted homography and shifts positions by up to several pixels; for measurement the first-order fix is cheap: model per-row time offset t(y) = t0 + y·(readout/height) and correct each detection's position using the frame-to-frame motion already estimated (velocity × row-delay). Full geometric treatments: Baker et al., "Removing Rolling Shutter Wobble" (CVPR 2010, https://www.microsoft.com/en-us/research/wp-content/uploads/2010/06/0198.pdf); unified RS+blur registration model (Meilland et al., https://hal.science/hal-01357360/document); modern learned RSC ("Fast Rolling Shutter Correction in the Wild", TPAMI 2023, https://hal.science/hal-04280534v1/document); RS+blur-aware reconstruction ("Gaussian Splatting on the Move", ECCV 2024, https://arxiv.org/abs/2403.13327 — useful for its explicit physical image-formation model driven by VIO velocities).
- **Motion blur.** Documented effects: fewer detected features, worse localization, harder matching (Peretroukhin et al., "Fast Motion Deblurring for Feature Detection and Matching", https://arxiv.org/pdf/1805.08542); detector performance gap on blurred frames with five classes of remedies (Sayed & Brostow, CVPR 2021, https://www.computer.org/csdl/proceedings-article/cvpr/2021/450900b706/1yeMmuCia9q); tracking degradation and when deblurring helps (Guo et al., https://arxiv.org/abs/1908.07904). Key measurement point: blur *centroid* bias is systematic — a blurred bbox is elongated along apparent motion and its center is the mid-exposure position, not the timestamp-nominal one; if bbox timestamps are treated as end-of-exposure this is a velocity-dependent position bias.
- Practical policy for us, cheap → expensive: (a) **blur-score each frame** (variance-of-Laplacian on static regions, or homography-derived image-velocity × exposure time as predicted blur length) and gate: exclude high-blur frames from being registration keyframes and down-weight/interpolate track points there; (b) treat detections as **mid-exposure** samples and apply the RS row-time correction; (c) only if needed, deblur high-value segments with an off-the-shelf video deblurrer before detection — evidence says heavy blur hurts and deblurring helps detection, but adds pipeline risk. Since F2's bad frames (fast moves) are the same frames as F3's blurry ones, one gating mechanism covers both.

## 6. Using known scene structure

- **Orthophoto/master-frame georeferencing** is the standard (Geo-trax: GCP+RTK orthophoto → master frame → all sessions; they report that GSD-based and manual-GIS georeferencing "often lack accuracy"). For us: register a mid-clip keyframe to a public orthophoto (state DOT orthoimagery / NAIP / Esri tiles) once, and hang the whole clip off it. Cross-modal drone↔satellite matching is a mature research area with practical recipes — hierarchical retrieval + local feature matching + PnP against satellite imagery (e.g., https://www.sciencedirect.com/science/article/pii/S0957417424029312 ; GNSS-denied UAV localization via map matching, https://arxiv.org/pdf/2103.14381). SuperPoint+LightGlue or LoFTR handle the appearance gap far better than SIFT.
- **Semantic anchors:** lane markings, gore points, bridge joints, barrier ends are permanent, high-contrast, and *on the deck plane* — exactly the plane we measure on. Detect once in the reference frame (or take from an annotated site map), then track/re-detect for registration. This also gives absolute scale (standard lane width / known dash spacing, e.g., MUTCD 10-ft dash / 30-ft cycle) independent of the orthophoto.
- **Elevation data:** the deck-vs-ground height offset (F1's root cause) can be read from lidar DEM/DSM or bridge plans; even a single scalar "deck is ~8 m above ground" lets a two-plane model be constructed analytically instead of estimated.

---

## Recommended architecture (v2)

Principle shift: **stop chaining, stop pretending one plane, stop warping pixels.** Register direct-to-reference, on the deck plane, transforming track coordinates only. Tiers are cumulative; each is shippable alone.

**Tier 0 — masking + direct-to-reference (days, no new deps).**
- Adopt the Geo-trax/stabilo pattern: every frame registers *directly* to the nearest of a small set of keyframes, keyframes register to one reference frame (graph depth ≤ 2, not a 400-link chain).
- Exclusion masks during registration: all detected vehicle bboxes (not just moving ones) + everything off the carriageway (ground, buildings). A per-site deck polygon solves F1 at the masking level: deck features only, deck plane only.
- Evaluate `stabilo` (pip) directly before writing our own — it implements exactly this (reference-frame registration, masks, track stabilization) and is battle-tested in Geo-trax.
- Per-frame QA: local-scale map of the homography + spatial variance of block-wise local homographies (MeshFlow-style) as a parallax/projective alarm; blur score. Gate or flag measurements on bad frames instead of publishing exploded speeds.

**Tier 1 — better matching + global consistency (a week).**
- Replace Shi-Tomasi+LK with SuperPoint+LightGlue (or keep SIFT/RootSIFT à la Geo-trax as the no-GPU fallback) — survives blur and large viewpoint change, enabling long-range keyframe↔reference matches during fast-motion segments.
- Keyframe pose graph in GTSAM (or a simple least-squares over homography parameters): sequential edges + all keyframe→reference edges + any re-detection edges, optimized globally. This formally replaces the mosaic-drift hack.
- Mid-exposure timestamp convention + rolling-shutter row-time correction on all track points (needs only readout time, from camera spec or calibration).

**Tier 2 — true poses + ray-plane measurement (the F1/F2 kill, ~2-3 weeks).**
- Offline pycolmap sequential SfM on vehicle-masked frames (features at ~2K resolution); calibrated intrinsics (OPENCV model) shared across frames. 400 frames = minutes.
- Fit the deck plane (RANSAC on reconstructed points inside the deck polygon); optionally a ground plane too.
- Measurement = back-project each detection ray from its posed camera, intersect with the deck plane. Exact under arbitrary translate/tilt: F1 and F2 cease to exist as geometry problems. Scale/georeference the reconstruction via one keyframe↔orthophoto registration (hloc + LightGlue) + lane-marking distances.
- Fallback for blur-dead segments where SfM feature tracks break: DPVO on downscaled frames as pose initializer, refined by pycolmap BA.

**Tier 3 — later / optional.**
- Frame-level georeferencing against satellite tiles for multi-clip, multi-day consistency (master-frame pattern from Geo-trax).
- Plane+parallax residual monitoring as continuous validation that the deck-plane assumption holds.
- Watch DUSt3R/MASt3R/VGGT for when feed-forward pose reaches photogrammetric accuracy on aerial blocks (not yet, per the UseGeo evaluation).

Rule of thumb: Tier 0-1 makes the current homography world honest and cheap; Tier 2 changes the geometry model so the two structural failures cannot occur. Blur (F3) is handled by gating + mid-exposure/RS conventions at every tier, deblurring only if gating leaves unacceptable gaps.

---

## Sources

1. Fonod et al. 2025, Advanced CV for georeferenced vehicle trajectories from drone imagery (TR-C 178:105205) — https://arxiv.org/abs/2411.02136
2. Geo-trax pipeline code — https://github.com/rfonod/geo-trax
3. stabilo Python library — https://github.com/rfonod/stabilo
4. Kim et al. 2023, Visual extensions and anomaly detection in pNEUMA (TR-C) — https://www.sciencedirect.com/science/article/pii/S0968090X22003795
5. pNEUMA open-traffic initiative — https://open-traffic.epfl.ch/
6. Krajewski et al. 2018, highD dataset (ITSC) — https://arxiv.org/abs/1810.05642
7. Berghaus et al. 2024, off-ramp drone trajectory dataset — https://www.sciencedirect.com/science/article/pii/S2772424724000167
8. Zheng et al. 2023, CitySim — https://arxiv.org/abs/2208.11036
9. Multi-UAV cooperative trajectory extraction 2025 — https://www.sciencedirect.com/science/article/pii/S1110016825008610
10. Triggs 2000, Plane + Parallax, Tensors and Factorization (ECCV) — https://lear.inrialpes.fr/people/triggs/pubs/Triggs-eccv00.pdf
11. Chapel & Bouwmans 2020, Moving object detection with a moving camera (survey; plane+parallax §5.2.1) — https://arxiv.org/pdf/2001.05238
12. Zaragoza et al. 2013/14, As-Projective-As-Possible stitching with Moving DLT — https://cs.adelaide.edu.au/~tjchin/apap/files/tpami_mdlt_lowres.pdf
13. APAP Python re-implementation — https://github.com/EadCat/APAP-Image-Stitching
14. Liu et al. 2013, Bundled Camera Paths for Video Stabilization (SIGGRAPH) — https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/Stabilization_SIGGRAPH13.pdf
15. Liu et al. 2016, MeshFlow (ECCV) — https://www.microsoft.com/en-us/research/publication/meshflow-minimum-latency-online-video-stabilization/ ; impl https://github.com/sudheerachary/Mesh-Flow-Video-Stabilization
16. COLMAP / PyCOLMAP — https://colmap.github.io/ , https://colmap.github.io/pycolmap/
17. LightGlue (ICCV 2023) — https://github.com/cvg/LightGlue , https://arxiv.org/abs/2306.13643
18. Teed et al., DPVO (NeurIPS 2023) — https://arxiv.org/abs/2208.04726 , https://github.com/princeton-vl/DPVO
19. Lipson et al. 2024, Deep Patch Visual SLAM — https://arxiv.org/abs/2408.01654
20. VGGT (CVPR 2025) — https://vgg-t.github.io/ , https://arxiv.org/pdf/2503.11651
21. Evaluation of DUSt3R/MASt3R/VGGT on aerial photogrammetric blocks (UseGeo) — https://arxiv.org/abs/2507.14798
22. Building Aerial Mosaics for visual MTI (keyframe BA + loop closure for aerial homography mosaics) — https://www.researchgate.net/publication/254951372_Building_aerial_mosaics_for_visual_MTI
23. McLauchlan & Jaenicke 2000, Image mosaicing using sequential bundle adjustment (BMVC) — https://bmva-archive.org.uk/bmvc/2000/papers/p62.pdf
24. Baker et al. 2010, Removing Rolling Shutter Wobble (CVPR) — https://www.microsoft.com/en-us/research/wp-content/uploads/2010/06/0198.pdf
25. Meilland et al., Unified rolling shutter and motion blur model for 3D visual registration — https://hal.science/hal-01357360/document
26. Qu et al. 2023, Fast Rolling Shutter Correction in the Wild (TPAMI) — https://hal.science/hal-04280534v1/document
27. Seiskari et al. 2024, Gaussian Splatting on the Move: blur + RS compensation (ECCV) — https://arxiv.org/abs/2403.13327
28. Sayed & Brostow 2021, Improved handling of motion blur in online object detection (CVPR) — https://www.computer.org/csdl/proceedings-article/cvpr/2021/450900b706/1yeMmuCia9q
29. Guo et al. 2019, Effects of blur and deblurring on visual object tracking — https://arxiv.org/abs/1908.07904
30. Peretroukhin et al., Fast motion deblurring for feature detection and matching — https://arxiv.org/pdf/1805.08542
31. Hierarchical UAV visual geo-localization vs satellite imagery (retrieval + matching + PnP) — https://www.sciencedirect.com/science/article/pii/S0957417424029312
32. GNSS-denied UAV geolocalization by visual map matching — https://arxiv.org/pdf/2103.14381

Raw per-source extraction notes: `research/notes/` in this repo (highD, CitySim, Fonod 2025).
