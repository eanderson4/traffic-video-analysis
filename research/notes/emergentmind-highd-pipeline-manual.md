# notes: https://www.emergentmind.com/topics/highd (manual extract, fetched 2026-07-29)

Key pipeline facts (all attributed to Krajewski et al. 2018, arXiv:1810.05642):

- Hardware: DJI Phantom 4 Pro Plus, 4K (4096x2160 @ 25 fps), hovering NEXT TO the highway
  (not directly overhead) to minimize perspective distortion. ~10x10 cm pixel size on road.
  Recordings only in sunny, windless conditions, 8AM-5PM. 60 recordings, 6 locations,
  ~420 m of road per recording, 16.5 h, 110,000 vehicles, 45,000 km driven, 5,600 lane changes.
- Stabilization: OpenCV; estimate transformation from each frame's background to the
  background of the FIRST frame (anchor frame, not chained). First frame rotated so lane
  markings are horizontal.
- Detection: U-Net semantic segmentation -> pixel clusters -> bounding boxes.
- Tracking: distance-based matching across frames.
- Smoothing: Rauch-Tung-Striebel (RTS) smoother with constant-acceleration model, refining
  position, speed, acceleration longitudinally and laterally.
- Reported accuracy: ~99% detection rate, ~2% false positives, mean midpoint position error
  < 3 cm (longitudinal and lateral) vs manual labels. Commonly summarized as "typically
  less than 10 cm" positioning error.
- Static infrastructure (lanes, signs, speed limits) annotated manually.
- Deliverables per recording: aerial image, site/infrastructure CSV, track-summary CSV
  (dims, class, direction, mean speed), per-frame trajectory CSV incl. DHW/THW/TTC and
  surrounding-vehicle relations; maneuver labels (free driving, following, critical, lane change).
