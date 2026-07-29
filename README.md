# traffic-video-analysis (tva)

Reusable library for analyzing traffic footage: camera stabilization, vehicle
detection/tracking, road-frame kinematics, and overlay/diagram rendering.
Built for Math vs Vibes traffic videos (first user: EP-03 phantom-jam hero
clip), designed to grow across episodes.

## Core idea: world plane first

Aerial/elevated traffic footage of a (locally) planar road means camera
motion is exactly a per-frame **homography**. So the pipeline is:

1. **Stabilize** — estimate per-frame homographies to a reference plane
   (frame 0's ground plane) from tracked feature points; RANSAC rejects
   moving vehicles, and in a jam most cars are static anchors anyway.
2. **Detect + track** — YOLO + ByteTrack on the native-resolution frames →
   per-frame boxes with persistent ids.
3. **World kinematics** — map every detection through its frame's homography
   into the reference plane. Camera pan/rotation cancels: a stopped car is a
   fixed world point. Speeds come from a constant-acceleration Kalman
   filter + RTS smoother per track (velocity is a state, never a finite
   difference), with per-observation measurement noise scaled by the local
   homography Jacobian (projective amplification) and re-weighted from each
   segment's residuals (motion blur). Moving/stopped hysteresis →
   stop-onset events with sub-frame timing (the jam wave front, measured).
4. **Road model** — centerline in the reference plane (derived from vehicle
   trajectories, refinable by hand annotation) gives a station coordinate
   `s` = arc length along the road. Car state becomes 1-D: `s(t)`.
5. **Render** — QA sheets, per-car overlay markers (speed-colored),
   space–time diagrams where shockwaves appear as diagonal boundaries.

Manual annotation is a *refinement* layer on top of the measured geometry,
not the source of it.

## Artifacts

Each analysis lives in a work directory of plain JSON + PNG artifacts so
every stage is inspectable and cacheable:

| file | producer | contents |
|------|----------|----------|
| `meta.json` | `init` | source path, width/height, fps, frame count |
| `homographies.json` | `stabilize` | per-frame 3×3 `H` (frame px → reference plane), inlier stats |
| `tracks.json` | `detect` | per track: class, per-frame box center/size/conf |
| `world_tracks.json` | `world` | per track: reference-plane positions, smoothed speed, state runs, stop events |
| `centerline.json` | `world` | road spine polyline in the reference plane |
| `qa/` | all | visual checks per stage |

## Usage

```bash
python -m tva init <video> --work <dir>
python -m tva stabilize --work <dir>            # cv2 only
python -m tva detect --work <dir> --model weights/visdrone-yolov8x.pt --imgsz 1920
python -m tva world --work <dir>
python -m tva spacetime --work <dir>            # space-time diagram PNG
python -m tva speedqa --work <dir>              # raw vs smoothed speed profiles
python -m tva render --work <dir>               # speed-colored overlay video
python -m tva render --work <dir> --roads --highlight 33,49   # optional layers
```

`render` draws cars only by default (`--roads` adds the inferred road/lane
overlay; `--highlight` enlarges + outlines specific track ids for
storytelling). `anchors` re-registers each frame against provably-static
vehicles near the analysis centerline — same-plane landmarks, because an
elevated carriageway parallax-shifts against the ground plane whenever the
camera translates.

**Model choice matters.** COCO-trained YOLO (yolo11m etc.) is near-blind to
nadir/top-down aerial vehicles (4 detections on a frame where VisDrone
weights find ~170; tiling does not help — it's a domain gap, not a scale
problem). For drone footage use VisDrone-trained weights:
`weights/visdrone-yolov8x.pt`, from
https://huggingface.co/mshamrai/yolov8x-visdrone (not committed; re-download
if missing). `detect` picks vehicle classes by name so COCO and VisDrone
models both work unmodified.

See `examples/hero-wave.md` for the EP-03 phantom-jam run.

## Roadmap

- Browser annotator (port from ep-03 hero-wave) reading/writing these
  schemas: verify/nudge stop events, refine centerline, mark lanes.
- Wave-front fitting through stop-onset events → wave speed (km/h via
  lane-width or car-length pixel scale).
- Anti-drift: long-range homography refinement against the reference frame
  (current v1 chains adjacent frames; QA = stopped-car world jitter).
- Detection masks feeding back into stabilization feature selection.
- Overlay renderers that compose into episode ffmpeg pipelines.
