# EP-03 phantom-jam hero clip

First real run of the pipeline. Source: `envato-hero-highway-jam-aerial-4k.mov`
(3840×2160, 25 fps, 426 frames, 17.04 s; drone pans + rotates ~70°).
Work dir lives in the episode repo so artifacts version with the episode:
`math-vs-vibes/promo/ep-03/phantom-jam-short/work/hero-wave/tva/`.

```bash
WORK=~/grove/math-vs-vibes/promo/ep-03/phantom-jam-short/work/hero-wave/tva
python3 -m tva init ~/grove/math-vs-vibes/promo/ep-03/phantom-jam-short/footage/envato-hero-highway-jam-aerial-4k.mov --work $WORK
python3 -m tva stabilize --work $WORK
python3 -m tva detect --work $WORK --model weights/visdrone-yolov8x.pt --imgsz 1920
python3 -m tva world --work $WORK
python3 -m tva spacetime --work $WORK
python3 -m tva worldmap --work $WORK
python3 -m tva render --work $WORK          # -> $WORK/overlay-speed.mp4
```

Numbers from the VisDrone run: 1541 tracks / 94,486 observations
(~170–300 vehicles per frame), 726 world tracks, 154 stop-onset events.
COCO yolo11m managed only a third of that and lost the clip's final 5 s
entirely (near-vertical view).

Notes:

- Stabilization: ~1300 RANSAC inliers/frame; `qa/stab-blend.jpg` shows the
  ground plane registering sharply. Elevated ramps double slightly (scene is
  an interchange, not one plane) — fine as long as the jammed carriageway
  itself is clean.
- The clip's existing hand-keyed pipeline (band + dim mask) is one level up
  in `work/hero-wave/`; the goal is for measured stop events + wave fit from
  here to replace the eyeballed `WAVE_KF` there.
