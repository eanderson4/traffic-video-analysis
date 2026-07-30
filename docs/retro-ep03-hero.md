# Retro: EP-03 phantom-jam hero clip (2026-07-29)

First real production run of the v2 pipeline. ~3 hours from raw drone clip
to the finished focus-lane overlay (426 frames, 4K, dense jam). Target for
the next clip of this shape: **1.5 hours**. This doc records where the time
actually went and what library work buys it back.

## What we ran (the recipe)

```bash
python -m tva init <video> --work W
python -m tva stabilize --work W
python -m tva detect --work W --model weights/visdrone-yolov8x.pt \
    --imgsz 3840 --max-det 900        # yolo11m@1920 missed most of the jam
python -m tva world --work W
python -m tva roads --work W
# hand-edit statics.json (parked lots / depots: registration drift reads
# as fake speed there, no automatic test catches it)
python -m tva render --work W
# focus lane: seed focus_lane.json (road + offset_band guess), then
python -m tva edit --work W --port 8123   # human pass: click loud cars,
                                          # shift-click manual keyframes
python -m tva render --work W             # re-render after each save
```

The render now does recovery automatically for focus lanes: backwards
fragment stitching, held-position backfill for cars already stopped at
first detection, manual-car forward linking, corridor fitted+sized from
the loud dots. None of that needed hand-tuning once written; it needed
*writing* today.

## Where the 3 hours went

| phase | ~time | notes |
|---|---|---|
| init → world (mechanical) | 25 min | mostly waiting; one detect re-run after the yolo11m@1920 recall failure |
| statics zones by hand | 20 min | parked lots found by eyeballing QA renders, polygons typed into JSON |
| overlay look tuning | 45 min | every tweak = full ~2 min render + QA jpg inspection; many loops (gap bridge, trail gates, corridor width ×3, dim level) |
| focus-lane selection | 45 min | automated band guess picked partly the wrong lane; built the editor mid-session; human click pass is fast (~20 min) and *better* than the heuristic |
| early-coverage recovery | 40 min | oblique early frames fragment/miss tracks; wrote stitch + backfill + manual linking to compensate |
| misc friction | 15 min | editor session lost once (no autosave), sparse-keyframe scrub stalls until the frame cache was added |

Two structural lessons:

1. **Detection quality upstream is the whole game.** Nearly half the total
   time (recovery code + manual keyframes + the accuracy pass) compensated
   for misses in the oblique early frames. Fixing recall there shrinks
   three phases at once.
2. **The human-in-the-loop editor beat the heuristic.** The offset-band
   auto-pick was wrong enough that Eric re-picked ~40% of the lane by hand
   — but the click pass itself was quick and definitive. Lean into fast
   human curation with good tooling rather than smarter auto-picking.

## Speedup backlog (ranked by expected minutes saved)

1. **Single-frame style preview** — `tva render --frame N` (or a preview
   button in the editor) that runs the full real styling pipeline on one
   frame in ~2s. Kills the 2-min-render look-tuning loop. (~30-40 min)
2. **Detect on stabilized frames / tiled oblique frames** — measure recall
   on the first 100 frames; the early oblique view is where tracks
   fragment. Even a second detect pass warped to the reference plane for
   low-confidence regions would help. (~20-30 min of downstream recovery)
3. **Promote stitching to a core tracking post-pass** — the focus-lane
   stitcher (backwards fragment chaining with lane + kinematic continuity)
   generalizes; run it for all tracks at `world` time so every stage sees
   clean ids. (~15 min + better everything)
4. **Editor autosave + undo** — journal edits to localStorage, restore on
   load; one lost session cost ~15 min and trust. (~15 min, worst case)
5. **Statics auto-proposal** — cluster static off-road tracks into
   candidate polygons, show them in the editor for one-click accept,
   instead of typing JSON from QA screenshots. (~15 min)
6. **Lane pick in the editor** — click one car → propose the band from the
   offset histogram; write offset_band for you. Removes the JSON
   guess-and-check seed step. (~10 min)
7. **Render turbo mode** — `-preset veryfast` (or NVENC) for iteration
   renders, `medium` only for final. (~5 min across the session)
8. **`no_backfill` toggle in the editor** — currently a hand-edit;
   right-click a backfilled dot to kill its synthetic history. (~5 min)

## Ideas that earned their keep (keep building on these)

- Per-clip hand-editable JSON specs (`focus_lane.json`, `statics.json`,
  `manual_tracks.json`) — versionable, diffable, survive editor rebuilds;
  the editor writes only its keys so hand-added keys persist.
- Editor mirrors the render's recovery pass exactly (same code path), so
  the human validates what will actually draw — parity killed a whole
  class of "why does the render differ" confusion.
- Manual cars as *links* to future tracks, not standalone annotations:
  one click at frame 0 inherits the real track's motion and speed.
- Markers glued to detections (never smoothed estimates), corridor fitted
  from the curated dots — human curation feeds geometry back in.
