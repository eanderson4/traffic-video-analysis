# How to make a traffic-analysis short (playbook)

Compiled 2026-08-06 from the EP-03 phantom-jam hero clip and the queue-short
intersection clip. Two kinds of shorts so far:

1. **Wave short** (hero): one focus lane, dots on cars, stop-wave marching
   backwards. Tooling: focus.py / waveqa / wavevid / render / edit.
2. **Intersection short** (queue): 4-way stoplight, rotated bboxes, per-approach
   queue counters + wait-time tallies, optional equation rain.
   Tooling: stabilize → register → detect → world → queue → xingvid.

Everything runs from the repo root (`~/grove/traffic-video-analysis`) as
`python3 -m tva <stage> --work <dir>`. Test footage is gitignored
(`testdata/`); analysis artifacts (tracks.json, queue.json, …) live next to
the video in the work dir.

---

## 1. Intersection short pipeline (the xing2/xing3 recipe)

Stages, in order. Each reads/writes the work dir.

```bash
python3 -m tva init <video.mp4> --work testdata/<name>   # probe, meta.json, copies video
python3 -m tva stabilize --work testdata/<name>          # per-frame homographies.json (jitter removal)
python3 -m tva register  --work testdata/<name>          # + world registration QA
python3 -m tva detect    --work testdata/<name>          # visdrone-yolov8x → tracks.json
python3 -m tva world     --work testdata/<name>          # world_tracks.json (v_stop/v_move thresholds)
python3 -m tva queue     --work testdata/<name>          # queue.json arrivals/waits per approach
python3 -m tva xingvid   --work testdata/<name>          # xing-queue.mp4 (annotated video)
python3 -m tva xingvid   --work testdata/<name> --rain   # xing-queue-rain.mp4 (equation-rain variant)
```

### Hand-authored config (the part that takes judgement)

`intersection.json` in the work dir defines approaches. This is THE file to
get right; bad geometry = zero arrivals or ghost queues. Per approach:

```json
{"name": "west", "dir": [1, 0],
 "in_line":   [[x1,y1],[x2,y2]],   // crossed on entry (arrival)
 "out_line":  [[x1,y1],[x2,y2]],   // crossed on exit (departure) = stop bar
 "queue_poly": [[…4 corners…]]}    // where a stopped car counts as queued
```

- `dir` = unit vector of travel in IMAGE coords (y down).
- in_line is up-stream of out_line; queue_poly sits between them.
- Pick coords from a QA frame (`qa/xing-<dir>.png` or a raw frame extract),
  err on the tight side; wide polys catch turning/through traffic.
- Better than eyeballing: after `world`, histogram the track positions per
  leg (`tr["x"]/tr["y"]` in world_tracks.json) and place lines where tracks
  actually flow — this fixed xing3, where the north leg turned out to be
  one-way and the "westbound lane" was parked cars. Also check `tr["runs"]`
  stopped spans and whole-clip static tracks to find parking rows that
  contaminate queue polys.
- Counting lines must sit where tracks actually cross: not at frame edges
  (max track x was 1909 on a 1920-wide frame — in_line at the edge = 0
  arrivals) and not under tree occlusion where tracks begin/end.
- Lessons from xing2: expect false positives on crosswalk paint (square-ish
  boxes) and parked cars — queue_poly placement is the fix, not detection.
- Cars leaving frame mid-wait used to bank false wait time; current logic
  keys waits off the signal change — keep an eye on it in QA.

### Detection pins (do not change without asking)

- model `weights/visdrone-yolov8x.pt`, conf 0.25, max_det 900.
- Hero work used imgsz 3840; intersection clips ran imgsz 1920 (dense
  top-down, small cars — worked well).
- Pedestrian detection: NOT good enough for shorts. We skip peds entirely.

### QA loop (always look, never assume)

```bash
mkdir -p /tmp/xingqa
ffmpeg -y -loglevel error -i <mp4> -vf "select='eq(n\,N)'" -vsync 0 /tmp/xingqa/fN.png
```

Then ReadMediaFile the frame. Check: bboxes rotated along heading (N/S/E/W
queues all correct orientation), no boxes on crosswalks, wait-timer starts at
signal change not at frame-exit, queue counts match eyeballs.

`queue.json` is the ground truth for the script: per approach
`n_arrivals`, `total_wait_veh_s`, `mean_wait_s`, `max_queue`. Use those
numbers verbatim; never extrapolate ("per hour", "all day").

---

## 2. Wave short pipeline (EP-03 hero, for reference)

Same init/stabilize/detect/world stages, then the focus-lane stack:

- `python3 -m tva edit --work <dir>` — browser editor: click cars loud/quiet,
  pin color-state overrides, view the 1D state strip. (Never restart a
  running editor server — it's the user's live session.)
- `python3 -m tva render --work <dir>` — overlay-speed.mp4.
- `python3 -m tva waveqa / wavevid` — 1D space-time wave diagram / strip
  video (the "arrange loud dots on a line" view).
- Loud-dot states: rolling=green, braking=amber, stopped=red, quantized with
  hysteresis+debounce from v_stop/v_move in world_tracks.json. Ring ping on
  the green→stopped flip; re-arms only after the car returns to green.
- `--monotonic-wavefront` flag exists on render/waveqa/wavevid but is OFF
  and known-buggy (order_wave mangles flip lists). Wave reads fine without
  it; fix or delete later.

---

## 3. xingvid overlay vocabulary (xingvid.py)

- Rotated bboxes with heading triangles; color = state (green rolling,
  amber braking, red stopped, yellow threshold tuned HIGH — practically
  stopped before amber).
- Per-approach HUD: current queue count + total wait this red phase.
- Wait-time "flight": when a queued car starts moving, its wait seconds
  animate bigger/bolder and fade — reads as the wait "ending".
- `--rain`: queueing-theory equations fall from the sky one every ~2.6s
  (ease-out, sway, settle 55% opacity in quadrant slots), `L=λW` last and
  center. Constants at top of xingvid.py: `RAIN_START_S`, `RAIN_EVERY_S`,
  `EQUATIONS`, `RAIN_SLOTS`. matplotlib gotcha: `math_to_image(color=)` is
  broken in mpl 3.10 — render black-on-white, use luminance as alpha.
- Equation set used: Poisson P(N=k), exp service, ρ=λ/μ, L=ρ/(1−ρ),
  W=1/(μ−λ), Pollaczek–Khinchine, Kingman, then Little's law.

## 4. Script guardrails (learned the hard way)

- Counts and wait-times only. The drone zoom invalidates speeds — never
  quote them.
- λ (arrival rate) IS measured from queue.json — fine to use.
- No temporal extrapolation. "In this 37.5 s clip" framing.
- No claims about signal-timing intent (we don't know the plan).
- Persona that landed: distressed math person — "I have the numbers, now
  what?" — cascading open questions about why optimization is hard
  (who sets the timing plan, optimized for what, Braess's paradox).
  Land on Little's law as the one solid thing.
- A Dr. Seuss cut (anapestic tetrameter, counting cadence) works as the
  playful alternative; keep numbers real inside the rhyme.
- Scripts live in `~/grove/math-vs-vibes/promo/ep-03/queue-short/script-ideas.md`
  (NOT promo/queue-short — that path confused things).
- Sol = accuracy review (numbers vs queue.json), Fable = narrative review.
  Run both as cold-context subagents with the script + queue.json paths.

## 5. Footage sources

- Hero wave clip: user-supplied drone footage (testdata/hero).
- Intersection clips: xing.mp4 (tree-lined 4-way, 60 s) and xing2.mp4
  (wide 4-way, 37.5 s) — user-supplied; both 1920x1080 ~24 fps top-down.
- SinD (testdata/datasets/sind): track CSVs + OSM maps + signal phases for
  Tianjin/Changchun intersections — NO video, but ground-truth signal
  timing and bbox tracks worth mining for validation.
- pNEUMA (testdata/datasets/pneuma): 15 GB Athens drone trajectories (CSV),
  no video either; backlog item is running the queue analysis at scale on it.
- Both open research datasets. For future shorts with real signal-phase
  truth, SinD's Traffic_Lights.csv can replace our signal-change inference.

## 6. Gotchas / tech debt (short version — see docs/v2-improvements.md)

- testdata/ is gitignored: keep analysis JSONs you care about backed up or
  regenerated (scripts/ has queue helpers).
- frame cache in xingvid is unbounded (fine at 37 s, watch at 5 min).
- trail double-blends under the fill disc (minor bright hot-spot).
- Editor Save overwrites whole state_overrides.json (lost-update race if CLI
  `override` is used while editor open).
- End-of-session retro notes from EP-03: docs/retro-ep03-hero.md.
