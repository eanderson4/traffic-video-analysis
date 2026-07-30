# TVA v2: improvement notes (running)

Accumulated during EP-03+ production sessions, for review after ~5-10 hero
clips. Process/tooling candidates only — not committed decisions. See
docs/retro-ep03-hero.md for the ranked backlog from the first clip.

## From the EP-03 red-wave polish session (2026-07-29/30)

- **3-state quantization (rolling/braking/stopped) beat the continuous
  ramp for storytelling.** Hysteresis + debounce thresholds derived from
  v_stop/v_move worked on the first try, no calibration needed. Validate
  on the next 2-3 clips before calling it settled.
- **Creep vs restart separation is ONE data point.** Queue creep topped
  out ~1.6x v_move, genuine restarts sustained >2.6x v_move, so stopped
  exits straight to green at 2.2x v_move / 0.5s. Check this gap holds on
  other clips (different altitudes, jam densities, registration quality).
- **No special state rules (Eric, emphatic).** Sticky-stop caused
  stuck-red on genuinely rolling cars; red→green-only exit skipped amber
  on roll-off. Both reverted (a7b3bec) in favor of the plain symmetric
  machine: hysteresis + debounce, amber both directions. Suppressing
  noise with asymmetric physics keeps producing worse artifacts than the
  noise. If creep shimmer bothers a future clip, fix the measurement
  (registration), not the renderer.
- **`--monotonic-wavefront` is storytelling sugar, default off.** Ordered
  flips read great but re-time real stops by up to ~1s. Fine for hero
  clips, wrong for analysis. Keep the honest default; revisit if every
  hero clip ends up wanting it on.
- **waveqa/wavevid are cheap and high-value.** The 1D space-time diagram
  diagnosed in one glance what took three render loops to suspect (noise
  was mid-queue drift, not flip-time scatter). Candidate: make both
  standard per-clip outputs, like the render QA jpgs.
- **Stop-onset rings now ride the state machine, not kinematics
  stop_events.** The stop_events path (computed at `world` time, keyed to
  pre-stitch ids) silently drops manual/stitched cars. If rings are ever
  wanted for quiet cars too, unify the event source instead of keeping
  two clocks.
- **Editor parity gap (new).** The editor mirrors the recovery pass but
  not the state machine — it can't show quantized colors, flips, or the
  wavefront. A "states" preview in the editor would let the human
  validate the wave look before a render. Relates to retro backlog #1
  (single-frame preview); maybe they're the same feature.
- **Render loop cost is still the tax.** ~40s full render per tweak.
  The retro's turbo-mode and single-frame items remain the top savings.
- **Per-clip JSON specs keep winning** (focus_lane/manual_tracks/statics
  all untouched this session while styling changed 4 times).

## Open questions for later clips

- Does the wavefront ever genuinely release mid-clip (queue drains)?
  Current exit rule handles it, but the pop/ring semantics assume
  stop-once. Double pop on re-stop is untested visually.
- Corridor extremes (min/max, no percentile) assume a clean curated lane
  — one wrongly-included track blows the corridor wide. Fine while lanes
  are hand-picked; needs a guard if lane picking is ever automated.
