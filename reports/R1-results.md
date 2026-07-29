# R1 — direct-to-reference registration: results

Branch `regv2`, testbed = EP-03 hero clip (4K, 426 frames @ 25 fps,
elevated carriageway, camera sweeps oblique -> near-nadir with ~90 deg of
rotation over the clip). Baseline = the shipped v1 artifacts (chained
LK homographies + plate mosaic refinement + landmark anchors).

All numbers from `scripts/r1_metrics.py` (full history in
`testdata/hero/qa/r1-metrics.jsonl`). World px scales differ slightly
between the two coordinate frames (v1 = frame 0 plane, R1 = frame 140
plane, Jacobian ratio ~1.2), so cross-column absolute M1 comparisons are
approximate; ratios and the frame-px M3 are exact.

## Metrics

| metric | v1 baseline | R1 (register) | R1 + anchors | target |
|---|---|---|---|---|
| M1 mid-clip (f100-350) med / p90, world px | 0.88 / 4.64 | 2.57 / 12.16 | 1.75 / 7.83 | — |
| M1 last 2 s (f376-425) med / p90 | 4.02 / 30.74 | 1.99 / 10.36 | 0.93 / 2.67 | — |
| **M1 last2s / mid (median)** | 4.58 | **0.77** | 0.53 | <= 2 |
| **M2 Jacobian max/median @ jam centroid** | 1.42 | **1.51** | 1.48 | < 2 |
| **M3 dot offset f300 / 380 / 415 / 424, frame px** | 1.5 / 14.3 / 69.1 / 64.9 | **4.8 / 1.3 / 2.3 / 1.0** | 3.9 / 1.6 / 2.0 / 1.2 | <= 5 all |
| plate deck-patch sharpness (Laplacian var) | 400 | 345 | — | >= v1 |

Success criteria:
- **M1 last-2s <= 2x mid: PASS** (0.77; v1 fails at 4.58).
- **M2 max/median < 2: PASS** (1.51). Caveat: the "v1 blows up 10-14x"
  figure does not reproduce on the saved baseline — even v1's raw chain
  (`homographies-raw.json`) reads only 1.54 at the jam centroid. v1's
  end-of-clip failure lives in M3 (69 px), not in this Jacobian probe.
- **M3 <= 5 px at all four probes: PASS** (worst 4.8 px at f300; v1 fails
  three of four, worst 69 px).
- **Plate at least as sharp as v1: NEAR-PARITY, not clearly met.** On the
  jam deck the R1 plate is visually comparable but scores ~14% lower
  Laplacian variance on a free-lane patch (345 vs 400). Off-deck ground
  is *intentionally* smeared (registration plane is the deck; ground
  parallax is pushed off-plane instead of contaminating measurements).
  v1's number also benefits from its plate-refine stage, which directly
  optimizes plate alignment.

Note on v1's low mid-clip M1: v1's `anchors` pass fits per-frame affine
corrections that minimize exactly the M1 landmark residual, so its
mid-clip M1 is low partly by construction. Running the same (unchanged)
anchors pass on top of R1 gives the best numbers everywhere
(2.50 -> 1.85 px global median) — it is now a small polish rather than a
structural rescue.

## What the experiment actually found

1. **`stabilo` works and its defaults are right** (ORB/rsift, SNN 0.9,
   MAGSAC++, 0.5x downscale, 15% bbox-dilation masks). Installed with
   `--no-deps` (its opencv pin would have downgraded cv2 5.0 -> 4.14;
   torch/torchvision/fsspec untouched, verified). It is used for the
   keyframe chain edges and frame->keyframe edges.
2. **Pure direct-to-reference matching is impossible on this clip.** The
   camera sweep (oblique -> nadir, ~90 deg roll) kills descriptor
   matching beyond ~4 s of baseline: ORB and SIFT both collapse to <10
   inliers vs the reference. Geo-trax-style "match every frame to the
   master frame" assumes a hovering drone; this clip is the harder case
   the research flagged. The fix that worked: **chain-initialize, then
   refine in reference coordinates** — warp each keyframe by its chain
   estimate, then fit only the residual against the reference (gated
   SIFT + coarse-to-fine masked ECC). Chain drift is measured and
   removed (16-215 px worth) instead of accumulated.
3. **The deck is a hall of mirrors.** Deck-only features are periodic
   (lane dashes, barrier posts, parallel lanes), so both descriptor
   matching (without a spatial gate) and ECC (from a wrong basin) happily
   lock one period off — with healthy-looking inlier counts / rho (a
   0.71-rho lock was 59 px off; the rho landscape over a +-120 px offset
   grid is a plateau of aliases). Appearance metrics cannot arbitrate.
4. **The jam itself is the best validator.** Final design: every keyframe
   gets two candidate locks — direct-to-ref and neighbor-template (2 s
   appearance gap) — and when they disagree on the deck by > 12 px, the
   co-tracked vehicles between adjacent keyframes vote (lower-half median
   displacement of shared detection centers; stopped cars score ~1 px
   under the true lock, ~the alias offset under a false one). This is
   v1's static-vehicle idea inverted a second time: vehicles are excluded
   from *estimation* but used for *verification*.
5. **Off-plane leak grows with baseline, so keyframe density must follow
   camera motion.** Fixed 2 s keyframes left 20-frame baselines during
   the fast end move -> up to 105 px mid-segment error. Two fixes:
   motion-adaptive keyframe windows (phase-correlation shift budget,
   13 keys instead of 9) and a per-frame LK polish in reference
   coordinates against the keyframe template (local flow cannot alias).
6. **Deck polygon mask (deck.json) is load-bearing.** Without it the
   refinement locks onto the ground plane and per-keyframe deck bias is
   10-30 px (M1 mid ~13 px). One coarse 7-vertex polygon drawn on the
   reference frame serves every keyframe, because all refinement happens
   in reference coordinates.

## Pipeline notes / breakage

- `python3 -m tva world` runs unchanged on the new `homographies.json`
  (same schema + `"method": "ref-frame"` and provenance keys).
- `python3 -m tva plate` **must not run its mosaic drift refinement** on
  ref-frame homographies: it re-anchors on off-deck content and "corrects"
  the deck registration by ~400 px (median). Built with
  `plate.build(ws, refine=False)`; the CLI still defaults to refine=True
  for v1 compatibility — flip the default when regv2 merges.
- `anchors` (unchanged) is compatible and gives a small improvement.
- Runtime: ~2.5 min for the full register pass on 426 4K frames (CPU).

## Remaining risks

- The stopped-car vote needs a jam. On free-flow clips with a moving
  camera the vote returns None (fallback: neighbor lock wins by rho),
  so far-from-reference geometry would rest on the neighbor chain alone.
- Plate deck sharpness is ~14% below v1 by Laplacian; if the plate is
  used as a visual asset (not just a road layer), a deck-masked
  plate-refine (align samples on deck features only) would close the gap.
- deck.json is hand-drawn per site; the annotator flow doesn't exist yet.
- M1 p90 mid-clip (12 px raw, 8 px anchored) is dominated by creeping
  "stopped" landmarks (the metric's landmark set includes sub-threshold
  creepers whose median world position is a moving target), so it
  overstates registration error — but that contamination applies equally
  to the v1 column.
- Single clip. The arbitration thresholds (DELTA_AGREE_PX=12,
  CAR_VOTE_MIN=12, ECC_MIN_RHO=0.30) have not been exercised elsewhere.

## Artifacts

- `tva/register.py` — the new module; `python3 -m tva register --work ...`
- `scripts/r1_metrics.py` — M1-M3 harness (appends qa/r1-metrics.jsonl)
- `testdata/hero/` (gitignored): `homographies-v1.json` (baseline),
  `homographies-refframe.json` (R1 snapshot), `deck.json`, plates,
  `qa/stab-blend.jpg`, `qa/plate-*-deckcrop.jpg`
