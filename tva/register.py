"""Direct-to-reference registration (v2, experiment R1).

Replaces chained stabilization with a two-level graph: every frame
registers to its nearest keyframe (ORB), keyframes register to one
blur-scored mid-clip reference frame. All detected vehicle bboxes
(dilated 15%) are exclusion masks during feature detection — in a jam
most corners sit on possibly-creeping cars, which is poison for the
road-plane homography.

The hero clip's camera swings from oblique to near-nadir with ~90 deg of
rotation, so distant keyframes cannot be matched to the reference in one
shot (ORB and SIFT both collapse to <10 inliers). Instead each keyframe is
*initialized* by chaining strong adjacent-keyframe RootSIFT edges, then
*refined* by warping it into reference coordinates and matching directly
against the reference (MAGSAC++ residual homography). The final geometry
of every keyframe is therefore a direct feature fit to the reference —
the chain only provides the initial warp, so its drift is measured and
removed (typically ~20 px here), never accumulated.

Estimation is stabilo (Geo-trax's stabilization core): BF kNN + SNN ratio
0.9, MAGSAC++ (cv2.USAC_MAGSAC), 0.5x downscale, full-res H. An optional
deck.json ({"polygon": [[x, y], ...]} in reference-frame full-res pixels)
restricts reference-side features to the elevated carriageway so
off-plane (ground) features can't poison the measurement plane.

Output schema matches stabilize.py: homographies.json {"H": [...]} with
H[i] mapping frame-i full-res pixels -> reference-frame pixels, plus
"method": "ref-frame" and provenance. Downstream world/render run
unchanged.
"""
from collections import defaultdict

import cv2
import numpy as np

from stabilo import Stabilizer

BLUR_WIDTH = 960          # downscale width for blur scoring
KEYFRAME_SPACING_S = 2.0  # one keyframe per window of this length
REF_WINDOW = (0.25, 0.75) # reference candidates: this fraction of the clip
MASK_MARGIN = 0.15        # bbox dilation for exclusion masks
MIN_INLIERS = 30          # below this a frame/edge is flagged
REFINE_MIN_INLIERS = 25   # accept a direct-to-ref residual fit above this

COMMON = dict(matcher_name="bf", filter_type="ratio", filter_ratio=0.9,
              transformation_type="projective",
              ransac_method=cv2.USAC_MAGSAC, downsample_ratio=0.5,
              mask_use=True, mask_margin_ratio=MASK_MARGIN)
# keyframe graph edges + direct-to-ref refinement: RootSIFT, more budget
KEY_CFG = dict(detector_name="rsift", max_features=8000,
               ransac_epipolar_threshold=3.0, **COMMON)
# frame -> nearest keyframe (small baseline): ORB is plenty and fast
FRAME_CFG = dict(detector_name="orb", max_features=4000,
                 ransac_epipolar_threshold=2.0, **COMMON)


REFINE_GATE_PX = 300.     # residual match must move less than this (full-res)
REFINE_SCALE = 0.5        # downscale for the residual fit
REFINE_RATIO = 0.8        # Lowe ratio for the residual fit


def exclusion_mask(shape, corners, deck_mask=None, margin=MASK_MARGIN):
    """255 = usable. Zeros dilated vehicle quads; ANDs the deck polygon."""
    m = np.full(shape, 255, np.uint8) if deck_mask is None \
        else deck_mask.copy()
    if corners is not None:
        for quad in corners:
            pts = quad.reshape(4, 2)
            c = pts.mean(axis=0)
            pts = c + (pts - c) * (1 + margin)
            cv2.fillPoly(m, [pts.astype(np.int32)], 0)
    return m


def residual_fit(ref_gray, warp_gray, ref_mask, warp_mask,
                 gate=REFINE_GATE_PX, scale=REFINE_SCALE):
    """Residual homography aligning an (already chain-warped) keyframe to
    the reference: SIFT + BF ratio test + spatial gate + MAGSAC++. The
    spatial gate kills lane-dash aliasing — with deck-only features the
    scene is periodic along the road, and ungated global matching happily
    locks onto a consensus shifted by whole dash periods.

    Returns (H full-res, inliers)."""
    def prep(g, m):
        g = cv2.resize(g, None, fx=scale, fy=scale)
        m = cv2.resize(m, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_NEAREST)
        return g, m
    g1, m1 = prep(ref_gray, ref_mask)
    g2, m2 = prep(warp_gray, warp_mask)
    sift = cv2.SIFT_create(nfeatures=8000)
    k1, d1 = sift.detectAndCompute(g1, m1)
    k2, d2 = sift.detectAndCompute(g2, m2)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None, 0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    src, dst = [], []
    for pair in bf.knnMatch(d2, d1, k=2):
        if len(pair) < 2 or \
                pair[0].distance >= REFINE_RATIO * pair[1].distance:
            continue
        p2 = np.asarray(k2[pair[0].queryIdx].pt)
        p1 = np.asarray(k1[pair[0].trainIdx].pt)
        if np.hypot(*(p2 - p1)) <= gate * scale:
            src.append(p2)
            dst.append(p1)
    if len(src) < 8:
        return None, 0
    H, inl = cv2.findHomography(np.asarray(src), np.asarray(dst),
                                cv2.USAC_MAGSAC, 3.0 * scale)
    if H is None:
        return None, 0
    S = np.diag([scale, scale, 1.0])
    Hf = np.linalg.inv(S) @ H @ S
    return Hf / Hf[2, 2], int(inl.sum())


ECC_SCALES = (0.2, 0.35, 0.5)  # coarse-to-fine; coarse blurs lane dashes away
ECC_MIN_RHO = 0.30        # accept ECC polish above this correlation


def ecc_refine(ref_gray, warp_gray, mask, R0=None):
    """Intensity-based residual polish: findTransformECC (homography),
    coarse-to-fine. At the coarse scale periodic lane dashes blur out and
    the bridge macro-structure locks the alignment, so this cannot alias
    by a dash period the way descriptor matching can. Returns (R, rho)
    with R mapping warped-keyframe coords -> reference coords."""
    W = np.eye(3) if R0 is None else np.linalg.inv(R0)
    rho = -1.0
    for s in ECC_SCALES:
        g1 = cv2.resize(ref_gray, None, fx=s, fy=s)
        g2 = cv2.resize(warp_gray, None, fx=s, fy=s)
        m = cv2.resize(mask, None, fx=s, fy=s,
                       interpolation=cv2.INTER_NEAREST)
        S = np.diag([s, s, 1.0])
        Ws = (S @ W @ np.linalg.inv(S)).astype(np.float32)
        try:
            rho, Ws = cv2.findTransformECC(
                g1, g2, Ws, cv2.MOTION_HOMOGRAPHY,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6),
                m, 5)
        except cv2.error:
            if rho < 0:      # not even the coarse scale converged
                return None, rho
            break            # keep the last scale that did converge
        W = np.linalg.inv(S) @ Ws.astype(np.float64) @ S
    R = np.linalg.inv(W)
    return R / R[2, 2], float(rho)


def make_stabilizer(cfg, deck_mask=None):
    stab = Stabilizer(**cfg)
    if deck_mask is not None:
        orig = stab.create_binary_mask

        def masked(boxes, box_format):
            return cv2.bitwise_and(orig(boxes, box_format), deck_mask)
        stab.create_binary_mask = masked
    return stab


def boxes_per_frame(ws):
    """{frame: (n,4) float32 [cx cy w h]} over ALL detections."""
    per = defaultdict(list)
    for tr in ws.load("tracks.json")["tracks"]:
        for f, cx, cy, w, h, conf in tr["obs"]:
            per[f].append((cx, cy, w, h))
    return {f: np.asarray(v, np.float32) for f, v in per.items()}


def warp_boxes(boxes, H):
    """xywh boxes -> (n,8) 'four' corner format mapped through H."""
    if boxes is None or not len(boxes):
        return None
    out = []
    for cx, cy, w, h in boxes:
        c = np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                      [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]],
                     np.float64)
        p = np.hstack([c, np.ones((4, 1))]) @ np.asarray(H).T
        out.append((p[:, :2] / p[:, 2:3]).ravel())
    return np.asarray(out, np.float32)


def load_deck_mask(ws):
    """Optional include-mask from deck.json polygon (reference-frame px)."""
    try:
        poly = ws.load("deck.json")["polygon"]
    except FileNotFoundError:
        return None
    meta = ws.meta
    mask = np.zeros((meta["height"], meta["width"]), np.uint8)
    cv2.fillPoly(mask, [np.asarray(poly, np.int32)], 255)
    print(f"deck polygon mask: {len(poly)} vertices, "
          f"{(mask > 0).mean():.0%} of frame")
    return mask


MOTION_WINDOW_PX = 25.    # new keyframe window per this much camera motion
                          # (px at BLUR_WIDTH scale; ~4x that at 4K)


def blur_pass(meta, boxes):
    """Per-frame blur score (variance of Laplacian on vehicle-masked,
    downscaled frames) + inter-frame camera shift (phase correlation)."""
    scale = BLUR_WIDTH / meta["width"]
    scores, shifts = [], []
    prev = None
    cap = cv2.VideoCapture(meta["src"])
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h = int(frame.shape[0] * scale)
        g = cv2.cvtColor(cv2.resize(frame, (BLUR_WIDTH, h)),
                         cv2.COLOR_BGR2GRAY)
        mask = np.ones(g.shape, bool)
        for cx, cy, w, bh in boxes.get(i, []):
            m = 1.0 + MASK_MARGIN
            x0 = max(0, int((cx - w * m / 2) * scale))
            x1 = min(g.shape[1], int((cx + w * m / 2) * scale))
            y0 = max(0, int((cy - bh * m / 2) * scale))
            y1 = min(g.shape[0], int((cy + bh * m / 2) * scale))
            mask[y0:y1, x0:x1] = False
        lap = cv2.Laplacian(g, cv2.CV_64F)
        scores.append(float(lap[mask].var()) if mask.any() else 0.0)
        gf = g.astype(np.float32)
        if prev is not None:
            (dx, dy), _ = cv2.phaseCorrelate(prev, gf)
            shifts.append(float(np.hypot(dx, dy)))
        else:
            shifts.append(0.0)
        prev = gf
        i += 1
    cap.release()
    return np.asarray(scores), np.asarray(shifts)


def pick_frames(scores, shifts, fps, n):
    """Reference = sharpest mid-clip frame. Keyframes = sharpest frame per
    window, where a window closes after ~2 s OR after MOTION_WINDOW_PX of
    accumulated camera motion — fast camera moves get dense keyframes so
    frame->keyframe baselines (and off-plane parallax error, which grows
    with baseline) stay small."""
    a, b = int(n * REF_WINDOW[0]), int(n * REF_WINDOW[1])
    ref = a + int(np.argmax(scores[a:b]))
    spacing = max(1, round(KEYFRAME_SPACING_S * fps))
    keys = []
    w0, acc = 0, 0.0
    for i in range(n):
        acc += shifts[i]
        if i - w0 + 1 >= spacing or acc >= MOTION_WINDOW_PX or i == n - 1:
            k = w0 + int(np.argmax(scores[w0:i + 1]))
            keys.append(ref if w0 <= ref <= i else k)
            w0, acc = i + 1, 0.0
    return ref, sorted(set(keys))


def grab_frames(src, idxs):
    cap = cv2.VideoCapture(src)
    out = {}
    for i in sorted(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            out[i] = frame
    cap.release()
    return out


DELTA_AGREE_PX = 12.      # direct + neighbor locks agreeing within this
                          # (median over the deck) -> take the direct lock
CAR_VOTE_MIN = 12         # min shared tracks for the stopped-car vote


def centers_per_frame(ws):
    """{frame: {track_id: (cx, cy)}} raw detection centers."""
    per = {}
    for tr in ws.load("tracks.json")["tracks"]:
        for f, cx, cy, *_ in tr["obs"]:
            per.setdefault(f, {})[tr["id"]] = (cx, cy)
    return per


def deck_samples(deck, shape, stride=120):
    """Sample points (ref coords) inside the deck polygon (or full frame)."""
    h, w = shape
    ys, xs = np.mgrid[stride // 2:h:stride, stride // 2:w:stride]
    pts = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    if deck is not None:
        keep = deck[pts[:, 1].astype(int), pts[:, 0].astype(int)] > 0
        pts = pts[keep]
    return pts


def apply_h_pts(H, pts):
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(H).T
    return p[:, :2] / p[:, 2:3]


def candidate_delta(H_a, H_b, pts_ref):
    """Median displacement (ref px) between two keyframe->ref candidates
    over the deck sample points."""
    q = apply_h_pts(np.linalg.inv(H_b), pts_ref)
    return float(np.median(np.hypot(
        *(apply_h_pts(H_a, q) - apply_h_pts(H_b, q)).T)))


def stopped_car_score(cent_prev, cent_k, H_prev, H_cand):
    """Lower-half-median displacement of co-tracked cars between adjacent
    keyframes under a candidate registration. In a jam most shared cars
    are stopped or barely creeping over 1-2 s, so the correct candidate
    scores near zero; an aliased lock scores near its alias offset."""
    shared = set(cent_prev) & set(cent_k)
    if len(shared) < CAR_VOTE_MIN:
        return None
    a = apply_h_pts(H_prev, np.array([cent_prev[t] for t in shared]))
    b = apply_h_pts(H_cand, np.array([cent_k[t] for t in shared]))
    d = np.sort(np.hypot(*(a - b).T))
    return float(np.median(d[:max(CAR_VOTE_MIN // 2, len(d) // 2)]))


ECC_GOOD_RHO = 0.55       # (legacy) direct-lock acceptance short-circuit

LK_SCALE = 0.5            # per-frame LK polish runs at this scale
LK_MIN_PTS = 30           # fewer tracked deck points than this -> no polish


def lk_residual(tmpl_gray, warp_gray, tmpl_mask):
    """Per-frame residual homography in reference coordinates: Shi-Tomasi
    corners on the (deck-masked) keyframe template, pyramidal LK into the
    chain-warped frame, MAGSAC++. LK is local, so unlike descriptor
    matching it cannot alias along the periodic lane dashes. Inputs are at
    LK_SCALE; the returned H maps full-res warped coords -> template."""
    p0 = cv2.goodFeaturesToTrack(tmpl_gray, 800, 0.01, 12, mask=tmpl_mask)
    if p0 is None or len(p0) < LK_MIN_PTS:
        return None, 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        tmpl_gray, warp_gray, p0, None, winSize=(31, 31), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    good = st.ravel() == 1
    if good.sum() < LK_MIN_PTS:
        return None, 0
    H, inl = cv2.findHomography(p1[good], p0[good], cv2.USAC_MAGSAC,
                                2.0 * LK_SCALE)
    if H is None:
        return None, 0
    S = np.diag([LK_SCALE, LK_SCALE, 1.0])
    Hf = np.linalg.inv(S) @ H @ S
    return Hf / Hf[2, 2], int(inl.sum())


def register_keyframes(imgs, boxes, keys, ref, deck, cent):
    """H_key[k]: keyframe-k px -> reference px. Chain-initialized outward
    from the reference, then each keyframe refined in reference
    coordinates: SIFT residual fit + ECC polish against the reference
    frame. Where the viewpoint gap to the reference is too big for a
    direct photometric lock (rho < ECC_GOOD_RHO), the template falls back
    to the previous *refined* keyframe's warp — a 2 s appearance gap
    instead of 10 s — still deck-masked, so plane consistency is kept and
    chain drift only enters through these rare fallback hops."""
    h, w = imgs[ref].shape[:2]
    ref_gray = cv2.cvtColor(imgs[ref], cv2.COLOR_BGR2GRAY)
    ref_mask = exclusion_mask(
        (h, w), warp_boxes(boxes.get(ref), np.eye(3)), deck)
    pts_ref = deck_samples(deck, (h, w))

    H_key = {ref: np.eye(3)}
    edge_inl = {ref: -1}
    refine_inl = {ref: -1}
    below = [k for k in sorted(keys) if k < ref][::-1]
    above = [k for k in sorted(keys) if k > ref]
    for side in (below, above):
        prev = ref
        prev_gray, prev_mask = ref_gray, ref_mask
        for k in side:
            edge = make_stabilizer(KEY_CFG)
            edge.set_ref_frame(imgs[prev], boxes.get(prev))
            edge.stabilize(imgs[k], boxes.get(k))
            He = edge.get_cur_trans_matrix()
            edge_inl[k] = edge.get_cur_inliers_count() or 0
            H_init = H_key[prev] @ (np.eye(3) if He is None
                                    else np.asarray(He))

            def warp_into_ref(H):
                g = cv2.cvtColor(cv2.warpPerspective(imgs[k], H, (w, h)),
                                 cv2.COLOR_BGR2GRAY)
                m = exclusion_mask((h, w), warp_boxes(boxes.get(k), H), deck)
                cov = cv2.warpPerspective(np.full((h, w), 255, np.uint8),
                                          H, (w, h))
                return g, cv2.bitwise_and(m, cov)

            warp_gray, warp_mask = warp_into_ref(H_init)
            # candidate A: direct-to-reference (feature fit + ECC polish)
            R1, rinl = residual_fit(ref_gray, warp_gray,
                                    ref_mask, warp_mask)
            feat_ok = R1 is not None and rinl >= REFINE_MIN_INLIERS
            Rd, rho_d = ecc_refine(
                ref_gray, warp_gray, cv2.bitwise_and(ref_mask, warp_mask),
                R0=R1 if feat_ok else None)
            # candidate B: neighbor-template ECC (short appearance gap,
            # reliable but drifts by one hop's error)
            Rn, rho_n = ecc_refine(
                prev_gray, warp_gray, cv2.bitwise_and(prev_mask, warp_mask))
            refine_inl[k] = rinl
            H_d = Rd @ H_init if Rd is not None and rho_d >= ECC_MIN_RHO \
                else None
            H_n = Rn @ H_init if Rn is not None and rho_n >= ECC_MIN_RHO \
                else None
            if H_d is not None and H_n is not None:
                delta = candidate_delta(H_d, H_n, pts_ref)
                if delta <= DELTA_AGREE_PX:
                    H_key[k] = H_d
                    how = (f"ref ecc rho {rho_d:.2f} "
                           f"(nbr agrees, {delta:.1f}px)")
                else:
                    # locks disagree: adjacent-keyframe stopped cars vote
                    s_d = stopped_car_score(cent.get(prev, {}),
                                            cent.get(k, {}),
                                            H_key[prev], H_d)
                    s_n = stopped_car_score(cent.get(prev, {}),
                                            cent.get(k, {}),
                                            H_key[prev], H_n)
                    if s_d is not None and (s_n is None or s_d <= s_n):
                        H_key[k] = H_d
                        how = (f"ref ecc rho {rho_d:.2f} by car vote "
                               f"({s_d:.1f} vs nbr {s_n:.1f}px, "
                               f"delta {delta:.0f}px)")
                    else:
                        H_key[k] = H_n
                        how = (f"nbr ecc rho {rho_n:.2f} by car vote "
                               f"({s_n:.1f} vs ref {s_d:.1f}px, "
                               f"delta {delta:.0f}px)")
            elif H_d is not None:
                H_key[k] = H_d
                how = f"ref ecc rho {rho_d:.2f} (no nbr lock)"
            elif H_n is not None:
                H_key[k] = H_n
                how = f"nbr ecc rho {rho_n:.2f} (ref rho {rho_d:.2f})"
            elif feat_ok:
                H_key[k] = R1 @ H_init
                how = f"feat {rinl} inl only"
            else:
                H_key[k] = H_init
                how = f"CHAIN ONLY (feat {rinl}, rho {rho_d:.2f})"
            print(f"  key {k}: edge {prev} ({edge_inl[k]} inl) -> {how}")
            prev = k
            prev_gray, prev_mask = warp_into_ref(H_key[k])
    return H_key, edge_inl, refine_inl


def run(ws):
    meta = ws.meta
    n, fps = meta["n_frames"], meta["fps"]
    boxes = boxes_per_frame(ws)
    deck = load_deck_mask(ws)

    scores, shifts = blur_pass(meta, boxes)
    n = min(n, len(scores))
    ref, keys = pick_frames(scores, shifts, fps, n)
    print(f"reference frame {ref} (blur {scores[ref]:.0f}), "
          f"{len(keys)} keyframes: {keys}")

    imgs = grab_frames(meta["src"], set(keys) | {ref})
    H_key, edge_inl, refine_inl = register_keyframes(
        imgs, boxes, keys, ref, deck, centers_per_frame(ws))

    # per-keyframe machinery for the frame->keyframe edges:
    # - an ORB stabilizer in the keyframe's own coords (initializer); its
    #   deck mask is the reference polygon projected into the keyframe,
    #   dilated so nearby frames' deck stays inside despite camera motion
    # - an LK polish template: the keyframe warped into reference coords
    #   at LK_SCALE, deck-masked with the ONE reference polygon (no
    #   dilation needed there — everything is in reference coordinates)
    w4, h4 = meta["width"], meta["height"]
    S = np.diag([LK_SCALE, LK_SCALE, 1.0])
    wl, hl = int(w4 * LK_SCALE), int(h4 * LK_SCALE)
    deck_l = None if deck is None else cv2.resize(
        deck, (wl, hl), interpolation=cv2.INTER_NEAREST)
    kstabs, tmpls = {}, {}
    for k in keys:
        deck_k = None
        if deck is not None:
            deck_k = cv2.warpPerspective(deck, np.linalg.inv(H_key[k]),
                                         (w4, h4))
            deck_k = cv2.dilate(deck_k, np.ones((151, 151), np.uint8))
        kstabs[k] = make_stabilizer(FRAME_CFG, deck_k)
        kstabs[k].set_ref_frame(imgs[k], boxes.get(k))
        M = S @ H_key[k]
        g = cv2.warpPerspective(
            cv2.cvtColor(imgs[k], cv2.COLOR_BGR2GRAY), M, (wl, hl))
        m = exclusion_mask((hl, wl),
                           warp_boxes(boxes.get(k), M), deck_l)
        cov = cv2.warpPerspective(np.full((h4, w4), 255, np.uint8),
                                  M, (wl, hl))
        tmpls[k] = (g, cv2.bitwise_and(m, cov))
    del imgs

    key_arr = np.asarray(keys)
    Hs, inliers, lk_inliers, flagged = [], [], [], []
    cap = cv2.VideoCapture(meta["src"])
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        k = int(key_arr[np.argmin(np.abs(key_arr - i))])
        if i == k:
            H, inl, lk_inl = H_key[k], -1, -1
        else:
            kstabs[k].stabilize(frame, boxes.get(i))
            Hk = kstabs[k].get_cur_trans_matrix()
            inl = kstabs[k].get_cur_inliers_count() or 0
            H = H_key[k] @ (np.eye(3) if Hk is None else np.asarray(Hk))
            # LK polish in reference coordinates against the keyframe
            # template — removes the off-plane error the ORB edge leaks
            # as the frame->keyframe baseline grows
            tg, tm = tmpls[k]
            M = S @ H
            wg = cv2.warpPerspective(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), M, (wl, hl))
            fm = exclusion_mask((hl, wl), warp_boxes(boxes.get(i), M),
                                None)
            R, lk_inl = lk_residual(tg, wg, cv2.bitwise_and(tm, fm))
            if R is not None:
                H = R @ H
        if 0 <= inl < MIN_INLIERS:
            flagged.append(i)
        Hs.append(H / H[2, 2])
        inliers.append(int(inl))
        lk_inliers.append(int(lk_inl))
        if i % 50 == 0:
            print(f"frame {i}: key {k}, {inl} inliers, lk {lk_inl}",
                  flush=True)
    cap.release()

    if flagged:
        print(f"low-inlier frames (<{MIN_INLIERS}): {flagged}")
    ws.save("homographies.json", {
        "H": [H.tolist() for H in Hs],
        "inliers": inliers,
        "lk_inliers": lk_inliers,
        "method": "ref-frame",
        "reference_frame": ref,
        "keyframes": keys,
        "keyframe_edge_inliers": {str(k): edge_inl[k] for k in keys},
        "keyframe_refine_inliers": {str(k): refine_inl[k] for k in keys},
        "blur": [round(s, 1) for s in scores.tolist()],
        "low_inlier_frames": flagged,
        "deck_mask": deck is not None,
    })
