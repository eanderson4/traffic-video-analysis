"""Per-frame homographies to a reference ground plane.

Hybrid registration: every frame is first registered DIRECTLY to the
reference frame (Shi-Tomasi corners detected once on the reference +
pyramidal LK into the current frame + MAGSAC++); when the direct lock
fails (appearance gap too large for LK), the frame falls back to
chaining an adjacent-frame homography onto the previous estimate —
which is itself direct-anchored in the common case, so chain drift only
accumulates over consecutive fallback stretches. For planar
(aerial/elevated) road scenes this captures camera pan/zoom/rotation
exactly; moving vehicles fall out as robust-fit outliers, and in
congested footage most vehicles are static anchors anyway. H[i] maps
frame-i pixel coords -> reference plane (frame 0 coords).

Design lessons ported from stabilo (github.com/rfonod/stabilo), which
register.py already builds on:
- MAGSAC++ (cv2.USAC_MAGSAC) as the robust estimator instead of RANSAC
- a match-consistency gate before the fit: forward/backward LK check
  (the optical-flow analog of stabilo's cross-check / ratio filter)
- optional CLAHE on the grayscale before feature detection
- estimation at reduced scale, H rescaled to full-res coords (already
  present here)

`tva stabilize --work X --selftest` benchmarks estimator settings
against known synthetic warps — no external ground truth needed.
"""
import cv2
import numpy as np

# estimation runs at reduced scale for speed; H is rescaled to full-res coords
EST_WIDTH = 1920
MAX_CORNERS = 1500
CORNER_QUALITY = 0.005
CORNER_MIN_DIST = 18
RANSAC_TOL = 2.5  # px at estimation scale
RANSAC_METHOD = cv2.USAC_MAGSAC  # was cv2.RANSAC
FB_CHECK = True   # forward/backward LK consistency gate before the fit
FB_TOL = 1.0      # px at estimation scale
USE_CLAHE = False  # contrast enhancement before corner detection / LK
                   # (option kept from stabilo; no measured gain on dusk
                   # footage whose corners already saturate MAX_CORNERS)
DIRECT_MIN_INLIERS = 60  # accept a direct-to-reference lock above this

LK_WIN = (31, 31)
LK_CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)


def iter_gray(src, width, clahe=USE_CLAHE):
    cap = cv2.VideoCapture(src)
    enh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) \
        if clahe else None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h = int(frame.shape[0] * width / frame.shape[1])
        g = cv2.cvtColor(cv2.resize(frame, (width, h)), cv2.COLOR_BGR2GRAY)
        yield g if enh is None else enh.apply(g)
    cap.release()


def detect_corners(g):
    return cv2.goodFeaturesToTrack(g, MAX_CORNERS, CORNER_QUALITY,
                                   CORNER_MIN_DIST)


def estimate_homography(g0, g1, p0=None, method=RANSAC_METHOD, fb=FB_CHECK):
    """H mapping g1 coords -> g0 coords, plus inlier count. Corners are
    detected on g0 unless p0 is supplied (reused reference corners).
    Returns (None, 0) when tracking or the robust fit fails."""
    if p0 is None:
        p0 = detect_corners(g0)
    if p0 is None or len(p0) < 20:
        return None, 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=LK_WIN,
                                         maxLevel=4, criteria=LK_CRIT)
    good = st.ravel() == 1
    if fb:
        p0b, stb, _ = cv2.calcOpticalFlowPyrLK(
            g1, g0, p1, None, winSize=LK_WIN, maxLevel=4, criteria=LK_CRIT)
        back = np.linalg.norm((p0b - p0).reshape(-1, 2), axis=1)
        good &= (stb.ravel() == 1) & (back <= FB_TOL)
    if good.sum() < 20:
        return None, 0
    H, inl = cv2.findHomography(p1[good], p0[good], method, RANSAC_TOL)
    if H is None:
        return None, 0
    return H, int(inl.sum())


def adjacent_homography(g0, g1):
    """H mapping g1 coords -> g0 coords, plus inlier count."""
    H, ninl = estimate_homography(g0, g1)
    return (np.eye(3), 0) if H is None else (H, ninl)


def h_delta(Ha, Hb, shape, n=8):
    """Median displacement (px, at the frames' own scale) between two
    homographies over an n x n grid of image points."""
    h, w = shape
    xs = np.linspace(w * 0.05, w * 0.95, n)
    ys = np.linspace(h * 0.05, h * 0.95, n)
    pts = np.array([(x, y) for y in ys for x in xs])
    return float(np.median(np.hypot(
        *(apply_h(Ha, pts) - apply_h(Hb, pts)).T)))


def run(ws):
    meta = ws.meta
    scale = EST_WIDTH / meta["width"]
    S = np.diag([scale, scale, 1.0])
    S_inv = np.diag([1 / scale, 1 / scale, 1.0])

    Hs = [np.eye(3)]
    inliers = [0]        # adjacent-fit inliers (comparable across versions)
    direct_inl = [-1]    # direct-to-reference inliers (-1: not attempted)
    deltas = []          # direct-vs-chain disagreement (per-hop chain error)
    n_direct = 0
    prev = g_ref = p_ref = None
    for i, g in enumerate(iter_gray(meta["src"], EST_WIDTH)):
        if i == 0:
            g_ref = g
            p_ref = detect_corners(g_ref)
        else:
            H_adj, ninl = adjacent_homography(prev, g)
            H_chain = Hs[-1] @ H_adj
            H_dir, ndir = estimate_homography(g_ref, g, p0=p_ref)
            if H_dir is not None and ndir >= DIRECT_MIN_INLIERS:
                Hs.append(H_dir)
                n_direct += 1
                deltas.append(h_delta(H_dir, H_chain, g.shape))
            else:
                Hs.append(H_chain)
            inliers.append(ninl)
            direct_inl.append(ndir)
            if i % 50 == 0:
                how = "direct" if H_dir is not None \
                    and ndir >= DIRECT_MIN_INLIERS else "chain"
                print(f"frame {i}: {ninl} adj inliers, {ndir} direct "
                      f"({how})", flush=True)
        prev = g

    n = len(Hs) - 1
    if deltas:
        d = np.asarray(deltas)
        print(f"direct lock on {n_direct}/{n} frames; chain-vs-direct "
              f"disagreement: median {np.median(d):.2f} px, "
              f"p95 {np.percentile(d, 95):.2f}, max {d.max():.2f} "
              f"(est scale)")
    else:
        print(f"direct lock on {n_direct}/{n} frames (all chained)")

    # rescale to full-res pixel coords, normalize
    out = []
    for H in Hs:
        Hf = S_inv @ H @ S
        out.append((Hf / Hf[2, 2]).tolist())
    ws.save("homographies.json", {
        "H": out,
        "inliers": inliers,
        "direct_inliers": direct_inl,
        "method": "hybrid-direct-chain",
        "reference_frame": 0,
    })


def apply_h(H, xy):
    """Map (n,2) points through a 3x3 homography."""
    pts = np.asarray(xy, dtype=np.float64)
    ones = np.ones((len(pts), 1))
    p = np.hstack([pts, ones]) @ np.asarray(H).T
    return p[:, :2] / p[:, 2:3]


def qa(ws, times=(0, 4, 8, 12, 16)):
    """Warp sample frames into the reference plane on a common canvas."""
    meta = ws.meta
    Hs = [np.asarray(h) for h in ws.load("homographies.json")["H"]]
    w, h, fps = meta["width"], meta["height"], meta["fps"]
    idxs = [min(int(t * fps), len(Hs) - 1) for t in times]

    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    warped = np.vstack([apply_h(Hs[i], corners) for i in idxs])
    x0, y0 = warped.min(axis=0)
    x1, y1 = warped.max(axis=0)
    out_scale = min(1.0, 3000.0 / max(x1 - x0, y1 - y0))
    T = np.array([[out_scale, 0, -x0 * out_scale],
                  [0, out_scale, -y0 * out_scale],
                  [0, 0, 1]])
    cw, ch = int((x1 - x0) * out_scale) + 2, int((y1 - y0) * out_scale) + 2

    cap = cv2.VideoCapture(meta["src"])
    acc = np.zeros((ch, cw, 3), dtype=np.float32)
    cnt = np.zeros((ch, cw, 1), dtype=np.float32)
    for k, i in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        wf = cv2.warpPerspective(frame, T @ Hs[i], (cw, ch))
        mask = cv2.warpPerspective(np.ones((h, w), np.float32), T @ Hs[i],
                                   (cw, ch))[..., None]
        acc += wf.astype(np.float32) * mask
        cnt += mask
        cv2.imwrite(str(ws.qa) + f"/stab-warp-t{int(i / fps)}.jpg", wf,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    cap.release()
    blend = (acc / np.maximum(cnt, 1e-6)).astype(np.uint8)
    cv2.imwrite(str(ws.qa) + "/stab-blend.jpg", blend,
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"QA: {ws.qa}/stab-blend.jpg (sharp road = good registration)")


# --- self-test: recovery of KNOWN synthetic warps, no ground truth needed ---

SELFTEST_FRAMES = 8
SELFTEST_ROT = 2.0      # max rotation, degrees
SELFTEST_SCALE = 0.03   # max scale deviation
SELFTEST_TRANS = 15.0   # max translation, px at estimation scale
SELFTEST_CARS = 12      # independently moving patches (robust-fit outliers)
SELFTEST_NOISE = 1.5    # Gaussian noise sigma on the warped frame

# old -> new, one lesson at a time
SELFTEST_CONFIGS = [
    ("old (ransac)", dict(method=cv2.RANSAC, fb=False, clahe=False)),
    ("magsac", dict(method=cv2.USAC_MAGSAC, fb=False, clahe=False)),
    ("new (magsac+fb)", dict(method=cv2.USAC_MAGSAC, fb=True,
                             clahe=False)),
    ("magsac+fb+clahe", dict(method=cv2.USAC_MAGSAC, fb=True,
                             clahe=True)),
]


def synth_warp(rng, shape, rot=None, scale=None, trans=None):
    """Random small rotation/scale/translation homography about the frame
    center, mapping source coords -> warped coords."""
    rot = SELFTEST_ROT if rot is None else rot
    scale = SELFTEST_SCALE if scale is None else scale
    trans = SELFTEST_TRANS if trans is None else trans
    h, w = shape
    a = np.deg2rad(rng.uniform(-rot, rot))
    s = 1 + rng.uniform(-scale, scale)
    t = rng.uniform(-trans, trans, 2)
    c, sn = np.cos(a) * s, np.sin(a) * s
    cx, cy = w / 2, h / 2
    return np.array([[c, -sn, cx - c * cx + sn * cy + t[0]],
                     [sn, c, cy - sn * cx - c * cy + t[1]],
                     [0, 0, 1]])


def synth_frame(g, M, rng, n_cars=SELFTEST_CARS):
    """Warp g by M, add sensor noise, then inject n_cars patches moved by
    an extra independent shift (fake moving vehicles — gross outliers for
    the robust fit)."""
    h, w = g.shape
    gw = cv2.warpPerspective(g, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    gw = np.clip(gw.astype(np.float32)
                 + rng.normal(0, SELFTEST_NOISE, gw.shape), 0, 255
                 ).astype(np.uint8)
    for _ in range(n_cars):
        pw, ph = int(w * rng.uniform(0.03, 0.08)), \
            int(h * rng.uniform(0.03, 0.08))
        x, y = rng.uniform(0, w - pw), rng.uniform(0, h - ph)
        d = rng.uniform(15, 45, 2) * rng.choice([-1.0, 1.0], 2)
        T = np.array([[1, 0, d[0]], [0, 1, d[1]], [0, 0, 1.0]])
        gm = cv2.warpPerspective(g, T @ M, (w, h),
                                 borderMode=cv2.BORDER_REPLICATE)
        corn = np.array([[x, y], [x + pw, y], [x + pw, y + ph],
                         [x, y + ph]])
        qc = apply_h(T @ M, corn)
        x0, y0 = np.maximum(np.floor(qc.min(axis=0)), 0).astype(int)
        x1 = min(int(np.ceil(qc[:, 0].max())), w)
        y1 = min(int(np.ceil(qc[:, 1].max())), h)
        gw[y0:y1, x0:x1] = gm[y0:y1, x0:x1]
    return gw


def synth_pair(g, rng):
    """Warp g by a known homography (with noise + moving patches).
    Returns (warped, M)."""
    M = synth_warp(rng, g.shape)
    return synth_frame(g, M, rng), M


def recovery_error(H_est, M, shape, n=16):
    """Per-point recovery error (px): source grid points mapped into the
    warped frame by the true M, then back by the estimated H."""
    h, w = shape
    xs = np.linspace(w * 0.05, w * 0.95, n)
    ys = np.linspace(h * 0.05, h * 0.95, n)
    pts = np.array([(x, y) for y in ys for x in xs])
    return np.hypot(*(apply_h(H_est, apply_h(M, pts)) - pts).T)


SELFTEST_HOPS = 30      # sequence length for the drift test


def drift_test(g0, n_hops=SELFTEST_HOPS, seed=1):
    """Random-walk camera: warp one real frame by n_hops cumulative small
    homographies, then estimate the final frame's H to the first frame
    two ways — pure adjacent chaining (old pipeline) vs the hybrid
    direct-to-reference fallback (new pipeline) — and compare recovery
    error against the known cumulative warp."""
    rng = np.random.default_rng(seed)
    Ms, frames = [np.eye(3)], [g0]
    for k in range(n_hops):
        M = synth_warp(rng, g0.shape, rot=0.5, scale=0.005, trans=5.0) \
            @ Ms[-1]
        Ms.append(M)
        frames.append(synth_frame(g0, M, rng, n_cars=6))

    p_ref = detect_corners(frames[0])
    Hc = Hh = np.eye(3)
    n_direct = 0
    for k in range(1, len(frames)):
        H_adj, _ = estimate_homography(frames[k - 1], frames[k])
        if H_adj is None:
            H_adj = np.eye(3)
        Hc = Hc @ H_adj
        H_dir, ndir = estimate_homography(frames[0], frames[k], p0=p_ref)
        if H_dir is not None and ndir >= DIRECT_MIN_INLIERS:
            Hh = H_dir
            n_direct += 1
        else:
            Hh = Hh @ H_adj
    err_c = np.median(recovery_error(Hc, Ms[-1], g0.shape))
    err_h = np.median(recovery_error(Hh, Ms[-1], g0.shape))
    print(f"sequence drift test ({n_hops} random-walk hops, "
          f"{n_direct} direct locks): chain med err {err_c:.2f} px, "
          f"hybrid med err {err_h:.2f} px")


def selftest(ws, n_frames=SELFTEST_FRAMES, seed=0):
    """Estimate homographies on synthetically warped real frames and
    report recovery error per estimator configuration (old vs new)."""
    meta = ws.meta
    idxs = np.linspace(0, meta["n_frames"] - 1, n_frames).astype(int)
    cap = cv2.VideoCapture(meta["src"])
    raw = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        h = int(frame.shape[0] * EST_WIDTH / frame.shape[1])
        g = cv2.cvtColor(cv2.resize(frame, (EST_WIDTH, h)),
                         cv2.COLOR_BGR2GRAY)
        raw.append((int(i), g))
    cap.release()

    print(f"selftest: {len(raw)} frames, synthetic warps up to "
          f"{SELFTEST_ROT} deg / {SELFTEST_SCALE:.0%} scale / "
          f"{SELFTEST_TRANS:.0f} px + {SELFTEST_CARS} moving patches")
    print(f"{'config':<26} {'med err':>8} {'p95 err':>8} {'max err':>8} "
          f"{'inliers':>8} {'fails':>5}")
    for name, cfg in SELFTEST_CONFIGS:
        enh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) \
            if cfg["clahe"] else None
        errs, inls, fails = [], [], 0
        for i, g0 in raw:
            if enh is not None:
                g0 = enh.apply(g0)
            rng = np.random.default_rng(seed * 100003 + i)
            gw, M = synth_pair(g0, rng)
            H, ninl = estimate_homography(g0, gw, method=cfg["method"],
                                          fb=cfg["fb"])
            if H is None:
                fails += 1
                continue
            errs.append(np.median(recovery_error(H, M, g0.shape)))
            inls.append(ninl)
        e = np.asarray(errs)
        print(f"{name:<26} {np.median(e):8.3f} "
              f"{np.percentile(e, 95):8.3f} {e.max():8.3f} "
              f"{np.mean(inls):8.0f} {fails:5d}")
    if raw:
        drift_test(raw[len(raw) // 2][1], seed=seed + 1)
