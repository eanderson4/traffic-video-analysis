"""Direct-to-reference registration (v2, experiment R1).

Replaces chained stabilization: every frame registers to its nearest
keyframe, keyframes register to one blur-scored reference frame (graph
depth <= 2, no 400-link chain). All detected vehicle bboxes (dilated 15%)
are exclusion masks during feature detection — in a jam most corners sit
on possibly-creeping cars, which is poison for the road-plane homography.

Estimation is stabilo (Geo-trax's stabilization core): ORB at 0.5x
downscale, BF kNN + SNN ratio 0.9, MAGSAC++ (cv2.USAC_MAGSAC). An optional
deck.json ({"polygon": [[x, y], ...]} in full-res pixels) further restricts
features to the elevated carriageway so ground-plane parallax can't poison
the fit.

Output schema matches stabilize.py: homographies.json {"H": [...]} with
H[i] mapping frame-i full-res pixels -> reference-frame pixels, plus
"method": "ref-frame" and provenance (reference frame, keyframes, blur
scores, inlier counts). Downstream world/render run unchanged.
"""
from collections import defaultdict

import cv2
import numpy as np

from stabilo import Stabilizer

BLUR_WIDTH = 960          # downscale width for blur scoring
KEYFRAME_SPACING_S = 2.0  # one keyframe per window of this length
REF_WINDOW = (0.25, 0.75) # reference candidates: this fraction of the clip
MAX_FEATURES = 4000       # ORB budget per frame (ref gets 2x via stabilo)
MASK_MARGIN = 0.15        # bbox dilation for exclusion masks
MIN_INLIERS = 30          # below this a frame is flagged in QA output


def make_stabilizer(deck_mask=None):
    stab = Stabilizer(
        detector_name="orb", matcher_name="bf", filter_type="ratio",
        filter_ratio=0.9, transformation_type="projective",
        downsample_ratio=0.5, max_features=MAX_FEATURES,
        ransac_method=cv2.USAC_MAGSAC, ransac_epipolar_threshold=2.0,
        mask_use=True, mask_margin_ratio=MASK_MARGIN)
    if deck_mask is not None:
        orig = stab.create_binary_mask

        def masked(boxes, box_format):
            m = orig(boxes, box_format)
            return cv2.bitwise_and(m, deck_mask)
        stab.create_binary_mask = masked
    return stab


def boxes_per_frame(ws):
    """{frame: (n,4) float32 [cx cy w h]} over ALL detections."""
    per = defaultdict(list)
    for tr in ws.load("tracks.json")["tracks"]:
        for f, cx, cy, w, h, conf in tr["obs"]:
            per[f].append((cx, cy, w, h))
    return {f: np.asarray(v, np.float32) for f, v in per.items()}


def load_deck_mask(ws):
    """Optional include-mask from deck.json polygon (full-res pixels)."""
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


def blur_pass(meta, boxes):
    """Variance of Laplacian on vehicle-masked, downscaled frames."""
    scale = BLUR_WIDTH / meta["width"]
    scores = []
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
        i += 1
    cap.release()
    return np.asarray(scores)


def pick_frames(scores, fps, n):
    """Reference = sharpest mid-clip frame; keyframes = sharpest per ~2s
    window (the reference replaces its window's pick)."""
    a, b = int(n * REF_WINDOW[0]), int(n * REF_WINDOW[1])
    ref = a + int(np.argmax(scores[a:b]))
    spacing = max(1, round(KEYFRAME_SPACING_S * fps))
    keys = []
    for w0 in range(0, n, spacing):
        w1 = min(n, w0 + spacing)
        k = w0 + int(np.argmax(scores[w0:w1]))
        keys.append(ref if w0 <= ref < w1 else k)
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


def run(ws):
    meta = ws.meta
    n, fps = meta["n_frames"], meta["fps"]
    boxes = boxes_per_frame(ws)
    deck = load_deck_mask(ws)

    scores = blur_pass(meta, boxes)
    n = min(n, len(scores))
    ref, keys = pick_frames(scores, fps, n)
    print(f"reference frame {ref} (blur {scores[ref]:.0f}), "
          f"{len(keys)} keyframes: {keys}")

    imgs = grab_frames(meta["src"], set(keys) | {ref})

    # keyframe -> reference homographies
    ref_stab = make_stabilizer(deck)
    ref_stab.set_ref_frame(imgs[ref], boxes.get(ref))
    H_key, key_inl = {}, {}
    for k in keys:
        if k == ref:
            H_key[k], key_inl[k] = np.eye(3), -1
            continue
        ref_stab.stabilize(imgs[k], boxes.get(k))
        H = ref_stab.get_cur_trans_matrix()
        inl = ref_stab.get_cur_inliers_count() or 0
        if H is None:
            raise RuntimeError(f"keyframe {k} -> reference failed")
        H_key[k], key_inl[k] = np.asarray(H), inl
        print(f"  keyframe {k} -> ref: {inl} inliers")

    # one stabilizer per keyframe (its reference = the keyframe)
    kstabs = {}
    for k in keys:
        kstabs[k] = make_stabilizer(deck)
        kstabs[k].set_ref_frame(imgs[k], boxes.get(k))
    del imgs

    key_arr = np.asarray(keys)
    Hs, inliers, flagged = [], [], []
    cap = cv2.VideoCapture(meta["src"])
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        k = int(key_arr[np.argmin(np.abs(key_arr - i))])
        if i == k:
            H, inl = H_key[k], key_inl[k]
        else:
            kstabs[k].stabilize(frame, boxes.get(i))
            Hk = kstabs[k].get_cur_trans_matrix()
            inl = kstabs[k].get_cur_inliers_count() or 0
            H = H_key[k] @ (np.eye(3) if Hk is None else np.asarray(Hk))
        if 0 <= inl < MIN_INLIERS:
            flagged.append(i)
        Hs.append(H / H[2, 2])
        inliers.append(int(inl))
        if i % 50 == 0:
            print(f"frame {i}: key {k}, {inl} inliers", flush=True)
    cap.release()

    if flagged:
        print(f"low-inlier frames (<{MIN_INLIERS}): {flagged}")
    ws.save("homographies.json", {
        "H": [H.tolist() for H in Hs],
        "inliers": inliers,
        "method": "ref-frame",
        "reference_frame": ref,
        "keyframes": keys,
        "keyframe_inliers": {str(k): key_inl[k] for k in keys},
        "blur": [round(s, 1) for s in scores.tolist()],
        "low_inlier_frames": flagged,
        "deck_mask": deck is not None,
    })
