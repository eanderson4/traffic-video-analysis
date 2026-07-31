"""Hand-set focus-lane state overrides: state_overrides.json.

Each entry pins one focus car to a state from a frame onward, sticky until
the engine's state machine next changes state, then the engine takes back
over. Managed via `tva override`; consumed by render/waveqa/wavevid.
"""
import bisect

STATES = ("rolling", "braking", "stopped")


def load(ws):
    """track id -> sorted [(frame, state)] from state_overrides.json.
    Entries with an unknown state are dropped with a warning — a typo'd
    state would otherwise survive until STATE_RGB[...] KeyErrors
    mid-render."""
    try:
        data = ws.load("state_overrides.json")
    except FileNotFoundError:
        return {}
    out = {}
    for o in data.get("overrides", []):
        if o.get("state") not in STATES:
            print(f"state_overrides: ignoring bad entry {o}")
            continue
        out.setdefault(o["track"], []).append((o["frame"], o["state"]))
    for ovs in out.values():
        ovs.sort()
    return out


def _load_edit(ws):
    try:
        return ws.load("state_overrides.json")
    except FileNotFoundError:
        return {"overrides": []}


def add(ws, track, frame, state):
    """Pin `track` to `state` from `frame` on (replaces any pin at that
    frame for that track)."""
    if state not in STATES:
        raise ValueError(f"state must be one of {STATES}, got {state!r}")
    data = _load_edit(ws)
    data["overrides"] = [o for o in data["overrides"]
                         if not (o["track"] == track
                                 and o["frame"] == frame)]
    data["overrides"].append(
        {"track": track, "frame": frame, "state": state})
    ws.save("state_overrides.json", data)


def save_all(ws, ovs):
    """Replace state_overrides.json with `ovs` ([{track, frame, state}]),
    dropping entries whose state is unknown (same rule as load). The
    editor's /save goes through here so a malformed client payload can't
    plant a state that KeyErrors the next render."""
    good = []
    for o in ovs:
        if o.get("state") not in STATES:
            print(f"state_overrides: ignoring bad entry {o}")
            continue
        good.append({"track": o["track"], "frame": o["frame"],
                     "state": o["state"]})
    ws.save("state_overrides.json", {"overrides": good})


def remove(ws, track, frame=None):
    """Drop all overrides for `track`, or just the one at `frame`."""
    data = _load_edit(ws)
    data["overrides"] = [
        o for o in data["overrides"]
        if not (o["track"] == track
                and (frame is None or o["frame"] == frame))]
    ws.save("state_overrides.json", data)


def apply(states, frames, ovs):
    """Rewrite `states` in place with this car's overrides. Each holds from
    its frame until the ENGINE's next state change (transitions are read off
    the original sequence, so overrides never extend each other)."""
    if not ovs:
        return
    orig = list(states)
    changes = [i for i in range(1, len(orig)) if orig[i] != orig[i - 1]]
    for f, st in sorted(ovs):
        k = bisect.bisect_left(frames, f)
        if k >= len(states):
            continue
        end = next((i for i in changes if i > k), len(states))
        states[k:end] = [st] * (end - k)


def stop_commits(states):
    """Sample indices where the (possibly hand-overridden) sequence commits
    to stopped - the ring/pop points. Starting out stopped is not a commit."""
    return [i for i in range(1, len(states))
            if states[i] == "stopped" and states[i - 1] != "stopped"]
