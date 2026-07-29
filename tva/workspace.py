"""Per-clip work directory: meta probe + artifact paths."""
import json
import os
import subprocess


class Workspace:
    def __init__(self, work):
        self.dir = os.path.abspath(work)
        self.qa = os.path.join(self.dir, "qa")

    def path(self, name):
        return os.path.join(self.dir, name)

    def load(self, name):
        with open(self.path(name)) as fh:
            return json.load(fh)

    def save(self, name, obj):
        os.makedirs(self.dir, exist_ok=True)
        with open(self.path(name), "w") as fh:
            json.dump(obj, fh)
        print(f"wrote {self.path(name)}")

    @property
    def meta(self):
        return self.load("meta.json")


def init(src, work):
    src = os.path.abspath(src)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
         "-of", "json", src],
        capture_output=True, text=True, check=True)
    st = json.loads(probe.stdout)["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    ws = Workspace(work)
    os.makedirs(ws.qa, exist_ok=True)
    ws.save("meta.json", {
        "src": src,
        "width": st["width"],
        "height": st["height"],
        "fps": float(num) / float(den),
        "n_frames": int(st["nb_frames"]),
    })
    return ws
