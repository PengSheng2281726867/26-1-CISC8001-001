"""Render implementation-level diagrams for the presentation and appendix."""
from build_charts import HERE, FancyArrowPatch, FancyBboxPatch, finish, plt

OUT = HERE / "materials/architectures"
PALETTE = {
    "input": ("#3A6986", "#EAF3F8"),
    "learned": ("#42509A", "#EDF0FF"),
    "rule": ("#2C7561", "#E9F6EF"),
    "train": ("#945924", "#FFF3E4"),
    "oracle": ("#963E52", "#FCECF0"),
}


class Diagram:
    def __init__(self, title, subtitle):
        self.fig, self.ax = plt.subplots(figsize=(13, 5.6))
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.ax.set(xlim=(0, 13), ylim=(0, 5.6))
        self.ax.axis("off")
        self.ax.text(.42, 5.18, title, color="#182638", fontsize=23, weight="bold", va="center")
        self.ax.text(.44, 4.76, subtitle, color="#475569", fontsize=11.5, va="center")

    def box(self, x, y, w, title, detail, kind="learned", *, h=.9):
        edge, fill = PALETTE[kind]
        self.ax.add_patch(FancyBboxPatch((x, y - h/2), w, h, boxstyle="round,pad=.02,rounding_size=.11",
                                        linewidth=1.28, edgecolor=edge, facecolor=fill, zorder=2))
        self.ax.text(x+w/2, y+.16, title, ha="center", va="center",
                     color=edge, weight="bold", fontsize=11, zorder=3)
        self.ax.text(x+w/2, y-.13, detail, ha="center", va="center",
                     color="#243246", fontsize=9.4, linespacing=1.22, zorder=3)
        return (x, y, w)

    def arrow(self, start, end, style="solid", color="#627187", curve=0):
        self.ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                         shrinkA=2, shrinkB=3, color=color, linewidth=1.35,
                                         linestyle=style, connectionstyle=f"arc3,rad={curve}"))

    def link(self, source, target, **kwargs):
        x0, y0, w0 = source
        x1, y1, _ = target
        self.arrow((x0+w0, y0), (x1, y1), **kwargs)

    def down(self, source, target, **kwargs):
        x0, y0, w0 = source
        x1, y1, w1 = target
        self.arrow((x0+w0/2, y0-.45), (x1+w1/2, y1+.45), **kwargs)

    def routed(self, points, color="#627187"):
        self.ax.plot(*zip(*points[:-1]), color=color, linewidth=1.35)
        self.arrow(points[-2], points[-1], color=color)

    def note(self, y, text, *, x=.44, color="#475569"):
        self.ax.text(x, y, text, va="center", color=color, fontsize=10.5)

    def save(self, stem):
        finish(self.fig, OUT / stem)


def alt(*, lift=False, b1=False):
    title = "Lift-PH | ALT" if lift else ("Push-T | ALT with pose supervision" if b1 else "Push-T | Visual ALT")
    subtitle = ("Two cameras at the current timestep" if lift else "One current RGB frame + 2-D pusher position")
    d = Diagram(title, subtitle)
    rgb = d.box(.45, 3.65, 2.22, "Current RGB", "2 × (3 × 84 × 84)" if lift else "3 × 96 × 96", "input")
    image = d.box(3.09, 3.65, 2.53, "ResNet-18", "shared weights, 2 × 128 features" if lift else "128 image features")
    state = d.box(.45, 2.28, 2.22, "Proprioception" if lift else "Pusher position",
                  "XYZ + quaternion + gripper (9D)" if lift else "XY (2D)", "input")
    pose = d.box(3.09, 2.28, 2.53, "Position MLP", "9 → 64 → 32" if lift else "2 → 64 → 32")
    fusion = d.box(6.17, 2.96, 2.37, "Concat + fusion MLP", ("288" if lift else "160") + " → 256 → 128")
    norm = d.box(9.10, 2.96, 3.17, "L2 normalize", "128-D retrieval embedding")
    for source, target in ((rgb, image), (state, pose), (image, fusion), (pose, fusion), (fusion, norm)):
        d.link(source, target)
    search = d.box(9.10, 1.63, 3.17, "Cosine top-1 lookup", "training bank → episode and frame", "rule")
    d.down(norm, search)
    d.box(6.17, 1.63, 2.37, "Replay action",
          "one 7-D command" if lift else "eight 2-D targets", "rule")
    d.arrow((9.10, 1.63), (8.54, 1.63))
    if b1:
        auxiliary = d.box(6.17, 4.04, 5.28, "Training-only pose head", "128 → 64 → 4  |  x, y, sin θ, cos θ", "train", h=.7)
        d.link(image, auxiliary, style="dashed", color=PALETTE["train"][0])
        d.note(.46, "T pose supervises the image branch in training. The auxiliary head is inactive at evaluation.")
    else:
        d.note(.46, "ImageNet initialization and NT-Xent fine-tuning on two augmented views. Retrieval uses the training split.")
    d.save("lift_alt" if lift else ("pusht_alt_b1" if b1 else "pusht_alt"))


def dp(*, lift=False, official=False):
    task = "Lift-PH" if lift else "Push-T"
    title = task + (" | Official visual CNN checkpoint" if official else " | Diffusion Policy CNN")
    d = Diagram(title, "RoboMimic observation encoder and a conditional 1-D U-Net over the action horizon")
    width = "256 / 512 / 1024" if lift else ("512 / 1024 / 2048" if official else "128 / 256 / 512")
    action = 7 if lift else 2
    inference = "10 DDIM steps" if lift else ("100 DDPM steps" if official else "16 DDIM steps")
    visual = d.box(.45, 3.68, 2.55, "RGB observations",
                   "2 cameras × one timestep\n84 × 84" if lift else "one camera × two timesteps\n96 × 96 → crop 84 × 84", "input")
    state = d.box(.45, 2.43, 2.55, "Robot state",
                  "end-effector + gripper (9D)" if lift else "pusher XY per timestep", "input")
    encoder = d.box(3.53, 3.68, 2.8, "Visual core", "ResNet-18 + SpatialSoftmax")
    features = d.box(6.82, 3.39, 2.62, "Global condition",
                     "concat RGB features + state" if lift else "concat features over 2 frames")
    noisy = d.box(.45, 1.11, 2.55, "Noisy action block", f"16 × {action} and diffusion timestep", "input")
    unet = d.box(3.72, 1.11, 2.7, "Conditional 1-D U-Net", f"down / mid / up with skips\nchannels {width}")
    scheduler = d.box(7.02, 1.11, 2.42, "Denoise iteratively", inference)
    execute = d.box(10.04, 1.11, 2.49, "Execute / reobserve",
                    "first 7-D action" if lift else "8 absolute 2-D targets", "rule")
    for source, target in ((visual, encoder), (encoder, features),
                           (noisy, unet), (unet, scheduler), (scheduler, execute)):
        d.link(source, target)
    x, y, w = state
    d.routed([(x+w, y), (6.58, y), (6.58, 3.16), (6.82, 3.16)])
    d.arrow((7.6, 2.92), (5.12, 1.57), color=PALETTE["learned"][0], curve=.14)
    d.note(.32, "Training uses 100 diffusion timesteps. The inference sampler starts from Gaussian action noise.")
    d.save("lift_dp" if lift else ("pusht_dp_official" if official else "pusht_dp"))


def geometry():
    d = Diagram("Push-T | Geometry-rule retrieval", "Simulator ground-truth T pose + observed pusher position")
    state = d.box(.43, 3.02, 2.36, "Current state", "pusher XY + T XYθ", "oracle")
    retrieve = d.box(3.31, 3.02, 2.57, "Rank training frames", "global / relative geometry", "rule")
    candidate = d.box(6.43, 3.02, 2.57, "Select a candidate", "episode and frame index", "rule")
    actions = d.box(9.56, 3.02, 2.96, "Get stored action block", "absolute XY targets", "rule")
    for source, target in ((state, retrieve), (retrieve, candidate), (candidate, actions)):
        d.link(source, target)
    d.box(6.43, 1.73, 2.57, "Contact gate variant", "same edge + distance limits", "rule")
    aligned = d.box(9.56, 1.73, 2.96, "Replay / align", "optional object-frame transform", "rule")
    d.arrow((7.72, 2.21), (7.72, 2.54), style="dashed", color=PALETTE["rule"][0])
    d.down(actions, aligned)
    d.note(.46, "Geometry and contact thresholds are fixed rules. Optional gates and alignment belong to distinct ablations.")
    d.save("pusht_geometry")


def residual():
    d = Diagram("Push-T | Residual action network", "A geometry rule retrieves the action block. The network predicts its correction.")
    state = d.box(.45, 3.61, 2.64, "Geometry context", "8 values for local pusher / T offsets", "oracle")
    history = d.box(.45, 2.27, 2.64, "Retrieved action prior", "8 × 2 targets in historical T frame", "rule")
    concat = d.box(3.62, 2.94, 2.02, "Concatenate", "24-D input", "input")
    mlp = d.box(6.15, 2.94, 2.64, "Residual MLP", "24 → 64 → 64 → 16\nReLU between hidden layers")
    output = d.box(9.35, 2.94, 3.16, "8 × 2 residual Δ", "linear / bounded tanh / taper")
    for source, target in ((state, concat), (history, concat), (concat, mlp), (mlp, output)):
        d.link(source, target)
    execute = d.box(9.35, 1.52, 3.16, "Add to prior and execute", "current T frame → world XY", "rule")
    d.down(output, execute)
    d.routed([(1.77, 1.82), (1.77, 1.52), (9.35, 1.52)], color=PALETTE["rule"][0])
    d.note(1.75, "Retrieved prior in object coordinates", x=3.64, color=PALETTE["rule"][0])
    d.note(.79, "The tapered variant scales the bounded residual by weights that decrease from 1 to 0 across the eight actions.")
    d.note(.37, "Geometry uses simulator ground-truth T pose. The network input contains geometry and retrieved targets.")
    d.save("pusht_residual")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path",
                         "svg.hashsalt": "course-architecture"})
    alt(lift=True)
    dp(lift=True)
    alt()
    alt(b1=True)
    dp()
    dp(official=True)
    geometry()
    residual()
