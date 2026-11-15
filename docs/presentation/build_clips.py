"""Cut two presentation excerpts from local recordings and verify their run provenance."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

LIFT = Path("experiments/alt_robomimic_lift")
LIFT_RUN = "runs/alt_lift_ph30_image_proprio_seed42_v1/evaluations/evaluation01"
LIFT_TABLE = LIFT / "experiments" / LIFT_RUN / "eval_rollouts.csv"
PUSH_TABLE = Path("experiments/pusht_residual/experiments/residual_study/evidence/round3_evaluate_aligned.csv")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def row(path, seed_field, seed):
    with path.open(newline="", encoding="utf-8") as stream:
        return next(r for r in csv.DictReader(stream) if int(r[seed_field]) == seed)


def sample(reader, indices):
    return {i: reader.get_data(i) for i in indices}


def composite(frame, title, subtitle):
    canvas = Image.new("RGB", (960, 540), "#142334")
    pic = Image.fromarray(frame)
    pic.thumbnail((936, 424))
    factor = min(936 / pic.width, 424 / pic.height)
    pic = pic.resize((round(pic.width * factor), round(pic.height * factor)), Image.Resampling.BICUBIC)
    canvas.paste(pic, ((960 - pic.width) // 2, 49 + (424 - pic.height) // 2))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 23)
    small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 21)
    draw.text((23, 11), title, font=title_font, fill="white")
    draw.text((23, 490), subtitle, font=small, fill="#DDEAF4")
    return canvas


def save_clip(output, frames, title, captions, poster, poster_path, fps):
    poster_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=fps, codec="libx264", quality=7,
                            macro_block_size=1, output_params=["-movflags", "+faststart"]) as writer:
        for i, frame in frames.items():
            canvas = composite(frame, title, captions(i))
            writer.append_data(np.asarray(canvas))
            if poster == i:
                canvas.save(poster_path)
    assert poster_path.is_file()
    return {"filename": output.name, "bytes": output.stat().st_size, "sha256": sha(output),
            "seconds": len(frames) / fps, "fps": fps, "frames": len(frames),
            "poster": poster_path.relative_to(ROOT).as_posix()}


def lift_clip(output):
    seed = 100049
    source = ROOT / LIFT / "artifacts" / LIFT_RUN / "videos" / f"seed_{seed}.mp4"
    record = row(ROOT / LIFT_TABLE, "eval_seed", seed)
    assert int(record["steps_to_success"]) == 40 and float(record["final_reward"]) == 0
    indices = [*range(0, 151), *range(930, 1001)]
    with imageio.get_reader(source) as reader:
        assert reader.count_frames() == 1001 and reader.get_meta_data()["fps"] == 20
        frames = sample(reader, indices)

    def captions(step):
        event = "First height crossing: step 40" if step < 930 else "Excerpt jumps ahead to the end of the episode"
        if step == 1000:
            event = "Final step: below the lift threshold"
        return f"Step {step:,} / 1,000   |   {event}"

    result = save_clip(output / "lift-alt30-seed100049.mp4", frames,
                       "Lift-PH  |  ALT-30  |  Seed 100049", captions, 40,
                       HERE / "materials/lift/lift_video_still.png", 20)
    return {**result, "seed": seed, "source": source.relative_to(ROOT).as_posix(),
            "source_sha256": sha(source), "rollouts": LIFT_TABLE.as_posix(),
            "excerpt_steps_inclusive": [[0, 150], [930, 1000]], "first_success_step": 40,
            "final_reward": float(record["final_reward"]), "run": "standalone ALT-30 evaluation01"}


def pusht_clip(output, previous, dp_source):
    sys.path.insert(0, str(dp_source))
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv

    seed = 100003
    record = row(ROOT / PUSH_TABLE, "evaluation_seed", seed)
    with (previous / "queries.jsonl").open(encoding="utf-8") as stream:
        queries = [r for r in map(json.loads, stream) if r["evaluation_seed"] == seed]
    env = PushTImageEnv(legacy=True, render_size=96)
    env.seed(seed)
    env.reset()
    scores, renders = [], []
    try:
        for query in queries:
            assert query["current_step"] == len(scores)
            for command in query["commands"][:query["executed_actions"]]:
                _, score, done, _ = env.step(np.asarray(command, dtype=np.float32))
                assert not done
                scores.append(float(score))
                renders.append(env.render("rgb_array").copy())
    finally:
        env.close()
    assert len(scores) == int(record["steps"]) == 300
    np.testing.assert_allclose([max(scores), scores[-1]],
                               [float(record["official_episode_score"]), float(record["final_score"])],
                               atol=1e-12, rtol=0)
    peak = int(np.argmax(scores))
    # Cross-check the original MP4 against re-rendering the logged commands.
    # The best of the four stored clips must match the recorded rollout.
    anchor_frames = sorted({0, 75, 150, 225, 299, peak})
    matches = []
    for path in sorted((previous / "media").glob("*.mp4")):
        with imageio.get_reader(path) as reader:
            if reader.count_frames() != len(scores):
                continue
            anchors = sample(reader, anchor_frames)
        error = np.mean([np.abs(anchors[i].astype(float) - renders[i]).mean() for i in anchors])
        matches.append((float(error), path))
    ranked = sorted(matches)
    error, source = ranked[0]
    assert error < 4 and error < ranked[1][0] * .5, ranked
    first_range = range(max(0, peak - 42), min(249, peak + 38) + 1)
    assert peak in first_range
    last_range = range(250, 300)
    with imageio.get_reader(source) as reader:
        frames = sample(reader, [*first_range, *last_range])

    def captions(index):
        marker = "Peak" if index == peak else "Final step" if index == 299 else "Coverage score"
        phase = "   |   Final excerpt" if index in last_range else ""
        return f"Step {index + 1:03d} / 300   |   {marker}: {scores[index]:.3f}{phase}"

    result = save_clip(output / "pusht-aligned-seed100003.mp4", frames,
                       "Push-T  |  Pose alignment  |  Simulator ground-truth pose",
                       captions, peak, HERE / "materials/pusht_geometry/pusht_video_still.png", 10)
    return {**result, "seed": seed, "source": source.name,
            "source_sha256": sha(source), "rollouts": PUSH_TABLE.as_posix(),
            "queries_sha256": sha(previous / "queries.jsonl"), "mean_video_match_error": error,
            "excerpt_steps_inclusive": [[first_range.start + 1, first_range.stop], [251, 300]],
            "peak_step": peak + 1, "peak": max(scores), "final": scores[-1],
            "run": "residual study round3_evaluate aligned"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-aligned", type=Path,
                        default=ROOT.parent / "push_T_alt/artifacts/round3_evaluate/aligned")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Use a new directory within this project's artifacts/")
    output.mkdir(parents=True)
    config = yaml.safe_load((ROOT / "configs/alt_pusht_base.yaml").read_text(encoding="utf-8"))
    dp_source = ROOT / config["paths"]["dp_source"]
    records = [lift_clip(output), pusht_clip(output, args.residual_aligned, dp_source)]
    (HERE / "materials/media_manifest.json").write_text(
        json.dumps({"release_tag": "presentation-media-v1", "clips": records}, indent=2) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
