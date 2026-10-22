"""Read-only visualization of the upstream ALT real cup-grasping dataset.

No model is imported or trained. No action is executed. The original videos and
Zarr remain untouched; timestamp-consistent boundaries are used only to draw
recorded positions alongside the videos, never to create dataset splits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT / 'artifacts/cache/matplotlib_alt_preview'))

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import zarr


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def font(size):
    for path in ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/arial.ttf']:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def frames(path, size):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Cannot decode '+str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            decoded.append(cv2.cvtColor(cv2.resize(bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    assert len(decoded) == count, (path, len(decoded), count)
    return decoded, fps


def boundaries(timestamps, stored):
    jumps = np.flatnonzero(np.abs(np.diff(timestamps)) > 2.)+1
    if np.array_equal(jumps, stored[:-1]+1):
        ends = np.r_[jumps, len(timestamps)]
        rule = 'timestamp discontinuities = stored episode_ends[:-1] + 1'
    elif np.array_equal(jumps, stored[:-1]):
        ends, rule = stored.copy(), 'stored boundaries agree with timestamp discontinuities'
    else:
        raise ValueError('Ambiguous video/state alignment; do not silently guess')
    assert int(stored[-1]) == len(timestamps)
    for start, end in zip(np.r_[0, ends[:-1]], ends):
        assert np.all(np.diff(timestamps[start:end]) > 0)
    return np.asarray(ends, dtype=int), rule


def draw_paths(pos, times, starts, ends, selected, output):
    fig = plt.figure(figsize=(15.6, 4.8), layout='constrained')
    ax = fig.add_subplot(131)
    for ep, (start, end) in enumerate(zip(starts, ends)):
        path = pos[start:end]
        ax.plot(path[:, 0], path[:, 1], color=plt.cm.turbo(ep/max(1, len(ends)-1)), lw=1, alpha=.7)
        ax.scatter(*path[-1, :2], s=9, marker='x', color=plt.cm.turbo(ep/max(1, len(ends)-1)))
    ax.set(title=f'All {len(ends)} recorded episodes: XY', xlabel='X (source units)', ylabel='Y (source units)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=.2)
    ax = fig.add_subplot(132, projection='3d')
    path = pos[starts[selected]:ends[selected]]
    lines = np.stack([path[:-1], path[1:]], axis=1)
    coll = Line3DCollection(lines, cmap='viridis', linewidths=2)
    coll.set_array(np.linspace(0, 1, len(lines)))
    ax.add_collection3d(coll)
    ax.auto_scale_xyz(path[:, 0], path[:, 1], path[:, 2])
    ax.scatter(*path[0], color='#162740', s=45, marker='o', label='Start')
    ax.scatter(*path[-1], color='#ce2b37', s=60, marker='*', label='End')
    ax.set(title=f'Episode {selected}: end-effector XYZ', xlabel='X', ylabel='Y', zlabel='Z')
    ax.legend(fontsize=8)
    ax.view_init(elev=24, azim=-62)
    ax.set_title(f'Episode {selected}: end-effector XYZ\nPurple to yellow: recorded order', fontsize=11, pad=8)
    ax.set_xticks(np.linspace(path[:, 0].min(), path[:, 0].max(), 4).round())
    ax.set_yticks(np.linspace(path[:, 1].min(), path[:, 1].max(), 4).round())
    ax = fig.add_subplot(133)
    time = times[starts[selected]:ends[selected]]-times[starts[selected]]
    for dim, color, label in [(0, '#2364aa', 'X'), (1, '#dd7724', 'Y'), (2, '#30865b', 'Z')]:
        ax.plot(time, path[:, dim], color=color, lw=1.8, label=label)
    ax.set(title=f'Episode {selected}: recorded position', xlabel='Dataset elapsed time (s)', ylabel='Position (source units)')
    ax.legend()
    ax.grid(alpha=.2)
    fig.suptitle('Original ALT public dataset | real cup-grasping demonstrations | recorded end-effector positions', fontsize=13)
    fig.savefig(output, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=HERE/'config.json')
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source = (args.source_root or Path(cfg['source_root'])).resolve()
    output = (args.output_dir or PROJECT/cfg['output_directory']).resolve()
    output.relative_to(PROJECT)
    if (output/'manifest.json').exists():
        raise FileExistsError('Preview already exists; use a new --output-dir to preserve its record')
    output.mkdir(parents=True, exist_ok=True)
    results = HERE/'results'
    results.mkdir(exist_ok=True)
    data = source/cfg['dataset_subdirectory']
    root = zarr.open(str(data/'replay_buffer.zarr'), mode='r')
    pos = root['data/robot_eef_pos'][:]
    times = root['data/timestamp'][:]
    actions = root['data/action'][:]
    stored = root['meta/episode_ends'][:]
    assert np.isfinite(pos).all() and pos.shape == (len(times), 3)
    assert np.isfinite(actions).all()
    ends, boundary_rule = boundaries(times, stored)
    starts = np.r_[0, ends[:-1]]
    assert len(ends) == len(list((data/'videos').iterdir()))
    manifest = {'source': str(source), 'source_commit': subprocess.check_output(
        ['git', '-c', 'safe.directory='+source.as_posix(), '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_dataset': cfg['dataset_subdirectory'], 'episode_count': len(ends), 'state_count': len(pos),
        'stored_episode_ends': stored.tolist(), 'visualization_episode_ends': ends.tolist(),
        'boundary_rule': boundary_rule,
        'boundary_note': 'Read-only visualization alignment; no source modification or new split. Hardware synchronization is not independently verified.',
        'position_units': 'Source coordinate units; no undocumented conversion to metres or millimetres.',
        'positions_sha256': hashlib.sha256(pos.tobytes()).hexdigest(),
        'timestamp_sha256': hashlib.sha256(times.tobytes()).hexdigest(),
        'config': cfg, 'script_sha256': sha(__file__), 'episodes': [], 'training': False, 'simulation': False}
    width, height = cfg['thumbnail_width'], cfg['thumbnail_height']
    for episode in cfg['episodes']:
        start, end = int(starts[episode]), int(ends[episode])
        n = end-start
        decoded = {}
        record = {'episode': episode, 'state_start': start, 'state_end_exclusive': end,
                  'states': n, 'dataset_duration_s': float(times[end-1]-times[start]), 'videos': []}
        for camera, camera_id in cfg['cameras'].items():
            path = data/'videos'/str(episode)/(str(camera_id)+'.mp4')
            images, fps = frames(path, (width, height))
            assert abs(fps-cfg['expected_video_fps']) < .01
            assert len(images) == n or (episode == len(ends)-1 and len(images) == n+1), (episode, camera, len(images), n)
            decoded[camera] = images[:n]
            record['videos'].append({'camera': camera, 'path': str(path), 'sha256': sha(path),
                                     'frames': len(images), 'used_frames': n, 'fps': fps})
        indices = np.linspace(0, n-1, cfg['keyframes']).round().astype(int).tolist()
        sheet = Image.new('RGB', (len(indices)*width, 2*(height+36)+50), '#f4f7fa')
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 10), f'原 ALT 抓杯数据 · Episode {episode} · 上：腕部相机 / 下：第三人称相机 · 时间从左到右', font=font(19), fill='#172a3c')
        for row, camera in enumerate(cfg['cameras']):
            for col, index in enumerate(indices):
                x, y = col*width, 50+row*(height+36)
                draw.text((x+8, y+8), f'帧 {index} / {n-1}    t={times[start+index]-times[start]:.1f}s', font=font(16), fill='#172a3c')
                sheet.paste(Image.fromarray(decoded[camera][index]), (x, y+36))
        sheet.save(output/f'episode_{episode:02d}_keyframes.png')
        record['keyframe_indices'] = indices
        if episode == cfg['full_preview_episode']:
            composites = []
            for index in range(n):
                canvas = Image.new('RGB', (width*2, height+80), '#f4f7fa')
                draw = ImageDraw.Draw(canvas)
                draw.text((8, 7), '腕部相机', font=font(18), fill='#172a3c')
                draw.text((width+8, 7), '第三人称相机', font=font(18), fill='#172a3c')
                for col, camera in enumerate(cfg['cameras']):
                    canvas.paste(Image.fromarray(decoded[camera][index]), (col*width, 40))
                xyz = ', '.join(f'{x:.1f}' for x in pos[start+index])
                draw.text((8, height+48), f'Episode {episode} | 帧 {index}/{n-1} | t={times[start+index]-times[start]:.1f}s | XYZ=({xyz})', font=font(14), fill='#172a3c')
                composites.append(canvas)
            video = output/f'episode_{episode:02d}_dual_view.mp4'
            with imageio.get_writer(str(video), fps=fps, codec='libx264', quality=8, macro_block_size=1) as writer:
                for frame in composites:
                    writer.append_data(np.asarray(frame))
            check = cv2.VideoCapture(str(video))
            assert int(check.get(cv2.CAP_PROP_FRAME_COUNT)) == n
            check.release()
            gif_indices = list(range(0, n, cfg['gif_stride']))
            if gif_indices[-1] != n-1:
                gif_indices.append(n-1)
            durations = [round(1000*(b-a)/fps) for a, b in zip(gif_indices, gif_indices[1:])]+[round(1000/fps)]
            gif_frames = [composites[i].resize((480, 240), Image.Resampling.LANCZOS).quantize(colors=64)
                          for i in gif_indices]
            gif_frames[0].save(output/f'episode_{episode:02d}_dual_view.gif', save_all=True,
                               append_images=gif_frames[1:], duration=durations, loop=0, optimize=False)
            shutil.copyfile(output/f'episode_{episode:02d}_keyframes.png', results/'example_keyframes.png')
        manifest['episodes'].append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    draw_paths(pos, times, starts, ends, cfg['full_preview_episode'], output/'trajectory_paths.png')
    shutil.copyfile(output/'trajectory_paths.png', results/'trajectory_paths.png')
    manifest['artifacts'] = [{'path': p.relative_to(PROJECT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                             for p in sorted(output.iterdir()) if p.is_file()]
    for path in [output/'manifest.json', results/'manifest.json']:
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print('Completed read-only original ALT dataset visualization: '+str(output), flush=True)


if __name__ == '__main__':
    main()
