"""Reconstruct one training batch and visualize actual positive/negative rules.

Runs only the saved data-loading/augmentation recipe, no model or optimizer.
Augmented tensors are reconstructed examples, not archived training screenshots.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, str(PROJECT/'src'))
from pusht.repro import configure_runtime, now, save_json, sha256
configure_runtime(PROJECT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='alt_030_split42_model42_pilot_v1')
    parser.add_argument('--output-name', default='contrastive_training_pairs_v1')
    args = parser.parse_args()
    if not args.output_name or any(s in args.output_name for s in ['/', '\\', '..']):
        raise ValueError('Invalid output name')
    folder = PROJECT/'results/visualizations'
    image_path = folder/f'{args.output_name}.png'
    manifest_path = folder/f'{args.output_name}.json'
    if image_path.exists() or manifest_path.exists():
        raise ValueError('Visualization exists; choose a new --output-name')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from alt.encoder import make_augmentation
    from pusht.data import ObservationFrames, PushTData
    from pusht.experiment import existing_run
    cfg, split, run_dir, _, metadata = existing_run(PROJECT, args.run_id)
    data = PushTData(cfg['paths']['dataset'])
    frames = ObservationFrames(data, split['train_subsets'][str(metadata['scale'])],
                               make_augmentation(cfg['augmentation']))

    class IndexedFrames(Dataset):
        def __len__(self):
            return len(frames)

        def __getitem__(self, index):
            a, b, pos = frames[index]
            return index, a, b, pos

    training = cfg['training']
    assert training['num_workers'] == 0 and training['batch_size'] >= 4
    torch.set_num_threads(training['torch_threads'])
    loader = DataLoader(IndexedFrames(), batch_size=training['batch_size'], shuffle=True,
        generator=torch.Generator().manual_seed(training['shuffle_seed']), drop_last=True, num_workers=0)
    torch.manual_seed(training['augmentation_seed'])
    indices, view_a, view_b, positions = next(iter(loader))
    assert len(indices.unique()) == training['batch_size']
    shown = []
    for batch_index in range(4):
        index = int(indices[batch_index])
        source = int(frames.indices[index])
        ep = int(np.searchsorted(data.ends, source, side='right'))
        assert ep in split['train_subsets'][str(metadata['scale'])]
        np.testing.assert_array_equal(positions[batch_index].numpy(), data.positions[source])
        assert not torch.equal(view_a[batch_index], view_b[batch_index])
        shown.append({'batch_index':batch_index, 'episode_id':ep,
            'local_frame_index':int(source-data.starts[ep]), 'source_index':source,
            'agent_pos':positions[batch_index].tolist(),
            'view1_tensor_sha256':hashlib.sha256(view_a[batch_index].numpy().tobytes()).hexdigest(),
            'view2_tensor_sha256':hashlib.sha256(view_b[batch_index].numpy().tobytes()).hexdigest()})
    font_path = Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts/msyh.ttc'
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(3, 3, figsize=(11, 11))

    def panel(ax, image, title, detail, color):
        ax.imshow(image, interpolation='nearest')
        ax.set_title(title+'\n'+detail, fontsize=10, pad=9, color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_edgecolor(color)

    def detail(item):
        return (f"示范 {item['episode_id']} · 帧 {item['local_frame_index']}\n"
                f"x,y = ({item['agent_pos'][0]:.1f}, {item['agent_pos'][1]:.1f})")

    first = shown[0]
    panel(axes[0, 0], frames.images[int(indices[0])], '原始帧 A（参考）', detail(first), '#555555')
    panel(axes[0, 1], view_a[0].permute(1, 2, 0).numpy(), '锚点：A 的增强视图 1', detail(first), '#14804a')
    panel(axes[0, 2], view_b[0].permute(1, 2, 0).numpy(), '正样本：A 的增强视图 2', detail(first), '#14804a')
    for column, sample in enumerate(shown[1:]):
        i = sample['batch_index']
        letter = ['B', 'C', 'D'][column]
        panel(axes[1, column], view_a[i].permute(1, 2, 0).numpy(),
              f'负样本 {letter} · 增强视图 1', detail(sample), '#b33a3a')
        panel(axes[2, column], view_b[i].permute(1, 2, 0).numpy(),
              f'负样本 {letter} · 增强视图 2', detail(sample), '#b33a3a')
    fig.suptitle('ALT Fusion 对比训练：正样本与负样本\n绿色：同一帧的两个视图，位置不变；红色：同批次其他帧的视图',
                 fontsize=16, y=.995)
    negatives = training['batch_size']*2-2
    fig.text(.5, .018,
        f"真实 batch={training['batch_size']}：每个锚点有 1 个正样本、{negatives} 个负样本；这里只展示 6 个负样本。\n"
        'B 的两个视图彼此仍为正样本，但相对于锚点 A 都是负样本。负样本不代表错误动作。\n'
        '按已保存配置和种子重建；仅做轻微颜色增强，图像之外还输入各自的推杆 x,y。',
        ha='center', fontsize=10, color='#555555')
    fig.tight_layout(rect=(.01, .09, .99, .93), h_pad=2)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=150, facecolor='white')
    plt.close(fig)
    save_json(manifest_path, {'created_at':now(), 'status':'completed', 'source_run_id':args.run_id,
        'source_resolved_config_sha256':sha256(run_dir/'resolved_config.yaml'),
        'split_manifest_sha256':metadata['split_manifest_sha256'],
        'dataset_fingerprint':metadata['dataset_fingerprint'], 'script_sha256':sha256(__file__),
        'batch_size':training['batch_size'], 'shuffle_seed':training['shuffle_seed'],
        'augmentation_seed':training['augmentation_seed'], 'augmentation':cfg['augmentation'],
        'selection':'first four frames in first reconstructed shuffled batch', 'shown_samples':shown,
        'positive_for_anchor_view1':training['batch_size'],
        'negative_count_per_anchor':negatives, 'positive_positions_identical':True,
        'display':'RGB before encoder ImageNet normalization; positions shown in original world units',
        'historical_limit':'Reconstructed with saved recipe/seeds, not an archived original training batch.',
        'no_model_training_or_simulator_execution':True, 'figure_sha256':sha256(image_path)}, immutable=True)
    print(str(image_path))


if __name__ == '__main__':
    main()
