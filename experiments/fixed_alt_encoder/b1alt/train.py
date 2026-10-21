"""Same batching, seeds and optimization as baseline, plus prespecified B1 loss."""
import csv
import json
import time
import torch
from torch.utils.data import DataLoader
from alt.encoder import make_augmentation
from pusht.data import PushTData
from pusht.experiment import event
from pusht.repro import save_json, sha256
from b1alt.data_adapter import PoseFrames
from b1alt.experiment import fresh
from b1alt.loss import losses


def train(cfg, manifest, rd, heavy, meta):
    model = fresh(cfg)
    device = next(model.parameters()).device
    before = model.image_encoder.conv1.weight.detach().cpu().clone()
    data = PushTData(cfg['paths']['dataset'])
    frames = PoseFrames(data, manifest['train_subsets']['30'], make_augmentation(cfg['augmentation']))
    t = cfg['training']
    loader = DataLoader(frames, batch_size=t['batch_size'], shuffle=True,
        generator=torch.Generator().manual_seed(t['shuffle_seed']), drop_last=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=t['learning_rate'])
    torch.manual_seed(t['augmentation_seed'])
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    steps = 0
    fields = ['epoch', 'contrast', 'xy', 'angle', 'pose', 'total', 'optimizer_steps', 'epoch_seconds']
    start = time.perf_counter()
    meta.update(training_status='running', stage='training')
    event(rd, 'training', 'running')
    with (rd/'training_metrics.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, t['epochs']+1):
            epoch_start = time.perf_counter()
            sums = dict.fromkeys(['contrast', 'xy', 'angle', 'pose', 'total'], 0.)
            batches = 0
            for a, b, pos, target in loader:
                a, b, pos, target = [x.to(device) for x in (a, b, pos, target)]
                optimizer.zero_grad(set_to_none=True)
                z1, p1 = model.forward_with_pose(a, pos)
                z2, p2 = model.forward_with_pose(b, pos)
                ls = losses(z1, z2, p1, p2, target, t['temperature'], cfg['b1']['lambda_pose'])
                if not all(torch.isfinite(x) for x in ls.values()):
                    raise ValueError('Nonfinite B1 loss')
                ls['total'].backward()
                if steps == 0:
                    grad = model.image_encoder.conv1.weight.grad
                    if not torch.isfinite(grad).all() or grad.abs().sum() == 0:
                        raise ValueError('Missing finite backbone gradient')
                    meta['first_backbone_gradient_norm'] = float(grad.norm())
                optimizer.step()
                for name in sums:
                    sums[name] += ls[name].item()
                steps += 1
                batches += 1
            row = {'epoch': epoch, **{k: v/batches for k, v in sums.items()},
                   'optimizer_steps': steps, 'epoch_seconds': time.perf_counter()-epoch_start}
            writer.writerow(row)
            f.flush()
            print(json.dumps(row), flush=True)
    elapsed = time.perf_counter()-start
    if torch.equal(before, model.image_encoder.conv1.weight.detach().cpu()):
        raise ValueError('Backbone did not fine-tune')
    path = heavy/'encoder_final.pt'
    if path.exists():
        raise ValueError('Checkpoint exists')
    torch.save({'state_dict': model.state_dict(), 'encoder_config': cfg['encoder'], 'b1': cfg['b1'],
                'split_manifest_sha256': meta['split_manifest_sha256'],
                'train_episode_ids': manifest['train_subsets']['30'],
                'pretrained_sha256': meta['pretrained_sha256'], 'optimizer_steps': steps, 'model_seed': 42}, path)
    meta.update(training_status='completed', stage='trained', optimizer_steps=steps,
        train_frame_count=len(frames), batches_per_epoch=len(loader), training_seconds=elapsed,
        max_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
        checkpoint_sha256=sha256(path), backbone_finetuning_verified=True,
        checkpoint_selection='final_epoch_40', status='running')
    event(rd, 'training', 'completed', optimizer_steps=steps, seconds=elapsed)
