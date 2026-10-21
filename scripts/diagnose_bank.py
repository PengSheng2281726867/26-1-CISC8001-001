"""Re-encode stored frames and audit trained-bank self retrieval and action alignment."""
import argparse
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, str(PROJECT/'src'))
from pusht.repro import configure_runtime
configure_runtime(PROJECT)


def main():
    import numpy as np
    import torch
    from alt.memory import MemoryBank
    from pusht.data import ObservationFrames, PushTData
    from pusht.experiment import existing_run, load_model
    from pusht.repro import now, save_json, sha256
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    cfg, manifest, run_dir, artifacts, metadata = existing_run(PROJECT, args.run_id)
    bank = MemoryBank.load(artifacts/'memory_bank.pt')
    if sha256(artifacts/'memory_bank.pt') != metadata['bank_sha256']:
        raise ValueError('Bank checksum mismatch')
    model = load_model(cfg, artifacts, metadata)
    device = next(model.parameters()).device
    bank.embeddings = bank.embeddings.to(device)
    data = PushTData(cfg['paths']['dataset'])
    episodes = manifest['train_subsets'][str(metadata['scale'])]
    frames = ObservationFrames(data, episodes)
    np.testing.assert_array_equal(frames.indices, bank.source_indices)
    exact = equivalent = 0
    min_similarity = 1.0
    with torch.inference_mode():
        for start in range(0, len(bank.embeddings), 128):
            expected = torch.arange(start, min(start+128, len(bank.embeddings)), device=device)
            idx, similarity = bank.query(bank.embeddings[expected])
            exact += int((idx == expected).sum())
            equivalent += int(((bank.embeddings[idx]-bank.embeddings[expected]).norm(dim=-1) < 1e-3).sum())
            min_similarity = min(min_similarity, float(similarity.min()))
        offsets = []
        for ep in episodes:
            local = np.flatnonzero(bank.episode_ids == ep)
            offsets.extend([int(local[0]), int(local[len(local)//2]), int(local[-1])])
        examples = []
        for start in range(0, len(offsets), 64):
            selected = offsets[start:start+64]
            samples = [frames[i] for i in selected]
            images = torch.stack([s[0] for s in samples]).to(device)
            positions = torch.stack([s[1] for s in samples]).to(device)
            embedding = model(images, positions)
            matches, similarity = bank.query(embedding)
            for j, original in enumerate(selected):
                match = int(matches[j])
                distance = float((embedding[j]-bank.embeddings[original]).norm())
                if distance > 1e-4 or float(similarity[j]) < .99999:
                    raise ValueError('Raw-frame re-encoding disagrees with deterministic bank')
                examples.append({'original_bank_index': original, 'matched_bank_index': match,
                                 'cosine': float(similarity[j]), 'reencoding_l2_error': distance,
                                 'exact_index_recovered': match == original})
    # Every stored frame's chunk begins at action_t and stays inside its episode.
    padded = 0
    for ep, local, source in zip(bank.episode_ids, bank.local_frames, bank.source_indices):
        chunk, indices = data.action_chunk(ep, local)
        np.testing.assert_array_equal(chunk[0], data.actions[source])
        assert indices[0] == source and indices[-1] < data.ends[ep]
        assert chunk.shape == (8, 2) and np.isfinite(chunk).all()
        padded += int(indices[-1] == indices[-2])
    result = {'checked_at': now(), 'status': 'passed', 'run_id': args.run_id,
              'bank_entries': len(bank.embeddings), 'all_embedding_self_queries': len(bank.embeddings),
              'exact_self_index_count': exact, 'equivalent_embedding_count': equivalent,
              'min_self_cosine': min_similarity, 'raw_frame_reencoding_samples': len(examples),
              'all_action_chunks_checked': len(bank.embeddings), 'padded_frame_count': padded,
              'examples': examples,
              'tie_explanation': 'Top1 may return an equivalent duplicate or a floating-point tie. First bank index wins; re-encoding must still match the original embedding.',
              'checkpoint_sha256': metadata['checkpoint_sha256'], 'bank_sha256': metadata['bank_sha256'],
              'diagnostic_script_sha256': sha256(__file__)}
    if min_similarity < .99999 or equivalent != len(bank.embeddings):
        raise ValueError('Stored embedding self retrieval failed')
    save_json(run_dir/'bank_validation.json', result, immutable=True)
    print({k: v for k, v in result.items() if k not in ['examples', 'tie_explanation']})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        from pusht.repro import now
        with (PROJECT/'docs/failure_log.md').open('a', encoding='utf-8') as f:
            f.write('\n## '+now()+' bank diagnostic failure\n\nCommand: `'+
                    ' '.join(sys.argv)+'`\n\n```text\n'+traceback.format_exc()+'```\n')
        raise
