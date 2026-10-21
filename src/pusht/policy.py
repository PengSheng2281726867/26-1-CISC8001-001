"""Single-current-frame ALT policy, duck-compatible with official BaseImagePolicy.

Only image and current agent_pos enter the encoder. Telemetry is output-only.
"""
import time
import numpy as np
import torch
from torch import nn


class ALTPolicy(nn.Module):
    def __init__(self, encoder, bank, data, evaluation_seeds=None):
        super().__init__()
        self.encoder = encoder.eval()
        self.bank, self.data = bank, data
        self.bank.embeddings = self.bank.embeddings.to(self.device)
        self.evaluation_seeds = evaluation_seeds
        self.query_records = []
        self.episode_index = -1
        self.current_step = 0
        self.previous_match = None
        self.invalid_events = 0
        if not np.array_equal(data.starts[bank.episode_ids] + bank.local_frames, bank.source_indices):
            raise ValueError('Bank episode/frame/source mapping mismatch')

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    @property
    def dtype(self):
        return next(self.encoder.parameters()).dtype

    def reset(self):
        self.current_step = 0
        self.previous_match = None
        self.episode_index += 1

    def _clock(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    @torch.inference_mode()
    def predict_action(self, obs_dict):
        if set(obs_dict) != {'image', 'agent_pos'}:
            raise ValueError('ALT accepts only current image + agent_pos')
        image, pos = obs_dict['image'], obs_dict['agent_pos']
        if image.ndim != 5 or pos.ndim != 3 or image.shape[:2] != pos.shape[:2]:
            raise ValueError('Runner observations must be [B,T,3,96,96] and [B,T,2]')
        image, pos = image[:, -1].to(self.device, self.dtype), pos[:, -1].to(self.device, self.dtype)
        if (not torch.isfinite(image).all() or not torch.isfinite(pos).all()
                or image.min() < 0 or image.max() > 1):
            self.invalid_events += 1
            raise ValueError('Nonfinite or mis-scaled current observation')
        start = self._clock()
        embedding = self.encoder(image, pos)
        after_encoder = self._clock()
        matches, similarity = self.bank.query(embedding)
        after_search = self._clock()
        chunks, rows = [], []
        for b, match in enumerate(matches.cpu().tolist()):
            ep, frame = int(self.bank.episode_ids[match]), int(self.bank.local_frames[match])
            chunk, indices = self.data.action_chunk(ep, frame, 8)
            if not torch.isfinite(similarity[b]):
                self.invalid_events += 1
                raise ValueError('Nonfinite cosine similarity')
            previous = self.previous_match if len(matches) == 1 else None
            rows.append({'episode_index': self.episode_index, 'batch_index': b,
                         'evaluation_seed': (self.evaluation_seeds[self.episode_index]
                            if self.evaluation_seeds is not None else None),
                         'current_step': self.current_step, 'matched_episode_id': ep,
                         'local_frame_index': frame, 'source_index': int(self.bank.source_indices[match]),
                         'top1_cosine': float(similarity[b].cpu()),
                         'agent_pos': pos[b].cpu().tolist(),
                         'matched_agent_pos': self.data.positions[self.bank.source_indices[match]].tolist(),
                         'action_source_indices': indices.tolist(), 'commands': chunk.tolist(),
                         'episode_switch': bool(previous is not None and previous[0] != ep),
                         'frame_delta': (frame-previous[1] if previous is not None and previous[0] == ep else None),
                         'boundary_padded': bool(indices[-1] == indices[-2]),
                         'invalid': False})
            chunks.append(chunk)
        actions = torch.from_numpy(np.stack(chunks)).to(self.device)
        end = self._clock()
        for row in rows:
            row.update(encoder_ms=(after_encoder-start)*1000,
                       search_ms=(after_search-after_encoder)*1000,
                       extraction_ms=(end-after_search)*1000)
        self.query_records.extend(rows)
        if len(matches) == 1:
            self.previous_match = (rows[0]['matched_episode_id'], rows[0]['local_frame_index'])
        self.current_step += 8
        return {'action': actions}
