"""Official Runner duck-typed CPU policy: no nn.Module, encoder or optimizer."""
import time
import numpy as np
import torch
from geometry_bank import wrap


class GeometryPolicy:
    device = torch.device('cpu')
    dtype = torch.float32

    def __init__(self, bank, tolerance=.001, sink=None):
        self.bank, self.tolerance, self.sink = bank, tolerance, sink
        self.query_records = []
        self.state_provider = None
        self.reset()

    def reset(self):
        self.previous = None

    def predict_action(self, obs):
        if tuple(obs['image'].shape) != (1, 1, 3, 96, 96) or tuple(obs['agent_pos'].shape) != (1, 1, 2):
            raise ValueError('Unexpected official image observation shape')
        t0 = time.perf_counter()
        snapshot = self.state_provider(obs['agent_pos'][0, -1].cpu().numpy())
        t1 = time.perf_counter()
        state = np.asarray(snapshot['state'], dtype=np.float64)
        i, distance, terms = self.bank.query(state)
        t2 = time.perf_counter()
        chunk, indices = self.bank.action_chunk(i)
        t3 = time.perf_counter()
        previous = self.previous
        same = previous is not None and previous['bank_index'] == i
        pusher_motion = None if previous is None else float(np.linalg.norm(state[:2]-np.asarray(previous['current_state'])[:2]))
        block_motion = None if previous is None else float(np.linalg.norm(state[2:4]-np.asarray(previous['current_state'])[2:4]))
        angle_motion = None if previous is None else float(abs(wrap(state[4]-previous['current_state'][4])))
        matched = self.bank.states[i].astype(float)
        row = {'evaluation_seed': snapshot['seed'], 'current_step': snapshot['step'],
               'current_state': state.tolist(), 'matched_state': matched.tolist(),
               'matched_episode': int(self.bank.episode_id[i]), 'local_frame': int(self.bank.frame_index[i]),
               'source_index': int(self.bank.source_index[i]), 'bank_index': i,
               'geometry_distance': distance, 'distance_components': terms.tolist(),
               'pusher_error': float(np.linalg.norm(state[:2]-matched[:2])),
               'block_error': float(np.linalg.norm(state[2:4]-matched[2:4])),
               'angle_error_degrees': float(abs(wrap(state[4]-matched[4]))*180/np.pi),
               'same_match': bool(same),
               'same_match_and_same_position': bool(same and pusher_motion <= self.tolerance),
               'state_stall': bool(previous is not None and pusher_motion <= self.tolerance and
                                  block_motion <= self.tolerance and angle_motion <= 1e-6),
               'pusher_motion_since_last_query': pusher_motion,
               'block_motion_since_last_query': block_motion,
               'angle_motion_since_last_query': angle_motion,
               'episode_switch': bool(previous and previous['matched_episode'] != int(self.bank.episode_id[i])),
               'boundary_padded': bool(np.any(np.diff(indices) == 0)),
               'action_source_indices': indices.tolist(), 'commands': chunk.tolist(),
               'state_sync_pusher_max_abs_error': snapshot['pusher_max_abs_error'],
               'state_observation_snapshot_equal': True,
               'state_read_ms': (t1-t0)*1000, 'search_ms': (t2-t1)*1000,
               'extraction_ms': (t3-t2)*1000}
        self.query_records.append(row)
        self.previous = row
        return {'action': torch.from_numpy(chunk[None])}

    def finish_query(self, executed):
        row = self.query_records[-1]
        row['executed_actions'] = executed
        if self.sink is not None:
            self.sink.write(__import__('json').dumps(row, allow_nan=False)+'\n')
            self.sink.flush()
