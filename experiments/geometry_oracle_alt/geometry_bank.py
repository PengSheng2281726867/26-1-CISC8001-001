"""Independent geometric retrieval, reusing the tested project action_chunk.

Raw float32 bank states are kept byte-exact. Only distance arithmetic is float64.
No images, actions or future states enter the distance function.
"""
import numpy as np
from pusht.data import PushTData


def wrap(x):
    return (np.asarray(x) + np.pi) % (2*np.pi) - np.pi


def components(query, states, position_scale=512.):
    delta = np.asarray(states, dtype=np.float64) - np.asarray(query, dtype=np.float64)
    return np.stack((np.sum(delta[..., :2]**2, axis=-1)/position_scale**2,
                     np.sum(delta[..., 2:4]**2, axis=-1)/position_scale**2,
                     (wrap(delta[..., 4])/np.pi)**2), axis=-1)


class GeometryBank:
    def __init__(self, path, episodes, weights=(1., 1., 1.), position_scale=512.):
        self.data = PushTData(path)  # read-only Zarr, original action_chunk(offset=0)
        self.episodes = np.asarray(episodes, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=np.float64)
        if self.weights.shape != (3,) or not np.isfinite(self.weights).all() or np.any(self.weights < 0):
            raise ValueError('Invalid geometry weights')
        self.position_scale = float(position_scale)
        self.source_index = np.concatenate([np.arange(self.data.starts[e], self.data.ends[e]) for e in episodes])
        self.episode_id = np.concatenate([np.full(self.data.ends[e]-self.data.starts[e], e) for e in episodes])
        self.frame_index = np.concatenate([np.arange(self.data.ends[e]-self.data.starts[e]) for e in episodes])
        self.states = np.concatenate([self.data.root['data/state'][self.data.starts[e]:self.data.ends[e]] for e in episodes])
        if self.states.shape != (len(self.source_index), 5) or not np.isfinite(self.states).all():
            raise ValueError('Expected finite [pusher_xy, block_xy, block_theta]')

    def query(self, state):
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (5,) or not np.isfinite(state).all():
            raise ValueError('Expected finite current 5D state')
        terms = components(state, self.states, self.position_scale)
        distances = terms @ self.weights
        index = int(np.argmin(distances))  # deterministic first bank index for exact ties
        return index, float(distances[index]), terms[index]

    def action_chunk(self, index):
        return self.data.action_chunk(self.episode_id[index], self.frame_index[index], 8)

    def save(self, path):
        np.savez_compressed(path, state=self.states, episode_id=self.episode_id,
                            frame_index=self.frame_index, source_index=self.source_index,
                            action_index=self.source_index)
