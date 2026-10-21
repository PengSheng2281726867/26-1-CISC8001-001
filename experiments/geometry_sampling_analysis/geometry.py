"""Geometry and sampling primitives; NumPy only, no learning or environment code."""
import hashlib
import itertools
import numpy as np


def wrapped_angle(a, b):
    return np.abs((np.asarray(a)-np.asarray(b)+np.pi) % (2*np.pi)-np.pi)


def distances(states, xy_scale=512.0, angle_scale=np.pi, block_rows=256):
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 5 or not np.isfinite(states).all():
        raise ValueError('Expected finite [N,5] state')
    n = len(states)
    result = {key: np.empty((n, n), dtype=np.float32) for key in ['d_p', 'd_T', 'd_theta', 'D']}
    for lo in range(0, n, block_rows):
        hi = min(n, lo+block_rows)
        p = np.linalg.norm(states[lo:hi, None, :2]-states[None, :, :2], axis=-1)
        t = np.linalg.norm(states[lo:hi, None, 2:4]-states[None, :, 2:4], axis=-1)
        a = wrapped_angle(states[lo:hi, None, 4], states[None, :, 4])
        for key, value in [('d_p', p), ('d_T', t), ('d_theta', a),
                           ('D', (p/xy_scale)**2+(t/xy_scale)**2+(a/angle_scale)**2)]:
            result[key][lo:hi] = value
    return result


def pools_and_time(episodes, local_frames, radius=8):
    episodes, local_frames = np.asarray(episodes), np.asarray(local_frames)
    same = episodes[:, None] == episodes[None, :]
    gap = np.abs(local_frames[:, None]-local_frames[None, :])
    other = ~np.eye(len(episodes), dtype=bool)
    pools = {'all': other, 'exclude_local8': other & ~(same & (gap <= radius)), 'cross': ~same}
    temporal = {'same_1_2': same & (gap >= 1) & (gap <= 2),
                'same_3_8': same & (gap >= 3) & (gap <= radius),
                'same_9_plus': same & (gap > radius), 'cross': ~same}
    if not np.array_equal(np.logical_or.reduce(list(temporal.values())), other):
        raise ValueError('Temporal strata do not partition distinct frames')
    assert sum(int(mask.sum()) for mask in temporal.values()) == int(other.sum())
    return pools, temporal


def summary(values):
    x = np.asarray(values)
    if not x.size:
        return {'count': 0, 'mean': None, 'std': None, 'min': None, 'max': None, 'quantiles': {}}
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite statistics')
    levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {'count': int(x.size), 'mean': float(np.mean(x, dtype=np.float64)),
            'std': float(np.std(x, dtype=np.float64)), 'min': float(x.min()), 'max': float(x.max()),
            'quantiles': {f'p{q:02d}': float(v) for q, v in zip(levels, np.percentile(x, levels))}}


def relation(d, kind, threshold):
    if kind == 'similar':
        mask = ((d['d_p'] <= threshold['p_close']) & (d['d_T'] <= threshold['T_close'])
                & (d['d_theta'] <= threshold['theta_close']))
    elif kind == 'hard_a':
        mask = ((d['d_T'] <= threshold['T_close']) & (d['d_theta'] <= threshold['theta_close'])
                & (d['d_p'] >= threshold['p_far']))
    elif kind == 'hard_b':
        mask = ((d['d_p'] <= threshold['p_close'])
                & ((d['d_T'] >= threshold['T_far']) | (d['d_theta'] >= threshold['theta_far'])))
    else:
        raise ValueError(kind)
    np.fill_diagonal(mask, False)
    return mask


def calibration(d, cross, near_levels, far_levels):
    tri = np.triu(cross, k=1)
    result = {'population': 'cross-episode unordered distinct training-frame pairs',
              'unordered_pair_count': int(tri.sum()), 'near': {}, 'far': {}}
    for key in ['d_p', 'd_T', 'd_theta']:
        x = d[key][tri]
        result['near'][key] = {str(q): float(np.percentile(x, q)) for q in near_levels}
        result['far'][key] = {str(q): float(np.percentile(x, q)) for q in far_levels}
    return result


def profile_thresholds(cal, near, far):
    return {'p_close': cal['near']['d_p'][str(near)], 'T_close': cal['near']['d_T'][str(near)],
            'theta_close': cal['near']['d_theta'][str(near)], 'p_far': cal['far']['d_p'][str(far)],
            'T_far': cal['far']['d_T'][str(far)], 'theta_far': cal['far']['d_theta'][str(far)]}


def resolve_profile(cal, values):
    return profile_thresholds(cal,values.get('near_level',values.get('near_percentile')),
                              values.get('far_level',values.get('far_percentile')))


def resolve_calibration(d, cross, settings):
    if 'absolute_values' not in settings:
        return calibration(d,cross,settings['near_percentiles'],settings['far_percentiles'])
    result = {'population':'fixed absolute physical sensitivity grid; no state-label fitting',
              'selection_reason':'Supplement declared after component distribution inspection, before reviewing candidate/simulation statistics.',
              'near':{},'far':{}}
    for side in ['near','far']:
        for key, values in settings['absolute_values'][side].items():
            if key == 'd_theta_degrees':
                result[side]['d_theta'] = {name:float(np.deg2rad(v)) for name,v in values.items()}
            else:
                result[side][key] = {name:float(v) for name,v in values.items()}
    return result


def threshold_grid(cal, near_levels, far_levels):
    for p, t, a in itertools.product(near_levels, repeat=3):
        yield ('similar', f'similar_p{p}_T{t}_a{a}',
               {'p_close': cal['near']['d_p'][str(p)], 'T_close': cal['near']['d_T'][str(t)],
                'theta_close': cal['near']['d_theta'][str(a)]})
    for t, a, p in itertools.product(near_levels, near_levels, far_levels):
        yield ('hard_a', f'hard_a_T{t}_a{a}_p{p}',
               {'T_close': cal['near']['d_T'][str(t)], 'theta_close': cal['near']['d_theta'][str(a)],
                'p_far': cal['far']['d_p'][str(p)]})
    for p, t, a in itertools.product(near_levels, far_levels, far_levels):
        yield ('hard_b', f'hard_b_p{p}_T{t}_a{a}',
               {'p_close': cal['near']['d_p'][str(p)], 'T_far': cal['far']['d_T'][str(t)],
                'theta_far': cal['far']['d_theta'][str(a)]})


def candidate_summary(mask, episodes, episode_ids):
    degree = mask.sum(axis=1)
    endpoint = np.array([degree[episodes == ep].sum() for ep in episode_ids], dtype=np.int64)
    count = int(degree.sum())
    assert count % 2 == 0
    shares = endpoint/count if count else np.zeros(len(episode_ids))
    sizes = np.array([np.count_nonzero(episodes == ep) for ep in episode_ids])
    return {'ordered_pairs': count, 'unordered_pairs': count//2,
            'anchors_with_candidate': int(np.count_nonzero(degree)),
            'anchors_without_candidate': int(np.count_nonzero(degree == 0)),
            'anchor_coverage_fraction': float(np.mean(degree > 0)),
            'anchor_failure_fraction': float(np.mean(degree == 0)),
            'anchor_fraction_at_least_4_candidates': float(np.mean(degree >= 4)),
            'anchor_fraction_at_least_8_candidates': float(np.mean(degree >= 8)),
            'candidates_per_anchor': summary(degree),
            'active_episode_count': int(np.count_nonzero(endpoint)),
            'largest_episode_endpoint_share': float(shares.max()),
            'top3_episode_endpoint_share': float(np.sort(shares)[-3:].sum()),
            'effective_episode_count': float(1/(shares@shares)) if count else 0.0,
            'max_endpoint_share_over_frame_share': float(np.max(shares/(sizes/len(episodes)))) if count else 0.0,
            'episode_endpoint_counts': endpoint.tolist(),
            'episode_anchors_with_candidate': [int(np.count_nonzero(degree[episodes == ep])) for ep in episode_ids]}


def nearest(d, allowed, k, block_rows=128):
    n = len(allowed)
    if np.min(allowed.sum(1)) < k:
        raise ValueError(f'At least {k} eligible neighbors required; do not pad/repeat neighbors')
    result = np.empty((n, k), dtype=np.int32)
    for lo in range(0, n, block_rows):
        hi = min(n, lo+block_rows)
        cost = np.where(allowed[lo:hi], d['D'][lo:hi], np.inf)
        result[lo:hi] = np.argsort(cost, axis=1, kind='stable')[:, :k]
    assert np.all(allowed[np.arange(n)[:, None], result])
    return result


def stable_seed(base, name):
    return (base+int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little')) % (2**63-1)


def sample_structured(anchors, hard_a_assignment, candidates, n, rng, unique, random_count=8):
    selected = list(map(int, anchors))
    roles = ['anchor']*len(anchors)
    role_pairs, failures = [], []
    for kind in ['similar', 'hard']:
        for i, anchor in enumerate(anchors):
            role = 'similar' if kind == 'similar' else ('hard_a' if hard_a_assignment[i] else 'hard_b')
            raw = candidates[role][int(anchor)]
            available = np.array([j for j in raw if not unique or int(j) not in selected], dtype=np.int32)
            if not len(available):
                failures.append({'anchor':int(anchor), 'role':role,
                                 'reason':'no_candidate' if not len(raw) else 'unique_constraint_exhausted'})
                continue
            match = int(rng.choice(available))
            selected.append(match)
            roles.append(role)
            role_pairs.append({'anchor':int(anchor), 'candidate':match, 'role':role})
    pool = np.array([i for i in range(n) if not unique or i not in selected], dtype=np.int32)
    if len(pool) < random_count:
        raise ValueError('Insufficient random pool; do not silently sample with replacement')
    selected.extend(rng.choice(pool, random_count, replace=False).tolist())
    roles.extend(['random']*random_count)
    return {'indices':selected, 'roles':roles, 'role_pairs':role_pairs, 'failures':failures}


def batch_metrics(batch, masks, episodes, expected_size=32):
    indices = np.asarray(batch['indices'], dtype=np.int32)
    distinct = np.unique(indices)
    counts = {}
    for name, mask in masks.items():
        counts[name+'_slot_pairs'] = int(np.triu(mask[np.ix_(indices, indices)], 1).sum())
        counts[name+'_unique_pairs'] = int(np.triu(mask[np.ix_(distinct, distinct)], 1).sum())
    ep, ep_counts = np.unique(episodes[indices], return_counts=True)
    return {**counts, 'actual_size':len(indices), 'complete':len(indices) == expected_size,
            'distinct_frames':len(distinct), 'duplicate_frame_fraction':1-len(distinct)/len(indices),
            'episode_count':len(ep), 'largest_episode_slot_share':float(ep_counts.max()/len(indices)),
            'failed_slots':len(batch['failures']),
            **{role+'_failed':sum(f['role'] == role for f in batch['failures']) for role in ['similar','hard_a','hard_b']},
            'empty_candidate_failures':sum(f['reason'] == 'no_candidate' for f in batch['failures']),
            'uniqueness_failures':sum(f['reason'] == 'unique_constraint_exhausted' for f in batch['failures'])}
