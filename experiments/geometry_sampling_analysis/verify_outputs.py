"""Independent checks, necessary feasibility bounds and documented CDF redraw.

Recomputes primary candidate counts from selected raw states in NumPy blocks;
does not call the candidate relation implementation being checked.
"""
import csv
import itertools
import json
import math
from pathlib import Path
import sys
import numpy as np
import yaml

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from analyze import git, load_data, now, PROJECT, save, sha
from figures import draw_distance_distributions


def all_selected_probability(n, eligible, selected):
    if eligible < selected:
        return 0.0
    return math.prod((eligible-i)/(n-i) for i in range(selected))


def at_least_probability(n, eligible, selected, minimum):
    denominator = math.comb(n,selected)
    return sum(math.comb(eligible,k)*math.comb(n-eligible,selected-k)
               for k in range(minimum,selected+1) if k <= eligible and selected-k <= n-eligible)/denominator


def main():
    config = yaml.safe_load((BASE/'config.yaml').read_text())
    states,ep,local,source,_,audit = load_data(config)
    n = len(states)
    families = {'quantile':BASE,'absolute':BASE/'absolute'}
    statistics = {family:json.loads((root/'results/geometry_statistics.json').read_text()) for family,root in families.items()}
    direct = {family:{pool:{role:np.zeros(n,dtype=np.int64) for role in ['similar','hard_a','hard_b']}
                       for pool in ['all','exclude_local8','cross']} for family in families}
    # Independent block computation; only storage-rounding agrees with main float32 domain.
    for lo in range(0,n,128):
        hi = min(n,lo+128)
        a,b = states[lo:hi].astype(np.float64),states.astype(np.float64)
        dp = np.sqrt(((a[:,None,:2]-b[None,:,:2])**2).sum(-1)).astype(np.float32)
        dt = np.sqrt(((a[:,None,2:4]-b[None,:,2:4])**2).sum(-1)).astype(np.float32)
        delta = np.abs(a[:,None,4]-b[None,:,4]) % (2*np.pi)
        da = np.minimum(delta,2*np.pi-delta).astype(np.float32)
        same = ep[lo:hi,None] == ep[None,:]
        gap = np.abs(local[lo:hi,None]-local[None,:])
        other = np.arange(lo,hi)[:,None] != np.arange(n)[None,:]
        allowed = {'all':other,'exclude_local8':other & ~(same & (gap <= 8)), 'cross':~same}
        for family,stat in statistics.items():
            t = stat['profiles']['primary']['thresholds']
            masks = {'similar':(dp<=t['p_close']) & (dt<=t['T_close']) & (da<=t['theta_close']),
                'hard_a':(dp>=t['p_far']) & (dt<=t['T_close']) & (da<=t['theta_close']),
                'hard_b':(dp<=t['p_close']) & ((dt>=t['T_far']) | (da>=t['theta_far']))}
            for pool,eligible in allowed.items():
                for role,mask in masks.items():
                    direct[family][pool][role][lo:hi] = (mask & eligible).sum(1)
    checks, bounds, corrections = {}, {}, []
    for family,root in families.items():
        stat = statistics[family]
        counts = list(csv.DictReader((root/'results/primary_anchor_candidate_counts.csv').open()))
        assert len(counts) == n
        for pool in direct[family]:
            for role,degrees in direct[family][pool].items():
                stored = np.array([int(r[f'{pool}_{role}_candidates']) for r in counts])
                np.testing.assert_array_equal(degrees,stored)
                summary = stat['profiles']['primary']['roles'][role][pool]
                assert int(degrees.sum()) == summary['ordered_pairs'] == summary['unordered_pairs']*2
                assert int(np.count_nonzero(degrees)) == summary['anchors_with_candidate']
        for item in stat['threshold_sensitivity']:
            assert item['ordered_pairs'] == item['unordered_pairs']*2
            assert item['anchors_with_candidate']+item['anchors_without_candidate'] == n
        batches = json.loads((root/'results/batch_sampling_statistics.json').read_text())
        rows = list(csv.DictReader((root/'results/batch_details.csv').open()))
        for group in batches['groups']:
            subset = [r for r in rows if (r['profile'],r['pool'],r['uniqueness']) ==
                      (group['profile'],group['pool'],group['uniqueness'])]
            assert len(subset) == 100
            assert sum(r['complete']=='True' for r in subset) == group['complete_batches']
            for key,value in group['metrics_all_attempts'].items():
                numbers = [float(r[key]) if key!='complete' else float(r[key]=='True') for r in subset]
                np.testing.assert_allclose(np.mean(numbers),value['mean'],atol=1e-12)
            for r in subset:
                size,distinct,failed = int(r['actual_size']),int(r['distinct_frames']),int(r['failed_slots'])
                assert size == 32-failed
                np.testing.assert_allclose(float(r['duplicate_frame_fraction']),1-distinct/size,atol=1e-12)
                assert int(r['episode_count']) <= min(30,distinct)
                if group['uniqueness'] == 'unique_frames':
                    assert size == distinct
        family_bounds = []
        for profile,pf in stat['profiles'].items():
            for pool in ['all','exclude_local8','cross']:
                roles = {role:pf['roles'][role][pool]['anchors_with_candidate'] for role in ['similar','hard_a','hard_b']}
                probabilities = {'all8_have_similar':all_selected_probability(n,roles['similar'],8),
                    'at_least4_have_hard_a':at_least_probability(n,roles['hard_a'],8,4),
                    'at_least4_have_hard_b':at_least_probability(n,roles['hard_b'],8,4)}
                family_bounds.append({'profile':profile,'pool':pool,'raw_eligible_anchor_counts':roles,
                    **probabilities,'necessary_probability_upper_bound':min(probabilities.values())})
        rng = np.random.default_rng(config['batch']['anchor_seed'])
        anchors = [rng.choice(n,8,replace=False) for _ in range(100)]
        primary_empirical = {}
        for pool in direct[family]:
            degrees = direct[family][pool]
            all_similar = sum(np.all(degrees['similar'][a]>0) for a in anchors)
            quota_possible = 0
            for a in anchors:
                if not np.all(degrees['similar'][a]>0):
                    continue
                for chosen in itertools.combinations(range(8),4):
                    chosen = np.array(chosen)
                    other = np.setdiff1d(np.arange(8),chosen)
                    if np.all(degrees['hard_a'][a[chosen]]>0) and np.all(degrees['hard_b'][a[other]]>0):
                        quota_possible += 1
                        break
            primary_empirical[pool] = {'same100_anchor_sets_all8_have_similar':int(all_similar),
                'same100_anchor_sets_similarity_and_4A4B_assignment_possible_ignoring_candidate_collisions':quota_possible}
        bounds[family] = {'population_hypergeometric_necessary_bounds':family_bounds,
            'primary_empirical_necessary_conditions':primary_empirical,
            'limit':'Necessary conditions only; candidate collisions, greedy selection and random A/B assignment can further reduce completion. Not a proof of optimal assignment feasibility.'}
        figure = root/'figures/distance_distributions.png'
        previous = sha(figure)
        draw_distance_distributions(root,stat['distance_distributions'])
        corrections.append({'family':family,'figure':figure.relative_to(BASE).as_posix(),
            'old_sha256':previous,'corrected_sha256':sha(figure),
            'reason':'Constrain nonnegative distance axes to x>=0; remove automatic negative symlog padding. No statistical data or thresholds changed.'})
        checks[family] = {'status':'passed','anchors_independently_checked':n,
            'primary_candidate_relations_independently_recomputed':9,
            'batch_rows_checked':len(rows),'sensitivity_conditions_checked':len(stat['threshold_sensitivity'])}
    reproduced = {}
    for name in ['geometry_statistics.json','batch_sampling_statistics.json','batch_details.csv','primary_anchor_candidate_counts.csv']:
        a,b = BASE/'results'/name,BASE/'.cache/reproduce_quantile/results'/name
        assert sha(a)==sha(b)
        reproduced[name] = sha(a)
    save(BASE/'results/reproduction_verification.json',{'status':'passed','checked_at':now(),
        'quantile_results_byte_identical_after_explicit_absolute_support':reproduced,
        'independent_real_data_checks':checks,'selected_state_fingerprint':audit['selected_ids_and_states_sha256'],
        'verification_script_sha256':sha(__file__)})
    save(BASE/'results/slot_feasibility_bounds.json',bounds)
    save(BASE/'results/figure_corrections.json',{'checked_at':now(),'corrections':corrections})
    assert git(PROJECT,'diff','--name-only','HEAD','--','.',':(exclude)experiments/geometry_sampling_analysis') == ''
    print(json.dumps({'verification':checks,'primary_bounds':{f:b['primary_empirical_necessary_conditions'] for f,b in bounds.items()}},ensure_ascii=False))


if __name__ == '__main__':
    main()
