"""ALT-30 geometry feasibility analysis. No network, policy or simulator imports."""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[1]
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLCONFIGDIR'] = str(BASE/'.cache/matplotlib')
import numpy as np
import yaml
import zarr
from geometry import (batch_metrics, candidate_summary, distances, nearest,
    pools_and_time, resolve_calibration, resolve_profile, relation, sample_structured, stable_seed, summary, threshold_grid)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git(path, *args):
    result = subprocess.run(['git','-c',f'safe.directory={Path(path).as_posix()}',
        '-c',f'core.excludesFile={(PROJECT/".gitignore").as_posix()}', '-C',str(path),*args],
        capture_output=True, text=True, encoding='utf-8', env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'})
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def source_record(path):
    return {'path':str(path), 'commit':git(path,'rev-parse','HEAD'),
            'dirty_status':git(path,'status','--porcelain=v1'),
            'tracked_diff_sha256':hashlib.sha256(git(path,'diff','HEAD').encode()).hexdigest()}


def read_selected(state_array, ends, episode_ids):
    """Logical slices are strictly restricted to the frozen train30 episodes."""
    starts = np.r_[0, ends[:-1]]
    arrays, episodes, local, source, reads = [], [], [], [], []
    for ep in episode_ids:
        lo, hi = int(starts[ep]), int(ends[ep])
        block = state_array[lo:hi, :]
        arrays.append(block)
        episodes.extend([ep]*(hi-lo))
        local.extend(range(hi-lo))
        source.extend(range(lo, hi))
        reads.append({'episode_id':int(ep), 'start_inclusive':lo, 'end_exclusive':hi, 'frames':hi-lo})
    return np.concatenate(arrays), np.array(episodes), np.array(local), np.array(source), reads


def load_data(config):
    project_cfg_path = PROJECT/config['project_config']
    project_cfg = yaml.safe_load(project_cfg_path.read_text(encoding='utf-8'))
    manifest_path = PROJECT/config['split_manifest']
    split = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = manifest_path.with_suffix('.sha256').read_text().strip().split()[0]
    if sha(manifest_path) != expected:
        raise ValueError('Frozen split byte hash mismatch')
    ids = split['train_subsets'][config['subset']]
    if len(ids) != 30 or len(set(ids)) != 30 or set(ids) & set(split['heldout_validation_episode_ids']):
        raise ValueError('Require original disjoint train30 episode IDs')
    dataset = Path(project_cfg['paths']['dataset'])
    if not dataset.is_absolute():
        dataset = PROJECT/dataset
    root = zarr.open(str(dataset), mode='r')
    ends = root['meta/episode_ends'][:]
    if not np.array_equal(ends, split['episode_ends']) or np.any(np.diff(np.r_[0,ends]) <= 0):
        raise ValueError('Episode metadata changed')
    state_array = root['data/state']
    if state_array.shape != (int(ends[-1]), 5):
        raise ValueError('Unexpected state schema')
    state, ep, local, source, reads = read_selected(state_array, ends, ids)
    if len(state) != config['expected_frames'] or not np.isfinite(state).all():
        raise ValueError('Unexpected selected frame count or invalid state')
    dp = Path(project_cfg['paths']['dp_source'])
    alt = Path(project_cfg['paths']['alt_source'])
    env_file = dp/'diffusion_policy/env/pusht/pusht_env.py'
    collector_file = dp/'demo_pusht.py'
    env, collector = [p.read_text(encoding='utf-8') for p in [env_file, collector_file]]
    compact = lambda text: re.sub(r'\s+', '', text)
    if "np.concatenate([info['pos_agent'],info['block_pose']])" not in compact(collector):
        raise ValueError('Collector state convention requires inspection')
    if "'block_pose':np.array(list(self.block.position)+[self.block.angle])" not in compact(env):
        raise ValueError('Environment block pose convention requires inspection')
    if 'self.block.angle%(2*np.pi)' not in compact(env):
        raise ValueError('Radian periodicity evidence missing')
    h = hashlib.sha256()
    for x in [ep.astype('<i8'), local.astype('<i8'), source.astype('<i8'), state.astype('<f4')]:
        h.update(x.tobytes())
    meta_files = sorted({p for pattern in ['.zarray','.zattrs','.zgroup'] for p in dataset.rglob(pattern)})
    mh = hashlib.sha256()
    for p in meta_files:
        mh.update((p.relative_to(dataset).as_posix()+'\0'+sha(p)+'\n').encode())
    audit = {'dataset_path':str(dataset), 'state_shape':list(state_array.shape), 'state_dtype':str(state_array.dtype),
        'state_columns':['pusher_x','pusher_y','block_x','block_y','block_theta'],
        'angle_unit':'radian; periodic distance in [0,pi]',
        'angle_evidence':{'collector':str(collector_file),'collector_sha256':sha(collector_file),
            'environment':str(env_file),'environment_sha256':sha(env_file),
            'interpretation':'Collector concatenates pos_agent and block_pose; block_pose uses block.angle. Environment wraps block.angle by 2*pi and bounds its angular observation by 2*pi.'},
        'episode_ids':ids, 'frame_count':len(state), 'episode_state_reads':reads,
        'validation_or_test_states_read_as_candidates':False,
        'selected_state_min':state.min(0).tolist(), 'selected_state_max':state.max(0).tolist(),
        'selected_ids_and_states_sha256':h.hexdigest(), 'dataset_metadata_sha256':mh.hexdigest(),
        'frozen_full_dataset_fingerprint_reference':split['dataset_fingerprint'],
        'full_dataset_bytes_rehashed':False,
        'integrity_note':'Only metadata and selected train30 state slices are accessed; no all-state adapter, action array or heldout-state scan.',
        'split_sha256':expected, 'project_config_sha256':sha(project_cfg_path),
        'source_provenance':{'alt':source_record(alt), 'dp':source_record(dp)}}
    return state, ep, local, source, root, audit


def distributions(d, pools, temporal, bins):
    result = {}
    for name, mask in {'all':pools['all'], **temporal}.items():
        upper = np.triu(mask, 1)
        row = {'unordered_pairs':int(upper.sum())}
        for key, matrix in d.items():
            values = matrix[upper]
            edges = (np.linspace(0, np.pi, bins+1) if key == 'd_theta' else
                     np.r_[0,np.geomspace(1e-6,5,bins)] if key == 'D' else
                     np.linspace(0,np.sqrt(2)*512,bins+1))
            hist, edges = np.histogram(values, edges)
            assert int(hist.sum()) == len(values)
            row[key] = {**summary(values), 'histogram_counts':hist.tolist(), 'histogram_edges':edges.tolist()}
        result[name] = row
    assert sum(result[k]['unordered_pairs'] for k in temporal) == result['all']['unordered_pairs']
    return result


def neighbor_statistics(d, pools, temporal, episodes, config):
    result = {}
    for name, allowed in pools.items():
        neighbors = nearest(d, allowed, max(config['top_k']))
        by_k = {}
        for k in config['top_k']:
            cols = neighbors[:, :k]
            rows = np.arange(len(episodes))[:,None]
            cross = episodes[rows] != episodes[cols]
            by_k[str(k)] = {'directed_neighbor_relations':int(cols.size),
                'cross_episode_fraction':float(cross.mean()),
                'anchors_with_any_cross_neighbor_fraction':float(np.mean(cross.any(1))),
                'anchors_with_only_same_episode_neighbors_fraction':float(np.mean(~cross.any(1))),
                'temporal_relation_fractions':{key:float(mask[rows,cols].mean()) for key,mask in temporal.items()},
                'distance_components':{key:summary(matrix[rows,cols]) for key,matrix in d.items()},
                'distinct_cross_episode_count_per_anchor':summary([
                    len(set(episodes[c].tolist())-{int(episodes[i])}) for i,c in enumerate(cols)])}
        result[name] = {'eligible_candidates_per_anchor':summary(allowed.sum(1)), 'top_k':by_k}
    return result


def sensitivity(d, pools, episodes, ids, cal, settings):
    records = []
    for kind, name, threshold in threshold_grid(cal,settings.get('near_levels',settings.get('near_percentiles')),
                                               settings.get('far_levels',settings.get('far_percentiles'))):
        base = relation(d, kind, threshold)
        for pool, eligible in pools.items():
            records.append({'kind':kind, 'condition_id':name, 'pool':pool, 'thresholds':threshold,
                            **candidate_summary(base & eligible, episodes, ids)})
    return records


def simulation(d, pools, episodes, cal, config):
    settings = config['batch']
    if (settings['size'],settings['anchors'],settings['similar'],settings['hard_a'],settings['hard_b'],settings['random']) != (32,8,8,4,4,8):
        raise ValueError('This analysis explicitly evaluates 8+8+4A+4B+8; change implementation before another allocation')
    n, repeats = len(episodes), settings['repeats']
    if repeats < 100:
        raise ValueError('At least100 simulated batches are required')
    rng = np.random.default_rng(settings['anchor_seed'])
    anchors = [rng.choice(n,8,replace=False) for _ in range(repeats)]
    assignments = [rng.permutation([True]*4+[False]*4) for _ in range(repeats)]
    random_rng = np.random.default_rng(settings['baseline_seed'])
    baselines = [np.r_[a,random_rng.choice(np.setdiff1d(np.arange(n),a),24,replace=False)].tolist() for a in anchors]
    groups, all_rows, examples = [], [], []
    primary_degrees = {i:{'row_index':i, 'episode_id':int(episodes[i])} for i in range(n)}
    profiles = config['thresholds']['profiles']
    for profile, values in profiles.items():
        threshold = resolve_profile(cal,values)
        raw = {role:relation(d,role,threshold) for role in ['similar','hard_a','hard_b']}
        for pool, eligible in pools.items():
            roles = {role:mask & eligible for role,mask in raw.items()}
            if profile == 'primary':
                for role,mask in roles.items():
                    for i, degree in enumerate(mask.sum(1)):
                        primary_degrees[i][pool+'_'+role+'_candidates'] = int(degree)
            candidates = {role:[np.flatnonzero(row).astype(np.int32) for row in mask] for role,mask in roles.items()}
            metrics_masks = {**{role+'_eligible':mask for role,mask in roles.items()}, 'similar_geometric':raw['similar']}
            for uniqueness in settings['uniqueness']+['random_baseline']:
                key = profile+'/'+pool+'/'+uniqueness
                choices = np.random.default_rng(stable_seed(settings['candidate_seed'],key))
                rows, failed_anchor_ids = [], []
                for i in range(repeats):
                    if uniqueness == 'random_baseline':
                        batch = {'indices':baselines[i], 'roles':['random']*32, 'role_pairs':[], 'failures':[]}
                    else:
                        batch = sample_structured(anchors[i],assignments[i],candidates,n,choices,
                                                  uniqueness == 'unique_frames',settings['random'])
                    for pair in batch['role_pairs']:
                        assert roles[pair['role']][pair['anchor'],pair['candidate']]
                    measurements = batch_metrics(batch,metrics_masks,episodes,settings['size'])
                    record = {'profile':profile, 'pool':pool, 'uniqueness':uniqueness, 'batch_index':i, **measurements}
                    rows.append(record)
                    all_rows.append(record)
                    failed_anchor_ids.extend(f['anchor'] for f in batch['failures'])
                    if i == 0:
                        examples.append({'group':key, 'batch_index':i, **batch})
                numerical = list(measurements)
                completed = [r for r in rows if r['complete']]
                groups.append({'group':key, 'profile':profile, 'pool':pool, 'uniqueness':uniqueness,
                    'thresholds':threshold, 'attempts':repeats, 'complete_batches':len(completed),
                    'completion_fraction':len(completed)/repeats,
                    'metrics_all_attempts':{key:summary([float(r[key]) for r in rows]) for key in numerical},
                    'metrics_completed_only':{key:summary([float(r[key]) for r in completed]) for key in numerical},
                    'failed_anchor_occurrences_by_episode':{str(ep):int(np.sum(episodes[failed_anchor_ids] == ep))
                                                           for ep in np.unique(episodes)} if failed_anchor_ids else {}})
    return {'groups':groups, 'selection_policy':settings, 'profiles':profiles,
            'random_baseline_note':'The same100 uniformly sampled32-frame batches are scored under9 threshold/pool combinations; first8 are the shared anchors. This is not900 independent random batches.',
            'unique_structured_batches_generated':repeats*len(profiles)*len(pools)*len(settings['uniqueness']),
            'unique_random_batches_generated':repeats,
            'partial_batch_note':'Failed roles remain empty. All-attempt pair counts include partial batches; completed-only summaries are provided separately. No threshold/anchor/backfill fallback.'}, all_rows, examples, list(primary_degrees.values())


def validate_real(d, state, episodes, local, pools, config):
    rng = np.random.default_rng(8001)
    max_error = {k:0.0 for k in d}
    for i,j in rng.integers(0,len(state),(512,2)):
        p = np.linalg.norm(state[i,:2].astype(float)-state[j,:2])
        t = np.linalg.norm(state[i,2:4].astype(float)-state[j,2:4])
        a = abs((float(state[i,4])-float(state[j,4])+np.pi)%(2*np.pi)-np.pi)
        actual = {'d_p':p,'d_T':t,'d_theta':a,'D':(p/512)**2+(t/512)**2+(a/np.pi)**2}
        for key,value in actual.items():
            max_error[key] = max(max_error[key],abs(float(d[key][i,j])-value))
            np.testing.assert_allclose(d[key][i,j],value,rtol=2e-6,atol=1e-6)
    assert all(not np.diag(mask).any() for mask in pools.values())
    assert not (pools['cross'] & (episodes[:,None] == episodes[None,:])).any()
    for name,pool in pools.items():
        top = nearest(d,pool,20)
        for i in rng.choice(len(state),8,replace=False):
            cost = np.where(pool[i],d['D'][i],np.inf)
            np.testing.assert_array_equal(top[i],np.argsort(cost,kind='stable')[:20])
    return {'status':'passed','independent_scalar_distance_pairs':512,'max_float32_storage_errors':max_error,
            'independent_top20_rows_per_pool':8,'self_exclusion_and_cross_episode_checks':'passed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(BASE/'config.yaml'))
    parser.add_argument('--output-root', default='.')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    output = (BASE/args.output_root).resolve()
    if output != BASE and BASE not in output.parents:
        raise ValueError('All outputs must remain in the isolated analysis directory')
    if (output/'results/geometry_statistics.json').exists():
        raise ValueError('Preserve completed results; choose a new --output-root for reproduction')
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if config['neighbors']['exclude_temporal_radius'] != 8 or config['distance']['xy_scale'] != 512 or config['distance']['angle_scale'] != float(np.pi):
        raise ValueError('Unexpected frozen analysis distance/temporal definitions')
    start = time.perf_counter()
    attempt = {'started_at':now(),'status':'running','config_sha256':sha(config_path),
        'code_commit_at_start':git(PROJECT,'rev-parse','HEAD'),
        'script_hashes':{p.name:sha(p) for p in sorted(BASE.glob('*.py'))},
        'software':{'python':platform.python_version(),'numpy':np.__version__,'zarr':zarr.__version__}}
    try:
        state,ep,local,source,root,audit = load_data(config)
        save(output/'results/data_audit.json',audit)
        print(json.dumps({'stage':'data_checked','episodes':len(audit['episode_ids']),'frames':len(state)}),flush=True)
        d = distances(state,block_rows=config['distance']['block_rows'])
        pools,temporal = pools_and_time(ep,local,config['neighbors']['exclude_temporal_radius'])
        dist = distributions(d,pools,temporal,config['statistics']['histogram_bins'])
        cal = resolve_calibration(d,pools['cross'],config['thresholds'])
        save(output/'results/resolved_thresholds.json',cal)
        print(json.dumps({'stage':'distributions_and_thresholds_ready','calibration':cal}),flush=True)
        nn = neighbor_statistics(d,pools,temporal,ep,config['neighbors'])
        grid = sensitivity(d,pools,ep,audit['episode_ids'],cal,config['thresholds'])
        profile_stats = {}
        temporal_close = {}
        for name,values in config['thresholds']['profiles'].items():
            threshold = resolve_profile(cal,values)
            profile_stats[name] = {'thresholds':threshold,'roles':{}}
            for role in ['similar','hard_a','hard_b']:
                mask = relation(d,role,threshold)
                profile_stats[name]['roles'][role] = {pool:candidate_summary(mask & eligible,ep,audit['episode_ids']) for pool,eligible in pools.items()}
                if role == 'similar':
                    temporal_close[name] = {key:{'similar_unordered_pairs':int(np.triu(mask & eligible,1).sum()),
                        'fraction_within_temporal_stratum':float((mask & eligible).sum()/eligible.sum())}
                        for key,eligible in temporal.items()}
        print(json.dumps({'stage':'neighbor_and_sensitivity_complete','conditions':len(grid)}),flush=True)
        batches,rows,examples,degrees = simulation(d,pools,ep,cal,config)
        for row in degrees:
            i = row['row_index']
            row.update(local_frame_index=int(local[i]),source_index=int(source[i]))
        write_csv(output/'results/primary_anchor_candidate_counts.csv',degrees)
        write_csv(output/'results/batch_details.csv',rows)
        save(output/'results/batch_examples.json',examples)
        flat = []
        for item in grid:
            flat.append({key:item[key] for key in ['kind','condition_id','pool','unordered_pairs',
                'anchors_with_candidate','anchors_without_candidate','anchor_coverage_fraction',
                'anchor_fraction_at_least_4_candidates','active_episode_count','largest_episode_endpoint_share',
                'effective_episode_count']})
        write_csv(output/'results/threshold_sensitivity.csv',flat)
        verification = validate_real(d,state,ep,local,pools,config)
        from figures import make_figures
        make_figures(output,config,d,pools,ep,local,source,root,dist,nn,cal,profile_stats,batches)
        result = {'status':'completed','config':config,'data_audit_ref':'results/data_audit.json',
            'frame_count':len(state),'episode_ids':audit['episode_ids'],
            'pair_count_convention':'Symmetric conditions count unordered distinct-frame pairs once; ordered pairs are twice this. Top-k relations and per-anchor candidate degrees are directed.',
            'array_storage_note':'Float64 distance computation, float32 RAM matrices; no dense matrix saved.',
            'distance_distributions':dist,'threshold_calibration':cal,'nearest_neighbors':nn,
            'threshold_sensitivity':grid,'profiles':profile_stats,'temporal_similar_coverage':temporal_close,
            'interpretation_limits':['Geometry is not action compatibility.',
              ('Thresholds are fixed absolute sensitivity criteria, not optimized action-compatibility tolerances.'
               if 'absolute_values' in config['thresholds'] else 'Thresholds are relative training-distribution percentiles, not calibrated physical/action tolerances.'),
              'Frame-uniform anchors and pair counts give longer episodes more weight.','Partial batches are reported, never padded into successful batches.']}
        save(output/'results/geometry_statistics.json',result)
        save(output/'results/batch_sampling_statistics.json',batches)
        save(output/'results/verification.json',verification)
        assert git(PROJECT,'diff','--name-only','HEAD','--','.',':(exclude)experiments/geometry_sampling_analysis') == ''
        for key,ref in audit['source_provenance'].items():
            assert source_record(Path(ref['path'])) == ref
        attempt.update(status='completed',completed_at=now(),wall_seconds=time.perf_counter()-start,
            external_tracked_files_unchanged=True,original_reference_git_states_unchanged=True,
            output_sha256={str(p.relative_to(output)):sha(p) for folder in ['results','figures']
                for p in sorted((output/folder).glob('*')) if p.is_file()})
        print(json.dumps({'stage':'completed','seconds':attempt['wall_seconds'],'verification':verification}),flush=True)
    except Exception:
        import traceback
        attempt.update(status='failed',error=traceback.format_exc(),failed_at=now())
        with (BASE/'execution_log.md').open('a',encoding='utf-8') as f:
            f.write('\n## '+now()+' failure\n\nCommand: `'+ ' '.join(sys.argv)+'`\n\n```text\n'+attempt['error']+'```\n')
        raise
    finally:
        suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        save(output/'results/execution_records'/f'{suffix}.json',attempt)


if __name__ == '__main__':
    main()
