"""Check legacy simulator against real dataset observations and action replay.

This is environment validation, not policy training or a new benchmark rollout.
Both state restoration and uninterrupted action replay are reported separately.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    import h5py
    import numpy as np
    import mujoco_py
    import robosuite
    import robomimic
    from robomimic.utils import env_utils, obs_utils
    from PIL import Image, ImageDraw
    cfg = json.loads((HERE / 'config.json').read_text())
    np.random.seed(cfg['validation_seed'])
    env_root = PROJECT / cfg['environment_root']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = env_root / 'results' / ('validation_'+stamp)
    directory.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running', 'started_at_utc': stamp, 'python': sys.version,
              'platform': platform.platform(), 'dataset_sha256': sha(PROJECT / cfg['dataset']),
              'versions': {'robosuite': robosuite.__version__, 'robomimic': robomimic.__version__,
                           'mujoco_py': mujoco_py.__version__,
                           'mujoco_engine_integer': int(mujoco_py.functions.mj_version())},
              'robosuite_source_commit': cfg['robosuite_commit'],
              'script_sha256': sha(__file__), 'episodes': [], 'training': False,
              'validation_seed': cfg['validation_seed'],
              'setup_sources': {p.name: sha(p) for p in HERE.iterdir()
                                if p.is_file() and p.suffix in ['.py', '.sh', '.ps1', '.json', '.txt']},
              'artifact_directory': str(directory.relative_to(PROJECT)),
              'installed_packages': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()}
    result['build_flags'] = os.environ.get('CFLAGS', '')
    result['compiler'] = subprocess.check_output([os.environ.get('CC', 'gcc'), '--version'], text=True).splitlines()[0]
    result['system_packages'] = subprocess.check_output(
        ['dpkg-query', '-W', '-f=${Package}=${Version}\n', 'gcc', 'g++', 'libosmesa6',
         'libglew2.2', 'libgl1-mesa-dev', 'patchelf'], text=True).splitlines()
    result['mujoco_engine_binary_sha256'] = sha(Path(os.environ['MUJOCO_PY_MUJOCO_PATH']) / 'bin' / 'libmujoco200.so')
    result['mujoco_binding_binary_sha256'] = sha(mujoco_py.cymj.__file__)
    result['torch_compatibility_patch'] = json.loads((env_root / 'torch_compatibility_patch.json').read_text())
    result['glibc'] = subprocess.check_output(['ldd', '--version'], text=True).splitlines()[0]
    result['download_archives'] = {p.name: sha(p) for p in sorted((env_root / 'downloads').glob('*')) if p.is_file()}
    if result['dataset_sha256'] != cfg['dataset_sha256']:
        raise ValueError('Dataset bytes changed')
    assert robosuite.__version__ == cfg['robosuite_version']
    assert robomimic.__version__ == cfg['robomimic_version']
    assert mujoco_py.__version__ == cfg['mujoco_py_version']
    assert result['versions']['mujoco_engine_integer'] == 200
    env = None
    try:
        with h5py.File(PROJECT / cfg['dataset'], 'r') as f:
            metadata = json.loads(f['data'].attrs['env_args'])
            result['environment_metadata'] = metadata
            result['existing_split'] = {'train': len(f['mask/train']), 'valid': len(f['mask/valid']),
                                        'changed': False}
            keys = list(f['data/demo_0/obs'])
            obs_utils.initialize_obs_utils_with_obs_specs({'obs': {'low_dim': keys}})
            env = env_utils.create_env_from_metadata(env_meta=metadata, render=False,
                                                      render_offscreen=True, use_image_obs=False)
            assert env.action_dimension == 7
            panels = []
            for name in cfg['validation_episodes']:
                demo = f['data'][name]
                states, actions = demo['states'][:], demo['actions'][:]
                xml = demo.attrs['model_file']
                if isinstance(xml, bytes):
                    xml = xml.decode()
                env.reset_to({'model': xml, 'states': states[0]})
                per_key_error = {key: 0. for key in keys}
                for t, state in enumerate(states):
                    obs = env.reset_to({'states': state})
                    for key in keys:
                        error = float(np.max(np.abs(obs[key]-demo['obs'][key][t])))
                        per_key_error[key] = max(per_key_error[key], error)
                    if name == cfg['validation_episodes'][0] and t in [0, len(states)//2, len(states)-1]:
                        for camera in cfg['render_cameras']:
                            image = env.render(mode='rgb_array', height=cfg['render_size'],
                                               width=cfg['render_size'], camera_name=camera)
                            assert image.shape == (cfg['render_size'], cfg['render_size'], 3)
                            assert np.isfinite(image).all() and float(np.std(image)) > 1.
                            path = directory / f'{name}_frame{t}_{camera}.png'
                            Image.fromarray(image).save(path)
                            panels.append((t, camera, image))
                # Fresh XML/initial state; no intermediate teacher forcing during action replay.
                env.reset_to({'model': xml, 'states': states[0]})
                # A separate simulator evaluates state-aligned geometry without adding
                # any forward() calls or observable updates to the live action replay.
                comparison_sim = mujoco_py.MjSim(env.env.sim.model)
                cube_body = env.env.cube_body_id
                eef_site = env.env.robots[0].eef_site_id
                cube_start, cube_end = env.env.sim.model.get_joint_qpos_addr(env.env.cube.joints[0])
                nq = env.env.sim.model.nq
                actual_states, aligned_object_errors = [], []
                state_errors, object_errors, action_success = [], [], False
                for t, action in enumerate(actions):
                    obs, reward, done, info = env.step(action)
                    success = bool(env.is_success()['task'])
                    action_success = action_success or success
                    if t+1 < len(states):
                        sim_state = env.get_state()['states']
                        actual_states.append(sim_state.copy())
                        state_errors.append(float(np.max(np.abs(sim_state-states[t+1]))))
                        object_errors.append(float(np.max(np.abs(obs['object']-demo['obs']['object'][t+1]))))
                        comparison_sim.set_state_from_flattened(sim_state)
                        comparison_sim.forward()
                        pos = comparison_sim.data.body_xpos[cube_body].copy()
                        quat = comparison_sim.data.body_xquat[cube_body][[1, 2, 3, 0]]
                        rel = comparison_sim.data.site_xpos[eef_site] - pos
                        aligned = np.concatenate([pos, quat, rel])
                        aligned_object_errors.append(float(np.max(np.abs(aligned-demo['obs']['object'][t+1]))))
                differences = np.abs(np.stack(actual_states) - states[1:])
                peak_step, peak_dimension = np.unravel_index(np.argmax(differences), differences.shape)
                np.savez_compressed(directory / (name+'_action_replay.npz'),
                                    actual_states=np.stack(actual_states), expected_states=states[1:])
                row = {'episode': name, 'frames': len(states), 'action_dimension': int(actions.shape[1]),
                       'state_restoration_max_abs_observation_error': per_key_error,
                       'continuous_action_replay_max_abs_state_error': max(state_errors),
                       'continuous_action_replay_max_abs_object_observation_error': max(object_errors),
                       'state_aligned_object_max_abs_error': max(aligned_object_errors),
                       'state_component_max_abs_error': {
                           'time': float(differences[:, 0].max()),
                           'qpos': float(differences[:, 1:1+nq].max()),
                           'qvel': float(differences[:, 1+nq:].max()),
                           'cube_xyz': float(differences[:, 1+cube_start:1+cube_start+3].max()),
                           'cube_quaternion': float(differences[:, 1+cube_start+3:1+cube_end].max())},
                       'largest_state_error_action_index': int(peak_step),
                       'largest_state_error_flat_dimension': int(peak_dimension),
                       'model_state_layout': {'nq': int(nq), 'nv': int(env.env.sim.model.nv),
                                              'cube_qpos_start': int(cube_start), 'cube_qpos_end': int(cube_end)},
                       'continuous_action_replay_final_success': success,
                       'continuous_action_replay_ever_success': action_success,
                       'continuous_action_replay_final_reward': float(reward)}
                result['episodes'].append(row)
                save(directory / 'validation.json', result)
                print(json.dumps(row, indent=2), flush=True)
            size = cfg['render_size']
            sheet = Image.new('RGB', (3*size, 2*(size+28)), 'white')
            draw = ImageDraw.Draw(sheet)
            frame_indices = sorted(set(t for t, _, _ in panels))
            for t, camera, image in panels:
                x = frame_indices.index(t)*size
                y = cfg['render_cameras'].index(camera)*(size+28)
                sheet.paste(Image.fromarray(image), (x, y+28))
                draw.text((x+5, y+6), f'demo_0 / t={t} / {camera}', fill='black')
            sheet.save(directory / 'lift_dataset_replay.png')
            result['rendered_cameras'] = cfg['render_cameras']
            result['state_restoration_passed'] = all(
                max(r['state_restoration_max_abs_observation_error'].values()) <= cfg['state_replay_atol']
                for r in result['episodes'])
            result['continuous_action_replay_passed'] = all(
                r['continuous_action_replay_final_success'] and
                r['continuous_action_replay_max_abs_state_error'] <= 1e-5 for r in result['episodes'])
            result['continuous_action_replay_task_success_count'] = sum(
                r['continuous_action_replay_final_success'] for r in result['episodes'])
            result['status'] = 'passed' if result['state_restoration_passed'] and result['continuous_action_replay_passed'] else 'needs_diagnosis'
            result['artifacts'] = [{'path': str(p.relative_to(PROJECT)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                                   for p in sorted(directory.glob('*.png'))]
            save(directory / 'validation.json', result)
            save(HERE / 'results' / 'validation.json', result)
            sheet.save(HERE / 'results' / 'lift_dataset_replay.png')
            if result['status'] != 'passed':
                raise RuntimeError('Environment runs, but dataset replay gates need diagnosis; see validation.json')
            print('ALL ENVIRONMENT AND REPLAY GATES PASSED', flush=True)
    except BaseException:
        result['error'] = traceback.format_exc()
        if result['status'] == 'running':
            result['status'] = 'failed'
        save(directory / 'validation.json', result)
        raise
    finally:
        if env is not None:
            env.env.close()


if __name__ == '__main__':
    main()
