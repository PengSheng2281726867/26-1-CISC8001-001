"""Clear a legacy Torch ELF executable-stack marker for recent glibc loaders.

Only touches the newly installed project environment. Keeps an original backup,
records hashes and actual differing byte count, and never modifies MuJoCo.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
cfg = json.loads((HERE / 'config.json').read_text())
root = PROJECT / cfg['environment_root']
expected = (root / 'venv').resolve()
assert Path(sys.prefix).resolve() == expected
library = expected / 'lib/python3.8/site-packages/torch/lib/libtorch_cpu.so'
library.resolve().relative_to(expected)
marker = subprocess.check_output(['patchelf', '--print-execstack', str(library)], text=True).strip()
record_path = root / 'torch_compatibility_patch.json'
if marker.endswith('X'):
    before = library.read_bytes()
    backup = library.with_name(library.name+'.original_executable_stack')
    if backup.exists():
        raise RuntimeError('A previous original backup exists; do not overwrite it')
    shutil.copy2(library, backup)
    subprocess.check_call(['patchelf', '--clear-execstack', str(library)])
    after = library.read_bytes()
    after_marker = subprocess.check_output(['patchelf', '--print-execstack', str(library)], text=True).strip()
    assert not after_marker.endswith('X')
    record = {'reason': 'glibc 2.41+ rejects legacy Torch CPU executable-stack request',
              'library': str(library.relative_to(PROJECT)), 'backup': str(backup.relative_to(PROJECT)),
              'before_marker': marker, 'after_marker': after_marker,
              'before_sha256': hashlib.sha256(before).hexdigest(),
              'after_sha256': hashlib.sha256(after).hexdigest(),
              'file_size_unchanged': len(before) == len(after),
              'differing_bytes': sum(a != b for a, b in zip(before, after)) if len(before) == len(after) else None}
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)
else:
    print('Torch CPU library does not request executable stack; no change', flush=True)
import torch
assert (torch.tensor([2, 3]).sum()).item() == 5
print('TORCH IMPORT AND CPU TENSOR CHECK PASSED:', torch.__version__, flush=True)
