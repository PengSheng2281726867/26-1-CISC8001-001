"""Read Git bytes explicitly; Windows GBK must not decode UTF-8 source diffs."""
import hashlib
import os
from pathlib import Path
import subprocess


def provenance(root):
    root = str(Path(root).resolve())
    def raw(*args):
        p = subprocess.run(['git', '-c', f'safe.directory={root}', '-C', root, *args],
                           capture_output=True, env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})
        if p.returncode:
            raise RuntimeError(p.stderr.decode('utf-8', errors='replace'))
        return p.stdout
    def text(*args):
        return raw(*args).decode('utf-8', errors='strict').strip()
    return {'path': root, 'commit': text('rev-parse', 'HEAD'),
            'branch': text('branch', '--show-current'), 'dirty_status': text('status', '--porcelain=v1'),
            'remote': text('remote', 'get-url', 'origin'),
            'tracked_diff_sha256': hashlib.sha256(raw('diff', 'HEAD')).hexdigest(),
            'diff_hash_definition': 'SHA256 exact stdout bytes of git diff HEAD (no locale decoding)'}
