"""Use Python -B; all reference code is imported read-only."""
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, str(PROJECT / 'src'))
from pusht.repro import configure_runtime
configure_runtime(PROJECT)

if __name__ == '__main__':
    from pusht.cli import main
    main(PROJECT)
