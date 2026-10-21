"""Run from any working directory with the existing Push-T Python interpreter."""
import os
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parent
PROJECT = EXPERIMENT.parents[1]
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(EXPERIMENT))
from pusht.repro import configure_runtime
configure_runtime(EXPERIMENT)

if __name__ == '__main__':
    from b1alt.cli import main
    main()
