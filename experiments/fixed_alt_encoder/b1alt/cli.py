import argparse
from b1alt.common import DEFAULT_CONFIG, config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['audit'])
    p.add_argument('--config', default=str(DEFAULT_CONFIG))
    args = p.parse_args()
    cfg = config(args.config)
    from b1alt.audit import run
    run(cfg)
