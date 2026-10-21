import argparse
from b1alt.common import DEFAULT_CONFIG, config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['audit', 'check', 'train', 'bank', 'diagnose', 'evaluate', 'report'])
    p.add_argument('--config', default=str(DEFAULT_CONFIG))
    p.add_argument('--run-id')
    args = p.parse_args()
    cfg = config(args.config)
    rd = heavy = meta = None
    try:
        if args.command in ['audit', 'check']:
            if args.command == 'audit':
                from b1alt.audit import run
            else:
                from b1alt.check import run
            return run(cfg)
        from b1alt.experiment import create, existing, bank, save_state
        if args.command == 'train':
            manifest, rd, heavy, meta = create(cfg, args.run_id)
            from b1alt.train import train
            train(cfg, manifest, rd, heavy, meta)
        else:
            manifest, rd, heavy, meta = existing(cfg, args.run_id)
            if args.command == 'bank':
                bank(cfg, manifest, rd, heavy, meta)
            elif args.command == 'diagnose':
                from b1alt.diagnose import diagnose
                diagnose(cfg, manifest, rd, heavy, meta)
            elif args.command == 'evaluate':
                from b1alt.evaluate import evaluate
                evaluate(cfg, manifest, rd, heavy, meta)
            elif args.command == 'report':
                from b1alt.report import report
                report(cfg, manifest, rd, heavy, meta)
        save_state(rd, heavy, meta)
    except Exception:
        from b1alt.experiment import failure
        failure(args.command, rd, heavy, meta)
        raise
