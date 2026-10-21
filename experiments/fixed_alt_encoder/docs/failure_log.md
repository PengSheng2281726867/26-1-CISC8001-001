# B1 failure log

## 2026-10-21T08:36Z — repeated audit command

Command: existing Push-T Python `-B run.py audit` (second invocation).
The first invocation completed the full-byte/data/reference audit and wrote its JSON
and Markdown, but the orchestration output was not visible. A redundant invocation
was issued while checking outputs; it correctly failed with
`ValueError: Refusing to overwrite immutable file: .../results/phase0_audit.json`.
The timestamp/code-status fields differ on repeat. The original successful audit is
preserved. Verified its completed Markdown and JSON directly. No training or scientific
setting was affected. Future orchestration retains command session IDs and checks
existing outputs before rerunning immutable stages.

## 2026-10-21T08:56Z — diagnostic artifact-path adapter

The first diagnostic attempt failed before loading a model: the new attempt-isolation
wrapper passed its output directory to the checkpoint loader, which searched for
`diagnostic_attempts/<timestamp>/encoder_final.pt` instead of the run's artifact root.
The automatic entry below preserves the exact command and traceback. Root cause is
path wiring, not training or data alignment. One fix: pass the immutable model/bank
input root separately from the unique diagnostic output directory. The trained
checkpoint, bank and failed attempt directory are preserved. Rerun the same heldout
diagnostic with a new attempt directory; no retraining or method change.

## 2026-10-21T08:56:00.302352+00:00 — diagnose

Command: `D:\Code\Course_code\DP\.conda\pusht\python.exe run.py diagnose --run-id b1_alt030_split42_model42_pose1_pilot_v1`

Run: b1_alt030_split42_model42_pose1_pilot_v1; model seed42.

```text
Traceback (most recent call last):
  File "D:\Code\Course_code\COURSE-HUZI-CODE\experiments\fixed_alt_encoder\b1alt\cli.py", line 39, in main
    diagnose(cfg, manifest, rd, heavy, meta)
  File "D:\Code\Course_code\COURSE-HUZI-CODE\experiments\fixed_alt_encoder\b1alt\diagnose.py", line 248, in diagnose
    _diagnose(cfg, manifest, work, raw, meta)
  File "D:\Code\Course_code\COURSE-HUZI-CODE\experiments\fixed_alt_encoder\b1alt\diagnose.py", line 183, in _diagnose
    model, bank = load_model(cfg, heavy, meta), load_bank(heavy, meta)
  File "D:\Code\Course_code\COURSE-HUZI-CODE\experiments\fixed_alt_encoder\b1alt\experiment.py", line 115, in load_model
    if sha256(path) != meta['checkpoint_sha256']:
  File "D:\Code\Course_code\COURSE-HUZI-CODE\src\pusht\repro.py", line 17, in sha256
    with Path(path).open('rb') as f:
  File "D:\Code\Course_code\DP\.conda\pusht\lib\pathlib.py", line 1119, in open
    return self._accessor.open(self, mode, buffering, encoding, errors,
FileNotFoundError: [Errno 2] No such file or directory: 'D:\\Code\\Course_code\\COURSE-HUZI-CODE\\artifacts\\fixed_alt_encoder\\b1_alt030_split42_model42_pose1_pilot_v1\\diagnostic_attempts\\2026-10-21T085600.196706_0000\\encoder_final.pt'
```

## 2026-10-21T08:56:30.546357+00:00 — diagnose

Command: `D:\Code\Course_code\DP\.conda\pusht\python.exe run.py diagnose --run-id b1_alt030_split42_model42_pose1_pilot_v1`

Run: b1_alt030_split42_model42_pose1_pilot_v1; model seed42.

```text
Traceback (most recent call last):
  File "D:\Code\Course_code\COURSE-HUZI-CODE\experiments\fixed_alt_encoder\b1alt\cli.py", line 33, in main
    {'time': now(), 'command': args.command, 'code': provenance(PROJECT),
  File "D:\Code\Course_code\COURSE-HUZI-CODE\src\pusht\repro.py", line 67, in provenance
    'tracked_diff_sha256': hashlib.sha256(git(root, 'diff', 'HEAD').encode()).hexdigest()}
  File "D:\Code\Course_code\COURSE-HUZI-CODE\src\pusht\repro.py", line 59, in git
    return result.stdout.strip()
AttributeError: 'NoneType' object has no attribute 'strip'
```
## Windows locale provenance fix (after 08:56Z path fix)

The next attempt failed before entering diagnostics: shared `pusht.repro.provenance`
uses `subprocess.run(text=True)` without an encoding, so Python's GBK default could
not decode a UTF-8 Git diff containing Unicode documentation. The full automatic
traceback is preserved below/above. Added an isolated provenance helper capturing
raw Git bytes, explicitly decoding text fields as UTF-8 and hashing exact diff bytes.
No shared baseline file or Git global setting was modified. This changes only
provenance serialization; model, data, seeds, loss and inference are unchanged.
The next diagnostic invocation also exercises the Unicode provenance regression.

Recovery verified: the next diagnostic invocation completed all2426 frames per model,
exactly reproduced the original offline metrics, passed bank/index/independent FP64
checks, and saved plots. A fresh check ID phase3_regression_v1 passed all7 tests and
the official one-seed untrained smoke. Formal training was executed only once.
