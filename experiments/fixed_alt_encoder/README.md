# B1 Pose-Supervised ALT-30

This isolated experiment adds an image-only pose auxiliary head and loss to the existing
ALT-30 training recipe. Inference remains one current RGB frame + pusher xy -> normalized
128D fusion embedding -> cosine top-1 -> eight absolute expert actions. Pose labels and
predictions never enter the policy. No shared baseline files or frozen splits are edited.

Use the existing environment, without installing packages:

```powershell
Set-Location D:\Code\Course_code\COURSE-HUZI-CODE\experiments\fixed_alt_encoder
$b1Python = 'D:\Code\Course_code\DP\.conda\pusht\python.exe'
& $b1Python -B run.py audit
& $b1Python -B run.py check
$b1Run = 'b1_alt030_split42_model42_pose1_pilot_v1'
& $b1Python -B run.py train --run-id $b1Run
& $b1Python -B run.py bank --run-id $b1Run
& $b1Python -B run.py diagnose --run-id $b1Run
& $b1Python -B run.py evaluate --run-id $b1Run
& $b1Python -B run.py report --run-id $b1Run
& $b1Python -B run.py verify --run-id $b1Run
```

Read [Phase 0 audit](docs/phase0_audit.md) and its machine-readable results for exact
data hashes and baseline provenance; [method/control](docs/method_and_control.md) for
the exact loss and controlled differences. Seven necessary tests and the first-frozen-seed
untrained integration smoke passed; reports are under results/. Neither is a trained B1 result.
Audit/check outputs are immutable: the saved pilot's commands must not overwrite them.
Repeat the gate without overwriting using `run.py check --check-id another_unique_id`.
For a new formal training repeat, choose an unused run ID. The resolved baseline config
is read and checked, and B1 settings come from configs/b1_pose_alt30_seed42.yaml.

Lightweight run files are in runs/<run_id>/. Heavy checkpoints, full match/trace CSVs,
and videos use root artifacts/fixed_alt_encoder/<run_id>/ and need write permission there.
No environment packages are changed. Heavy artifacts are local, ignored, and not backed up by Git.

All six phases are complete for `b1_alt030_split42_model42_pose1_pilot_v1`.
The saved ID cannot be reused to overwrite a run; choose a new ID for reproduction.

| Mean metric | Original ALT-30 | B1 |
|---|---:|---:|
| Official 50-seed score | 0.274424 | 0.178185 |
| Heldout matched block xy L2 | 46.42 | 32.20 |
| Heldout matched angle error (degrees) | 38.31 | 17.27 |
| Heldout matched pusher xy L2 | 39.85 | 46.63 |

B1 pose prediction averages 20.34 world-unit xy error and 13.55-degree angle error.
Paired score difference is **-0.096239**, bootstrap 95% CI **[-0.190912, -0.000689]**.
This is a negative closed-loop result despite improved block retrieval. Only one
training seed was evaluated; the interval covers simulator seeds, not model variation.
No tuning or additional formal evaluation was performed in response to scores.

Read [measured report](docs/final_analysis.md) and [interpretation in Chinese](docs/interpretation.md).
Full numerical evidence: [run directory](runs/b1_alt030_split42_model42_pose1_pilot_v1/),
including immutable config, training CSV, offline metrics, paired scores, inference
isolation, preservation checks, artifact checksums and final independent verification.
The directory also contains readable loss curves, heldout matches and a contact sheet
decoded from the predefined first four rollout videos. All 7 regression tests passed;
all 10 heavy artifact files were rehashed. Both diagnostic failures and fixes are retained
in [failure log](docs/failure_log.md); training and formal evaluation each ran once.
