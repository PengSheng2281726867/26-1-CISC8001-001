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
```

Read [Phase 0 audit](docs/phase0_audit.md) and its machine-readable results for exact
data hashes and baseline provenance; [method/control](docs/method_and_control.md) for
the exact loss and controlled differences. Seven necessary tests and the first-frozen-seed
untrained integration smoke passed; reports are under results/. Neither is a trained B1 result.
Audit/check outputs are immutable: the saved pilot's commands must not overwrite them.
For a new formal training repeat, choose an unused run ID. The resolved baseline config
is read and checked, and B1 settings come from configs/b1_pose_alt30_seed42.yaml.

Lightweight run files are in runs/<run_id>/. Heavy checkpoints, full match/trace CSVs,
and videos use root artifacts/fixed_alt_encoder/<run_id>/ and need write permission there.
No environment packages are changed. Heavy artifacts are local, ignored, and not backed up by Git.
