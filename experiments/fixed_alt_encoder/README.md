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
```

Read [Phase 0 audit](docs/phase0_audit.md) and its machine-readable results for exact
data hashes and baseline provenance. Full run commands and measured results will be
added as their gates pass. Heavy artifacts are local, ignored, and not backed up by Git.
