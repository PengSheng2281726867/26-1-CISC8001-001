# Failure log

## Initial environment and editor diagnostics (2026-10-20)

- `git status` in the empty project returned ?not a git repository?; confirmed existing content was AGENTS.md only, then initialized main locally without committing scaffold alone.
- The earlier AGENTS.md project path differed from the permitted workspace. Read-only audit proceeded; user updated AGENTS.md to the current Course_code path before any project writes.
- Git's default Windows Schannel backend failed remote inspection with SEC_E_NO_CREDENTIALS. One attributable fix: command-local OpenSSL backend; ls-remote then succeeded with no refs (empty remote). No credential changes.
- The sandbox identity differs from the source owner, so Git reports dubious ownership. All Git access uses command-local `-c safe.directory=<exact repo>`; global config and reference config are untouched.
- An initial Python `-c` audit probe lost quotes through PowerShell and raised SyntaxError. Fixed by literal here-string stdin; the smallest data inspection then succeeded.
- `apply_patch` failed to write both .gitignore and README.md despite permitted project writes. A native PowerShell write probe succeeded. Project files are therefore written with literal Python stdin, confined to the project root. The probe was removed.
- An initial pusher-color diagnostic included slate block pixels. Tightened the diagnostic color difference mask; exact state-to-image reconstruction subsequently verified synchronization (five episodes, pixel MAE=0). No policy or scientific variable changed.
- Reconstructing an already stored body pose with legacy=True applies initial-state CoM rotation twice. Diagnostic reconstruction uses legacy=False (MAE=0); official rollout seed initialization remains legacy_test=True.

No model experiments have been claimed from these checks. Subsequent failed commands append exact UTC time and traceback below.

## 2026-10-20T12:47:03.068042+00:00

Command: `scripts/run.py check`

Config: `D:\Code\Course_code\COURSE-HUZI-CODE\configs\alt_pusht_base.yaml`; model seed: 42

```text
Traceback (most recent call last):
  File "D:\Code\Course_code\COURSE-HUZI-CODE\src\pusht\cli.py", line 61, in main
    raise RuntimeError(f'Core gate failed: {len(result.failures)} failures, {len(result.errors)} errors')
RuntimeError: Core gate failed: 0 failures, 1 errors
```

Core gate failure detail: test_pretrained_finetuning_and_bank_roundtrip reached bank roundtrip after successful ImageNet tensor comparison, finite backward and backbone parameter update; tempfile.TemporaryDirectory(dir=artifacts/tests) raised FileNotFoundError [WinError 3]. Root cause: test order placed this case before the case that created artifacts/tests. Fix: create the parent locally in each test. No method change. Re-run the core gate.

## 2026-10-21T02:03:33.477272+00:00 execution infrastructure interruption

On 2026-10-20 after the core gate passed, command creation failed before Git staging/commit/push. Subsequent git status and even Get-Location failed with `Rejected("Failed to create unified exec process: helper_unknown_error: setup refresh had errors")`. apply_patch also could not write the diagnostic. No training or Git push was executed.

On 2026-10-21 local read commands recovered, while a concurrent remote ls-remote still failed before process creation with the same infrastructure error. The five core tests were then rerun successfully. This is not evidence of GitHub authorization failure. Attempt the validated local commit and then push sequentially; preserve the actual result. Phase 2 training/Runner integration remains unimplemented.

## 2026-10-21T02:19:26.417956+00:00 sandbox ownership diagnosis

The user supplied sandbox logs showing failure to open this project .git for READ_CONTROL | WRITE_DAC, Windows error5. Read-only user diagnostics confirmed an elevated administrator terminal and .git ownership by CodexSandboxOffline. The initial git init was performed in the sandbox, which is the likely origin of that ownership. User-side ownership repair restored the directory owner to DESKTOP-T1L0A7S/yc67001. A subsequent normal sandbox Get-Acl and Git status now execute successfully. No reference repository ownership or ACL was changed.

2026-10-21T02:20:02.696680+00:00: After restoring owner, git commit -m "feat: initialize Push-T ALT core and reproducibility scaffold" launched but failed with fatal: Unable to create .git/index.lock: Permission denied. This is a separate Git-metadata write restriction, not the former setup-refresh process-launch failure. No commit or push was created; do not weaken sandbox ACLs to work around it.

## 2026-10-21T02:37:16.427108+00:00 frozen split Git byte-integrity correction

While ALT-30 training continued unchanged, a read-only SHA256 comparison found the working frozen manifest hash1b786b545a200f8fbbb28fb9385bc1ff80ed8699aba7c14d6fd117af7670322a but the initial Git blob hash1501d6db20d943c54fcf7932989320aabea585de77edc935252e295158231abd. Windows core.autocrlf converted635 CRLF line endings to LF in Git. This could make cross-platform reproduction fail its checksum. Fix: add narrow .gitattributes -text rules for frozen split JSON/sidecars and re-stage their original exact bytes. No working manifest bytes, episode IDs, seeds, data, model settings or training process are changed. Verify the new Git blob hash equals the unchanged advertised manifest hash. This is a serialization repair, not a new split.

2026-10-21T02:38:18.415365+00:00: The serialization repair produced expected CR-at-EOL whitespace warnings in git diff --check because raw original CRLF bytes are now preserved. Narrowly mark those frozen files whitespace=cr-at-eol in .gitattributes, without changing any manifest bytes, and rerun the check. Other files retain normal whitespace checking.

## Completed ALT-30 pilot: performance limitation, not an execution failure

Run alt_030_split42_model42_pilot_v1 completed all50 frozen rollouts with mean official score0.2744238994. Predefined seed100000 scores0 and repeatedly retrieves67:30 despite cosine0.9924 and substantial current/expert pusher-position mismatch. Inspected real video frames and all1875 traces. Checks ruled out temporal offset, action normalization, bank episode crossing, history order and Runner API errors; input/transition parity and official-score cross-checks pass. Likely method limitation: learned cosine similarity and unmodified absolute action reuse need not preserve task-relevant geometry. This is a hypothesis from observed traces, not a proved mechanism. No scientific settings changed; preserve this negative/weak result and continue the frozen feasible-scale pilot.
