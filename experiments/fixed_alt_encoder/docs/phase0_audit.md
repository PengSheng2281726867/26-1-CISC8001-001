# B1 Phase 0 read-only audit

Completed 2026-10-21T08:35:50.169834+00:00. Branch main; starting commit 4aea96bf57a8da623d5c183438ed7b0763842c78.
Only the user-supplied experiment AGENTS.md was untracked at entry. No pre-existing code was overwritten.

Frozen split SHA256: `1b786b545a200f8fbbb28fb9385bc1ff80ed8699aba7c14d6fd117af7670322a`. Full dataset bytes were rehashed and match
`9b8c01cfa9d217337d1bad6cc84f81a0d8e1187b918acf0c76af0c08e9b110ee`. Exact 30 training IDs, 20 heldout IDs and all 50 simulator
seeds match the original ALT-30 run and are saved in [phase0_audit.json](../results/phase0_audit.json).
No split was created or modified.

State columns are pusher x,y, block x,y, block angle (radians); extrema are
[13.45642375946045, 32.93829345703125, 57.47176742553711, 108.27995300292969, 0.00021559125161729753] to [496.14617919921875, 510.9578857421875, 439.9153137207031, 485.6640930175781, 6.283087730407715]. Ten first/last episode frames reconstructed
their stored RGB exactly using block pose for audit only. Pose labels use columns 2:5
at the same global index. Position targets remain world coordinates, not image pixels.

Original ALT-30: 40 epochs; 3806 bank entries; official 50-seed score
0.27442389936027484. Config, training log, bank validation, metrics,
representation diagnostics and source hashes were read and recorded. Reference source
commits, dirty states and tracked-byte fingerprints match the existing reference snapshot.
The baseline is an attributed independent implementation; B1 imports it without copying
upstream files. See root docs/upstream_provenance.md and licenses/ALT-MIT.txt.

Environment: D:\Code\Course_code\DP\.conda\pusht\python.exe; Torch 2.7.1+cu128; CUDA 12.8;
GPU NVIDIA GeForce RTX 5090. No packages were installed or changed. Official Runner integration,
offset=0, same-episode last-action padding and 8 absolute-position actions will be reused.
The pre-existing audit confirms collection saves observation/action before env.step.
Shared source/config/result byte hashes are saved for a final preservation check.
