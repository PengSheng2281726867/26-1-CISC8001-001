# Upstream provenance and license review

References were inspected locally, read-only, not assumed to equal a current remote branch.

| Component | Verified local repository | Commit | License |
|---|---|---|---|
| ALT | D:/Code/Course_code/ACl/alt | 939ed620276f0088d7e86021dbf1bc37240f02e0 | MIT, licenses/ALT-MIT.txt |
| DP environment/data interfaces | D:/Code/Course_code/DP/diffusion_policy | 5ba07ac6661db573af695b419a7947ecb704690f | MIT, licenses/DP-MIT.txt |

Both initial Git worktrees were clean. Full current tracked-file hashes are in results/reference_snapshot.json; repeat the audit to verify no reference changes. Wrapper directories ACl and DP are not the actual Git roots. No reference code, config, environment, dataset, or Git config is modified. Python bytecode writes are disabled; caches and output paths point into this project.

Primary references: [ALT source](https://github.com/StanfordMSL/alt), [paper v1, He et al. 2025](https://arxiv.org/html/2505.05787v1), [DP source](https://github.com/real-stanford/diffusion_policy).

File-level conceptual provenance:

- src/alt/encoder.py: ALT training.py:121-179 supplies the ResNet18 projection/fusion idea. Independent code activates a 2D numerical position branch and uses one image.
- src/alt/loss.py: ALT training.py:184-217 and SimCLR mathematical NT-Xent objective. Independent dynamic-batch implementation excludes self similarity and uses symmetric positive targets.
- src/alt/memory.py: ALT data_creation.py / rollout.py nearest-neighbor memory concept. Independent indices-only Push-T memory with deterministic non-augmented encoding and episode-safe action extraction.
- src/pusht/data.py: reads actual Zarr and follows verified DP collection semantics. It does not copy the upstream dataset normalizer, which fits all replay episodes by default.
- src/pusht/audit.py: independently reconstructs images via imported DP environment. Full simulator state is confined to this diagnostic and is never a policy input.
- Future rollout adapter imports DP environment/Runner from the configured read-only path; no redundant DP repository is distributed.

ALT's supplied license unusually names '.NET Foundation and Contributors'. The exact MIT text is preserved verbatim without inventing a replacement copyright holder. This project uses independently written source, attributed to the method and inspected source; MIT notices are retained conservatively. DP's Columbia AIRLab MIT notice is also retained. PyTorch/torchvision code and ImageNet weights are dependencies, not vendored source; exact official weight URL/checksum is in results/pretrained_weights.json.
