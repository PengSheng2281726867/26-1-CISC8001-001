# Method differences and frozen implementation

| Aspect | ALT paper | Inspected ALT main | Push-T adaptation |
|---|---|---|---|
| Inputs | camera images + numerical pose fusion | hand + third images; pose branch commented out | one current 96x96 RGB image + current pusher x,y |
| Backbone | ResNet18 image encoding | ImageNet ResNet18 shared across two images | ImageNet1K_V1 ResNet18, all parameters fine-tuned |
| Image feature | low dimensional representation | 128 per image | 128 |
| Pose feature | numerical pose | 7D pose MLP exists in comments | active 2->64->32 MLP |
| Fusion | concatenated features | 256->256->128 | 160->256->128, L2 normalized |
| Augmentation | contrastive views | color jitter, random crop, pose noise | two independent mild color-jitter views, same exact position; no geometry or pose noise |
| Input resolution | robot camera setting | 240x320 | native Push-T 96x96, no resize/crop |
| Training | contrastive representation | Adam 1e-4, batch32, 40 epochs, tau0.4 | same starting recipe; not tuned for Push-T |
| Retrieval | lookup; optional OOD monitor | source runtime includes additional robot handling | top-1 cosine, always return best match, no threshold/fallback/compensation |
| Action | robot trajectory | robot-specific | absolute 2D targets, eight actions then fresh observation |

This is an Image + Position adaptation following the paper's fusion idea, not an unchanged main-branch reproduction. The position tensor is numerical, not a second image or temporal frame.

Image preprocessing: data RGB HWC float32 in integer 0..255 is losslessly cached as uint8, then CHW float32 /255. Forward uses ImageNet mean [0.485,0.456,0.406] and std [0.229,0.224,0.225]. No ImageNet crop/224 resize is applied; this explicit task adaptation preserves geometry. Positions are mapped x/256-1 using environment bounds, with no data-fitted statistics. Actions are never normalized.

Training pairs are independently augmented versions of the same observation and the same current position. Actions are absent from training batches and loss. Model/shuffle/augmentation seeds are separately recorded. Checkpoint selection uses the final epoch, never rollout scores. Default epochs40/batch32/lr1e-4/tau0.4 are a starting recipe only.

Bank includes every frame in the exact selected training episode list and is re-encoded in eval mode after training. Each entry stores normalized embedding, episode, local frame, and source index. A single source action array is accessed by index. For a frame near an episode end, each chunk index is min(t+k, episode_end-1); last actions repeat and cannot cross episode boundaries. Offset=0 follows collection order. Exact ties choose the first bank entry; duplicate observations may therefore recover an equivalent entry.

Policy is restricted to image and agent_pos. Any wrapper history must use its latest frame. No block pose, reward, goal, future observation, future target position as input, OOD handling, interpolation, or relative compensation is included. Official test settings remain legacy_test=True, seeds100000..100049 and max_steps300.

The official PushTImageRunner.run remains unchanged. The adapter imports it from the read-only reference path, requests zero demonstration/training-seed rollouts and evaluates the fixed simulator seed list. ObservedImageEnv only records returned termination flags and actual control-step counts; VectorTrace observes rewards and timing through the official vector API. Neither affects simulator transitions. A test compares original/instrumented reset and action transitions exactly. Per-seed and aggregate telemetry must agree with the score returned by the official Runner.

One environment at a time is used for auditable episode-to-query attribution. Training is FP32, TF32 disabled, cuDNN deterministic with benchmark disabled, deterministic Torch algorithms enabled, and CUBLAS_WORKSPACE_CONFIG=:4096:8. Dataloader workers=0. Model initialization, shuffle and augmentation use separately recorded seeds; validation views use augmentation_seed+1. All selected frames enter the bank, including drop-last training remainders. Final checkpoint is selected without validation or test score feedback.

Status: Phase 1 core gate passed; seven real-data/interface gates now pass. Training and actual closed-loop results are recorded separately by run ID. No claim is based on contrastive loss alone.

Git serialization note: the first published commits normalized frozen split CRLF to LF. The current .gitattributes preserves the original exact bytes and advertised SHA256 without changing episode IDs, order or seeds. Reproducing the ALT-30 pilot uses its recorded policy/training source commit69e3517 together with the exact split blob from7a13763 or later (SHA2561b786b545a200f8fbbb28fb9385bc1ff80ed8699aba7c14d6fd117af7670322a). The serialization fix did not modify training/policy code or the on-disk manifest used by the running pilot. Study definitions and planned additional model seeds are in experiments/study_plan.json; only registry.csv and completed run records establish actual experiments.
