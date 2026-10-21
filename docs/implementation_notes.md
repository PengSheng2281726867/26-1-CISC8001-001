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

Status: Phase 1 core gate passed (five real-data tests); training and real closed-loop results will be recorded separately. No claim is based on contrastive loss alone.
