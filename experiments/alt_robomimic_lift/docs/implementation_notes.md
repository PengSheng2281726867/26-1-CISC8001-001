# Lift ALT implementation notes

## Scope and provenance

The encoder and policy are a small, independent implementation inside this experiment folder. No ALT source code was copied. The supplied local ALT snapshot is MIT-licensed, but contains no Git metadata; its file hashes and the limits of commit attribution are recorded in `results/source_manifest.json` and `docs/upstream_provenance.md`. The public ALT path branch is commented out in the supplied snapshot. This Lift baseline intentionally fuses two camera features with current robot proprioception, following the paper's fusion idea without claiming an unmodified upstream runtime.

## Data and temporal contract

`LiftFrameStore` uses only `obs/agentview_image[t]`, `obs/robot0_eye_in_hand_image[t]`, `robot0_eef_pos[t]`, `robot0_eef_quat[t]`, `robot0_gripper_qpos[t]`, plus `actions[t]` for the action store. The action is the command applied from the pre-action observation at the same index. No `states`, object pose, `next_obs`, reward, task success, or future action enters the encoder or policy. Image HWC RGB bytes are converted to CHW float, divided by 255, then normalized with ImageNet statistics. No image geometry is changed.

The proprioception quaternion is normalized in audited XYZW order and sign-canonicalized by nonnegative `w`. Mean and standard deviation are fitted on the 30 selected training episodes only; near-constant dimensions use scale 1. The same saved statistics are used for memory-bank and online preprocessing.

## Encoder and loss

One torchvision ResNet-18 instance processes both cameras with shared weights. Its verified ImageNet classifier is replaced by a 128D projection, then fine-tuned. A 9→64→32 proprioception MLP and 288→256→128 fusion head produce the final unit-normalized embedding. The `image_only` architecture option is available for later ablation work but is not trained here.

Every training example creates two independent `ColorJitter` views from its synchronized camera pair. The numeric proprioception vector is the same in both views. The only training objective is NT-Xent at temperature 0.4; expert actions and all task labels are excluded. The run uses the frozen batch size 32, Adam 1e-4, and 40 epochs. The final epoch is saved; no held-out test seed selects a checkpoint.

## Memory and actions

The bank re-encodes every observation from exactly the frozen 30 training episodes with deterministic, non-augmented transforms. Each row contains a unit 128D embedding, episode ID, local frame index, flattened source index, and current proprioception for audit. It does not duplicate action sequences. Retrieval is cosine top-1 with lowest-row deterministic tie breaking. The selected source HDF5 `actions[t]` is returned in the original normalized incremental OSC_POSE-plus-gripper units. No relative correction, interpolation, fallback, or similarity threshold is applied. The formal policy replans after every one-action command. Experimental 4/8-action chunks have padding unit tests only and are not used in the pilot.

## Train/evaluation runtime split

Training uses the local Python 3.10 venv with verified CUDA Torch/TorchVision packages. The evaluator imports the same model definitions in the existing Python 3.8/Torch 1.10 CPU environment because the matched RoboSuite 1.2.0 / MuJoCo 2.0 simulator is installed there. Checkpoint keys are standard ResNet-18 state dictionaries so the legacy inference runtime can load them. Evaluation logs encoder, search, preprocessing, and action extraction timings separately; CUDA training time and CPU simulator inference time are not conflated.
