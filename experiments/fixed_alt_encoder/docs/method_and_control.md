# B1 method and controlled differences

Imports the existing attributed `FusionEncoder`, `ObservationFrames`, `PushTData`,
NT-Xent, memory bank, ALTPolicy and official Runner integration. See root provenance
and ALT MIT license. No upstream source is copied or modified.

`B1FusionEncoder` constructs all baseline modules first, then a 128->64->4 image-only
pose head. The inherited inference `forward(image, agent_pos)` is unchanged and never
executes the head. Training reuses one image feature per view for fusion and pose.
No pose label/prediction, reward or block state enters inference or bank similarity.

The same model seed42 reproduces every baseline initial state tensor bit-for-bit in
the current code/environment; original historical initialization tensors were not saved,
so this is a repeat-instantiation test, not a comparison to an archived initial model.
The same verified ImageNet checkpoint is loaded; all parameters fine-tune. Extra head
initialization consumes RNG only after all shared layers; training resets augmentation
seed44. Shuffle generator43 and single-process drop_last batches match the baseline.

Only method difference: head and auxiliary loss. Both independent mild ColorJitter views
share the same-index pose target `[x/256-1,y/256-1,sin(theta),cos(theta)]`.
`L_xy` and `L_angle` each average the two view MSEs; `L_pose=L_xy+L_angle`;
`L_total=NTXent(tau=.4)+1.0*L_pose`. Lambda1 was fixed before results, reflecting
the comparable normalized target ranges. Actions are never training targets.

Training is fresh, Adam1e-4, batch32, 40 epochs, final checkpoint; no early stopping.
Data are exactly the frozen 30 episodes; heldout20 are diagnostic only. Bank encodes
all training frames deterministically. Cosine top1, no OOD/fallback/blending, offset0,
8 absolute-position commands, same-episode last-action padding, latest single frame.
Official evaluation remains legacy_test=True, max_steps300, exact frozen50 seeds.
First four frozen seeds get videos. Score/bootstrap definitions reuse baseline code.

Inference consumes no pose labels. Offline label arrays support pose/retrieval error
measurement only. A baseline without a trained pose head has no comparable pose-output
accuracy; B1 heldout pose error alone cannot prove improvement over that baseline.
