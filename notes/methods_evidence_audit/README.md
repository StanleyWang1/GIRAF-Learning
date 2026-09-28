# Methods evidence audit — 2026-09-14

## Scope and evidence

Ring/peg means **ring_over_peg_redo only**, as requested. The initial ring dataset and its checkpoints are excluded. Baseline is interpreted as DP-Full and joint_angles as DP-Angles, confirmed by checkpoint configuration.

Directly inspected: the three raw and cleaned Zarr stores; cleaned ZIP low-dimensional arrays; four saved training config.json files and their W&B metadata; all six epoch-0500 checkpoint payloads; available training metrics; named rollout configs and per-episode control logs; committed learning-code differences at 2ffc15c, 3bd233e, baa852d, and 8f3860a. No training, cleaning, or deployment was rerun. Existing datasets/checkpoints were not modified.

## 1. Dataset sizes and time

| Task | Raw recorded trials | Retained original trials | Cleaned episodes/segments | Cleaned frames | Train / validation |
|---|---:|---:|---:|---:|---:|
| Ring/peg redo | 218 | 206 | 213 | 119,125 | 192 / 21 |
| Cable stringing | 121 | 120 | 120 | 119,188 | 108 / 12 |
| Whiteboard checkmark | 211 | 202 | 202 | 76,776 | 182 / 20 |

Ring cleaning split seven original trials into two segments each. Therefore 213 is the actual Zarr episode count, but not a count of independent original demonstrations. Across tasks: 528 retained original trials, represented by 535 cleaned episodes/segments and 315,089 aligned frames.

| Task | Raw recording wall time, seconds | Raw recording minutes | Cleaned frame-equivalent minutes at 30 Hz | Sum of cleaned first-to-last sample spans, seconds |
|---|---:|---:|---:|---:|
| Ring/peg redo | 4165.858421381 | 69.431 | 66.181 | 3962.376612 |
| Cable stringing | 4010.861877480 | 66.848 | 66.216 | 3967.577513 |
| Whiteboard checkmark | 2666.125907632 | 44.435 | 42.653 | 2551.593176 |

Raw recording wall time sums episode stop-minus-start monotonic timestamps. It excludes time between recordings, rejected/uncommitted attempts, setup, and resets outside recorded episodes. Frame-equivalent duration is N/30, not measured elapsed time. Cleaned sample spans omit one frame interval per segment and preserve timestamp irregularities. Do not conflate these measures. A precise scale sentence is: “The cleaned datasets contain 315,089 aligned frames, equivalent to approximately 175 minutes at 30 Hz, from 528 retained original trials represented as 535 continuous episodes.”

## 2. Splits and a source-trial overlap

The four available training configs (ring Full, cable Full, whiteboard Full and Angles) explicitly save val_fraction=0.1, seed=0 and episode ID lists. Their lists exactly match the implementation's seeded shuffle and round(0.1*N); the whiteboard variants have identical lists. See episode_splits.json for complete train/validation lists and cleaned-to-source mappings, and episodes.csv for one row per cleaned segment.

Ring Angles and cable Angles folders contain only policy_epoch_0500.pt (plus an irrelevant macOS sidecar for cable). These checkpoint payloads save policy/optimizer/normalizer state, **not train/validation episode IDs or the full trainer configuration**. Their split sizes above are the expected split for these datasets, not independently verified saved lists for those two runs. Matching normalizer bounds support consistency but do not prove an identical split.

**Confirmed ring Full source-trial overlap:** cleaned episode 163 is validation and cleaned episode 164 is training; both come from original trial 170. These are distinct temporal segments, not duplicate frames, but the split is not independent at the original-trial level. This follows from splitting by cleaned episode IDs. A claim of source-trial-disjoint validation is false for the saved ring Full split. If ring Angles used the same split it inherits the issue. Fixing it would require grouping source trials and retraining; no split or checkpoint was changed in this audit.

## 3–4. What 40in means and whether d3 varied

The collection YAML uses 40in in output filenames, without a corresponding fixed-extension collection setting. **The exact physical meaning of that label is not documented in the inspected evidence**: it cannot establish a tape-measure reference, initial deployed length, nominal working length, or a motor calibration datum. Treat it as an experimental condition label until the collector defines it.

What is directly verified: data/state[:,2] is boom_extension_m and exactly equals data/joint_position_command[:,2]. The teleoperation loop integrates commanded joint velocities, maps extension through boom_motor_position(), and sends a separate can_position_target. Thus d3 is the commanded kinematic extension in meters, not an encoder/spool readout or independent physical length measurement. See src/giraf/teleop.py:90 and :197, and the Zarr state_semantics/state_fields attributes.

| Task | Cleaned d3 range, m | Median within-episode max-minus-min, m | Minimum within-episode range, m | Maximum within-episode range, m |
|---|---:|---:|---:|---:|
| Ring/peg redo | 0.593607–1.100254 | 0.229829 | 0.021062 | 0.437522 |
| Cable stringing | 0.386643–1.093218 | 0.442584 | 0.315118 | 0.624360 |
| Whiteboard checkmark | 0.767741–1.197526 | 0.085754 | 0.026631 | 0.273665 |

All 213/120/202 cleaned episodes vary by more than 1 mm. All raw episodes also vary. DP-Full's d3 channel is not constant. These limits also match the saved checkpoint normalizer bounds. Within-episode range is not cumulative distance traveled.

**Ablation interpretation:** DP-Angles excludes both boom extension AND all nine FK-derived pose fields, leaving five joint angles. It is not a pure one-channel d3 ablation. Its comparison tests removing explicit extension and FK-derived pose conditioning; the image stream can still convey length-related information.

## 5. Grasp and actual deployment settings

Every inspected named task rollout config has allow_grasp=true, action_scale=1.0, inference_steps=20, enforce_training_bounds=false. All six saved checkpoint policy configs have temporal_ensemble=true. Deployment runner replaces inference_steps while preserving the other policy fields, so the saved evidence supports temporal ensembling enabled and **20 DDIM steps during rollout**, despite 16 in the training checkpoint configuration. See deployment_runs/*/config.json, src/giraf/deployment/runner.py:85 and src/giraf/learning/diffusion.py:316.

Actual control logs contain commanded grasp values, not merely an enabled flag. The evidence supports policy-controlled gripper commands during clutch-enabled autonomous segments. It does not establish contact sensing or universal grasp success. Operators still activate/end policy segments using the clutch. Whiteboard Full has no identified saved rollout config here, so its deployment settings cannot be independently verified.

## 6. Cleaning actually performed

- Ring redo: remove failed source trials 40, 51, 52, 53, 54, 55, 74, 131, 134, 147, 159, 205. Prune selected tracking interruptions in source trials 50, 67, 106, 124, 170, 189, 208; split at each internal cut. Remove 5,144 failed-trial frames and 574 pause frames. Preserve other data and timestamps, including 951 invalid-alignment frames and an 11-frame tracking-off tail. This was selective interruption pruning, not a universal idle-motion filter. Source: data/ring_over_peg_redo/boom_40in_cleaned_notes.md and *_report.json.
- Cable: remove whole source episode 86, 1,104 frames; retain all other episodes with no additional filtering/trimming. The report explicitly says this is a curation decision, not a task-failure label, and not an independent visual success audit. Source: data/cable_stringing/boom_40in_cleaned_report.json.
- Whiteboard: discard whole source episodes 67, 68, 128, 135, 143, 150, 174, 175, 182; remove 3,088 frames. Zarr clean attributes explicitly record prune_inactive=false and exact preservation of retained arrays.

Training subsequently filters required invalid observation/action windows. The cleaned Zarr counts above precede that window selection. Do not describe all three datasets as uniformly idle-pruned or uniformly independently success-verified.

## 7–8. Camera and operator

All three datasets' saved IMU session metadata identify **OAK-D-S2-AF**, device ID 19443010A11E3C2E00. This is evidence from the actual collection sessions. Wrist placement is supplied by the existing experimental description; metadata here verifies product/device identity.

The inspected records do not identify the human demonstrator. Same operating-system account, machine, or camera cannot establish the same operator. “One operator per task” remains author-provided; whether the same person performed all three tasks remains unresolved.

## 9–10. Checkpoint/revision consistency and dirty trees

All six inspected epoch-0500 checkpoint policy configurations match after resolving historical missing fields to their source-code defaults, except state_input=full versus joint_angles. All use imu_input=none. Model tensor key/shape layouts agree across tasks within each variant.

Common saved policy settings: ResNet-18; observation/prediction/action horizons 2/16/8; diffusion steps 100; stored inference steps 16; vision features 128; U-Net down dimensions 64/128/256; timestep features 128; kernel size 5; crop fraction 0.9; color jitter 0.1; LR 1e-4; weight decay 1e-6; gradient clipping 1; EMA decay 0.999; eval seed 0. The four full trainer configs additionally verify 500 epochs, batch 256, warmup 500 steps, minimum LR ratio 0.5, seed 0, checkpoint interval 20, and validation fraction 0.1. The two checkpoint-only runs cannot establish every trainer-level setting from policy config alone.

Local W&B metadata verifies ring Full at 2ffc15c, cable Full at baa852d, and both whiteboard runs at 8f3860a. The asserted 3bd233e provenance for ring/cable Angles is not independently recorded in their local checkpoint-only artifacts. Whiteboard Angles is explicitly 8f3860a, not 3bd233e.

Committed diff: 2ffc15c→3bd233e adds the state selector; later commits add optional IMU support (off in these runs), remove unused simulation interfaces, and improve output/log handling. The core network, augmentation behavior, optimizer update, training/evaluation loops and scheduler behavior for these no-IMU configurations remain consistent in the inspected committed source. This is corroborated by saved policy configs and tensor shapes, but is not proof of a historically clean working tree.

No historical dirty flag, code snapshot, or saved diff/patch was found in these run artifacts. Current git status contains unrelated untracked files; current status cannot reconstruct past uncommitted changes. Therefore “identical code except the ablation, with no uncommitted changes” cannot be certified.

## 11. Why epoch 500?

Named saved rollouts point to policy_epoch_0500.pt. Available trainer configs requested a fixed 500-epoch budget. **No saved rationale for choosing final rather than validation-best was found.** best.pt is selected by validation action MSE, not validation diffusion loss.

| Run with metrics available | Best validation action-MSE epoch | Best action MSE | Epoch-500 action MSE |
|---|---:|---:|---:|
| Ring Full | 268 | 0.02498385 | 0.02513070 |
| Cable Full | 473 | 0.02801749 | 0.02835662 |
| Whiteboard Full | 158 | 0.04508258 | 0.05031040 |
| Whiteboard Angles | 108 | 0.04453932 | 0.04832074 |

A defensible factual sentence: “We evaluated the final checkpoint after 500 epochs rather than selecting checkpoints by validation performance.” If the authors confirm their motivation, add “to use a common training budget across variants.” Do not claim that intent was established by this audit. Ring/cable Angles metrics are not available locally; whiteboard Full's final checkpoint exists but its rollout usage is not locally verified.

## 12. Evaluation beyond training extension

Yes: actual recorded command traces extend beyond the training support; this conclusion does not rely solely on folder names.

| Task | DP-Angles 60in rollout d3 range, m | DP-Angles 80in rollout d3 range, m |
|---|---:|---:|
| Ring redo | 1.083517–1.584079 | 1.568830–2.120737 |
| Cable | 0.826027–1.792637 | 1.141462–2.120737 |
| Whiteboard | 1.307663–1.683690 | 1.835719–2.120737 |

Cable 60in redo is separately recorded at 0.854901–1.730642 m. Each named set has 20 inventoried episodes after excluding X_ and 2m_test prefixes and including supplemental episode directories. These inventory conventions are not a new adjudication of success, partial outcomes, or exclusion validity. The consolidated ring folder is a copy of the ring trials and was excluded from the range inventory to prevent double-counting.

All three 80in DP-Angles trace ranges are entirely above their respective training maxima. Whiteboard 60in is also entirely above its training maximum. Ring/cable 60in traces partly overlap training. The 40in label itself does not guarantee in-range execution: ring Full 40in reaches 1.186270 m, above its training maximum.

Saved DP-Full evaluation exists at 40in/60in for ring redo and cable. No named DP-Full 80in or whiteboard DP-Full rollout set was found. Thus the local evaluation matrix is incomplete for a full three-task × two-variant × three-length comparison. Extrapolation trials exist, but those trials alone do not establish superior generalization without outcome comparisons and the missing cells. Commanded lengths are not independently measured physical extension.

## Reproducible outputs

- audit_data.py: reads raw/cleaned low-dimensional arrays and saved split configs; writes dataset_summary.json, episode_splits.json, episodes.csv.
- audit_runs.py: inspects six checkpoint payloads and named rollout control logs; writes checkpoint_summary.json and rollout_summary.json.
- zip_array_verification.json: checks cleaned ZIP versus directory episode boundaries, state, action, timestamps and alignment validity; does not claim an exhaustive RGB/IMU archive comparison.

Open author questions: physical definition/reference for 40in; same versus different demonstrator; motivation for final-checkpoint selection; missing ring/cable Angles trainer artifacts and historical code diffs; missing evaluation records.
