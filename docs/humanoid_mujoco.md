# Humanoid Playground in Mujoco

We provide an interface to play with the humanoid in Mujoco. During the development
of this project, we use this script to verify the rigid transformations of different humanoid
data and human data are correctly aligned to the same coordinate frame.

There are two usage of this script:

1. Replay a human-centric trajectories from human/humanoid recordings. For example,

```bash
cd cet/
python mujoco_rollout_replay.py  --hdf_file_path ../data/recordings/processed/1061new_sim_pepsi_grasp_h1_2_inspire-2025_02_11-22_20_48/processed_episode_0.hdf5 --tasktype h1_only
```

2. Policy rollout in Mujoco. We collected some [humanoid demonstrations in Mujoco](https://huggingface.co/datasets/RogerQi/PH2D/tree/main/1061new_sim_pepsi_grasp_h1_2_inspire-2025_02_11-22_20_48). You can find a toy example pre-trained weights in [OneDrive](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/riqiu_ucsd_edu/EWa4xbhxJmNAnfwTbQt-vpsBxeUGzkoiM3xlxOalsnlwhg?e=Q2bneA). Note that the policy was trained with fine-tuned ResNet and used for illustration purpose of input/output representation handling only.

```bash
cd cet/
# Download the zip from OneDrive link above
unzip ./act_resnet_100cs.zip

python mujoco_rollout_replay.py  --hdf_file_path ../data/recordings/processed/1061new_sim_pepsi_grasp_h1_2_inspire-2025_02_11-22_20_48/processed_episode_0.hdf5 --norm_stats_path ./act_resnet_100cs/dataset_stats.pkl  --plot --model_path ./act_resnet_100cs/policy_traced.pt  --tasktype pepsi --chunk_size 100 --policy_config_path ../hdt/configs/models/act_resnet.yaml
```
