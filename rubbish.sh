python data/plot_keypoints.py     --file DATASETS/UnifoLM_WBT/human_policy_rel/0.hdf5 --full_hand     --save_mp4 out_unitree_rel00.mp4
python data/plot_keypoints.py     --file DATASETS/UnifoLM_WBT/human_policy_rel/10.hdf5 --full_hand     --save_mp4 out_unitree_rel10.mp4



python /root/shengyin/human_policy/data/plot_keypoints.py --file /root/shengyin/DATASETS/UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher/hdf5_EE/0.hdf5 --full_hand --save_mp4 out_unitree_EE_0.mp4

python /root/shengyin/human_policy/data/plot_keypoints.py --file /root/shengyin/DATASETS/UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher/hdf5_XQY1/0.hdf5 --full_hand --save_mp4 out_unitree_XQY1_0.mp4


python /root/shengyin/human_policy/data/plot_keypoints.py --file /root/shengyin/DATASETS/UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher/hdf5_XQY2/0.hdf5 --full_hand --save_mp4 out_unitree_XQY2_V1.mp4

python /root/shengyin/human_policy/data/plot_keypoints.py --file /root/shengyin/DATASETS/UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher/hdf5_XQY_PH2D/0.hdf5 --full_hand --save_mp4 out_unitree_XQY_PH2D.mp4

python /root/shengyin/human_policy/data/plot_keypoints.py --file /root/shengyin/DATASETS/UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher/hdf5_XQY3/0.hdf5 --full_hand --save_mp4 out_unitree_XQY3_0.mp4

python /root/shengyin/human_policy/data/plot_keypoints.py     --file /root/shengyin/DATASETS/PH2D/402-pick_on_color_pad_right-2025_01_09-16_36_15/processed_episode_1.hdf5 --full_hand     --save_mp4 out_PH2D_new_axis.mp4

python /root/shengyin/human_policy/convert_to_hdf5.py --mode real --max-episodes 13



python data/plot_keypoints.py     --file /DATASETS/UnifoLM_WBT/G1_WBT_Inspire_Collect_Clothes_MainCamOnly/human_policy/0.hdf5 --full_hand     --save_mp4 out_unitree_inspirw00.mp4
python data/plot_keypoints.py     --file /DATASETS/UnifoLM_WBT/G1_WBT_Inspire_Collect_Clothes_MainCamOnly/human_policy/1.hdf5 --full_hand     --save_mp4 out_unitree_inspirw01.mp4
python data/plot_keypoints.py     --file /DATASETS/UnifoLM_WBT/G1_WBT_Inspire_Collect_Clothes_MainCamOnly/human_policy/2.hdf5 --full_hand     --save_mp4 out_unitree_inspirw02.mp4


python data/plot_keypoints.py     --file DATASETS/arker01/wholebody/data_egodex_processed/episode_2.hdf5 --full_hand     --save_mp4 out_egodex.mp4
python data/plot_keypoints.py     --file DATASETS/arker01/wholebody/data_whole_body/episode_20.hdf5 --full_hand     --save_mp4 out_visionpro1.mp4
python data/plot_keypoints.py     --file DATASETS/arker01/wholebody/PH2D/402-pick_on_color_pad_right-2025_01_09-16_36_15/processed_episode_1.hdf5 --full_hand     --save_mp4 out_visionpro.mp4


python data/plot_keypoints.py     --file DATASETS/arker01/wholebody/data_whole_body/episode_19.hdf5 --full_hand     --save_mp4 out_visionpro2.mp4

python visualize_to_mp4.py \
    --data_root DATASETS/Xperience/021c920e-ed68-47c8-9a94-fda10d4fe4cd/ep1 \
    --output DATASETS/Xperience/021c920e-ed68-47c8-9a94-fda10d4fe4cd/ep1/vis_skeleton.mp4 \
    2>&1


# First HDF5 from each dataset entry in data/act_fm_pretrain_convert_ego_ph2d.json
python /root/shengyin/human_policy/data/plot_keypoints.py --file /root/shengyin/DATASETS/human_policy/convert_ego/episode_0.hdf5 --full_hand --save_mp4 out_act_fm_train_00_human_policy_convert_ego_first.mp4
python /root/shengyin/human_policy/data/plot_keypoints.py --file /root/shengyin/DATASETS/PH2D/111-picking-colorful-toycube_2024-11-13_20-25-34/processed_episode_1.hdf5 --full_hand --save_mp4 out_act_fm_train_01_ph2d_111_picking_colorful_toycube_2024_11_13_20_25_34_first.mp4
python /root/shengyin/human_policy/data/plot_keypoints.py --file /root/shengyin/DATASETS/PH2D/112-picking-brownbox-human_2024-11-13_20-42-14/processed_episode_10.hdf5 --full_hand --save_mp4 out_act_fm_train_02_ph2d_112_picking_brownbox_human_2024_11_13_20_42_14_first.mp4
python /root/shengyin/human_policy/data/plot_keypoints.py --file /root/shengyin/DATASETS/PH2D/113-picking-blackcube-human_2024-11-13_22-09-05/processed_episode_0.hdf5 --full_hand --save_mp4 out_act_fm_train_03_ph2d_113_picking_blackcube_human_2024_11_13_22_09_05_first.mp4
python /root/shengyin/human_policy/data/plot_keypoints.py --file /root/shengyin/DATASETS/PH2D/402-pick_on_color_pad_right-2025_01_09-16_36_15/processed_episode_0.hdf5 --full_hand --save_mp4 out_act_fm_train_04_ph2d_402_pick_on_color_pad_right_2025_01_09_16_36_15_first.mp4
python /root/shengyin/human_policy/data/plot_keypoints.py --file /root/shengyin/DATASETS/PH2D/403-pick_on_color_pad_left-2025_01_09-16_58_04/processed_episode_1.hdf5 --full_hand --save_mp4 out_act_fm_train_05_ph2d_403_pick_on_color_pad_left_2025_01_09_16_58_04_first.mp4
python /root/shengyin/human_policy/data/plot_keypoints.py --file /root/shengyin/DATASETS/PH2D/404-pick_on_color_pad_right_far-2025_01_12-20_20_57/processed_episode_0.hdf5 --full_hand --save_mp4 out_act_fm_train_06_ph2d_404_pick_on_color_pad_right_far_2025_01_12_20_20_57_first.mp4
python /root/shengyin/human_policy/data/plot_keypoints.py --file /root/shengyin/DATASETS/PH2D/405-pick_on_color_pad_right_far_far-2025_01_13-19_29_04/processed_episode_0.hdf5 --full_hand --save_mp4 out_act_fm_train_07_ph2d_405_pick_on_color_pad_right_far_far_2025_01_13_19_29_04_first.mp4
python /root/shengyin/human_policy/data/plot_keypoints.py --file /root/shengyin/DATASETS/human_policy/convert_ego/episode_817.hdf5 --full_hand --save_mp4 out_act_fm_val_00_human_policy_convert_ego_first.mp4
python /root/shengyin/human_policy/data/plot_keypoints.py --file /root/shengyin/DATASETS/PH2D/401-picking-2024_11_12-22_39_57/processed_episode_0.hdf5 --full_hand --save_mp4 out_act_fm_val_01_ph2d_401_picking_2024_11_12_22_39_57_first.mp4
python /root/shengyin/human_policy/data/plot_keypoints.py --file /root/shengyin/DATASETS/PH2D/903-picking-val-2024_11_18-18_58_16/processed_episode_0.hdf5 --full_hand --save_mp4 out_act_fm_val_02_ph2d_903_picking_val_2024_11_18_18_58_16_first.mp4


# 预训练
# Resnet-18
conda activate human_policy && cd /root/shengyin/human_policy/hdt && \
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file ./1_gpu.yaml main.py   --batch_size 64   --num_epochs 100000   --lr 1e-4   --chunk_size 100   --seed 0   --exptid /root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_1e-4   --dataset_json_path /root/shengyin/human_policy/data/act_fm_pretrain_convert_ego_ph2d.json   --model_cfg_path /root/shengyin/human_policy/hdt/configs/models/act_flow.yaml   --base_dir /root/shengyin/DATASETS   --human_slow_down_factor 4   --no_wandb

conda activate human_policy && cd /root/shengyin/human_policy/hdt && \
CUDA_VISIBLE_DEVICES=2 accelerate launch --config_file ./1_gpu.yaml main.py   --batch_size 64   --num_epochs 100000   --lr 1e-5   --chunk_size 100   --seed 0   --exptid /root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_1e-5   --dataset_json_path /root/shengyin/human_policy/data/act_fm_pretrain_convert_ego_ph2d.json   --model_cfg_path /root/shengyin/human_policy/hdt/configs/models/act_flow.yaml   --base_dir /root/shengyin/DATASETS   --human_slow_down_factor 4   --no_wandb

# Dino-V2
conda activate human_policy && cd /root/shengyin/human_policy/hdt && \
CUDA_VISIBLE_DEVICES=3 accelerate launch --config_file ./1_gpu.yaml main.py   --batch_size 64   --num_epochs 100000   --lr 1e-5   --chunk_size 100   --seed 0   --exptid /root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_dinov2_1e-5   --dataset_json_path /root/shengyin/human_policy/data/act_fm_pretrain_convert_ego_ph2d.json   --model_cfg_path /root/shengyin/human_policy/hdt/configs/models/act_flow_dinov2.yaml   --base_dir /root/shengyin/DATASETS   --human_slow_down_factor 4   --no_wandb

# fine-tune

# Dino-V2
CUDA_VISIBLE_DEVICES=4 accelerate launch --config_file ./1_gpu.yaml main.py \
  --batch_size 64 \
  --num_epochs 50000 \
  --lr 5e-5 \
  --chunk_size 100 \
  --seed 0 \
  --exptid /root/shengyin/outputs/act_fm_finetune_whole_wbt \
  --dataset_json_path /root/shengyin/human_policy/data/act_fm_finetune_whole_wbt.json \
  --model_cfg_path /root/shengyin/human_policy/hdt/configs/models/act_flow_dinov2.yaml \
  --base_dir /root/shengyin/DATASETS \
  --human_slow_down_factor 4 \
  --load_pretrained_path /root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_dinov2_ckpt/policy_last.ckpt \
  --no_wandb

conda activate human_policy && cd /root/shengyin/human_policy/hdt && \
  CUDA_VISIBLE_DEVICES=5 accelerate launch --config_file ./1_gpu.yaml main.py \
  --batch_size 64 \
  --num_epochs 50000 \
  --lr 5e-6 \
  --chunk_size 100 \
  --seed 0 \
  --exptid /root/shengyin/outputs/act_fm_finetune_whole_wbt_5e-6 \
  --dataset_json_path /root/shengyin/human_policy/data/act_fm_finetune_whole_wbt.json \
  --model_cfg_path /root/shengyin/human_policy/hdt/configs/models/act_flow_dinov2.yaml \
  --base_dir /root/shengyin/DATASETS \
  --human_slow_down_factor 4 \
  --load_pretrained_path /root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_dinov2_ckpt/policy_last.ckpt \
  --no_wandb