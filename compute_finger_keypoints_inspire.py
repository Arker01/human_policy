"""
Compatibility wrapper.

Use compute_finger_keypoints.py directly for new runs:

python3 human_policy/compute_finger_keypoints.py \
  --dataset DATASETS/UnifoLM_WBT/G1_WBT_Inspire_Collect_Clothes_MainCamOnly/G1_WB_Dex5_Collect_Clothes
"""

import os
import sys

from compute_finger_keypoints import main


DEFAULT_INSPIRE_DATASET = (
    "DATASETS/UnifoLM_WBT/"
    "G1_WBT_Inspire_Collect_Clothes_MainCamOnly/"
    "G1_WB_Dex5_Collect_Clothes"
)


if __name__ == "__main__":
    if "--dataset" not in sys.argv:
        sys.argv.extend(["--dataset", DEFAULT_INSPIRE_DATASET])
    if "--hand-type" not in sys.argv:
        sys.argv.extend(["--hand-type", "inspire"])
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    main()
