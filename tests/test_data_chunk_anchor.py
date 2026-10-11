"""Check that lagged encoder states decode to the unchanged demonstration."""
import importlib.util
import os
import unittest
import numpy as np
from hdt.action_representation import POSE_BLOCKS, to_absolute_chunk

if os.environ.get('HDT_DATA_UTILS_TEST_FILE'):
    spec = importlib.util.spec_from_file_location('hdt.data_utils_anchor_test', os.environ['HDT_DATA_UTILS_TEST_FILE'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    EpisodicDataset = module.EpisodicDataset
else:
    from hdt.data_utils_hdt import EpisodicDataset


class RelativeLagAnchorTests(unittest.TestCase):
    def test_augmented_translation_rotation_and_fingers_preserve_future(self):
        reader = EpisodicDataset.__new__(EpisodicDataset)
        reader.source_embodiment_list = ['dex5']
        reader.episode_len_list = [6]
        reader.episode_frame_indices = [np.arange(6)]
        reader._raw_index = lambda index, ts: ts
        reader.chunk_size = 3
        reader.action_representation = 'relative_chunk'
        states = np.zeros((6, 128), np.float32)
        for ids in POSE_BLOCKS:
            states[:, ids[3:9]] = [1, 0, 0, 0, 1, 0]
        actions = states.copy()
        actions[:, 30] = np.arange(6) * .02
        observed = states[1].copy()
        observed[30] -= .06
        observed[33:39] = [0, -1, 0, 1, 0, 0]
        observed[43:58] = -.01
        before = observed.copy()
        relative, anchor = reader._read_action_chunk(
            0, 1, states, actions, anchor_override=observed)
        decoded = to_absolute_chunk(anchor, relative)
        np.testing.assert_allclose(decoded, actions[2:5], atol=1e-7)
        np.testing.assert_array_equal(observed, before)
        # Previously zero training loss still decoded 6cm away from the demo.
        clean_label, _ = reader._read_action_chunk(0, 1, states, actions)
        wrong = to_absolute_chunk(observed, clean_label)
        np.testing.assert_allclose(wrong[:, 30] - actions[2:5, 30], -.06, atol=1e-7)

    def test_original_absolute_label_is_independent_of_augmented_anchor(self):
        reader = EpisodicDataset.__new__(EpisodicDataset)
        reader.source_embodiment_list = ['dex5']
        reader.episode_len_list = [6]
        reader.episode_frame_indices = [np.arange(6)]
        reader._raw_index = lambda index, ts: ts
        reader.chunk_size = 3
        reader.action_representation = 'original'
        states = np.zeros((6, 128), np.float32)
        actions = np.arange(6 * 128, dtype=np.float32).reshape(6, 128)
        chunk, _ = reader._read_action_chunk(0, 1, states, actions, anchor_override=np.ones(128))
        np.testing.assert_array_equal(chunk, actions[1:4])


if __name__ == '__main__':
    unittest.main()
