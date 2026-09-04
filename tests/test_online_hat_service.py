import numpy as np
import json
import tempfile
import unittest
from pathlib import Path

from hdt import constants as C
from hdt.online_hat_service import (
    _act_config,
    _g1_wrist_to_hand_base,
    _hand_base_to_g1_wrist,
    _quat_wxyz_to_rotation_6d,
    _rotation_6d_to_quat_wxyz,
    action_chunk_to_targets,
    OnlineHATInputRecorder,
    robot_state_to_policy_state,
)


def trainer_config(**model_extra):
    model = {
        "enc_layers": 4, "dec_layers": 7, "nheads": 8, "hidden_dim": 512,
        "kl_weight": 10, "dim_feedforward": 3200, "lr_backbone": "1e-5",
        "backbone": "resnet18", "image_feature_strategy": "ACT_linear",
    }
    model.update(model_extra)
    return {
        "common": {"policy_class": "ACT", "state_dim": 128, "action_dim": 128,
                   "camera_names": ["top"]},
        "model": model,
    }


def state():
    from hdt.online_hat_service import _install_protocol, DEFAULT_SCALEBRIDGE_ROOT
    make_hat_chunk, monotonic_ns, validate, Publisher, Subscriber = _install_protocol(DEFAULT_SCALEBRIDGE_ROOT)
    from scalebridge.online.protocol import make_robot_state
    return make_robot_state(
        sequence_id=4,
        timestamp_ns=1,
        root_pos_world=[0.0, 0.0, 0.8],
        root_quat_world_wxyz=[1.0, 0.0, 0.0, 0.0],
        head_pos_world=[0.0, 0.0, 1.2],
        head_quat_world_wxyz=[1.0, 0.0, 0.0, 0.0],
        left_wrist_pos_world=[0.1, 0.2, 1.0],
        left_wrist_quat_world_wxyz=[1.0, 0.0, 0.0, 0.0],
        right_wrist_pos_world=[0.1, -0.2, 1.0],
        right_wrist_quat_world_wxyz=[1.0, 0.0, 0.0, 0.0],
        body_q19=np.arange(19) * 0.01,
        left_arm_q=np.arange(7),
        left_hand_q=np.arange(6) + 10,
        left_hand_keypoints_local=np.arange(18).reshape(6, 3) * 0.01,
        right_arm_q=np.arange(7) + 20,
        right_hand_q=np.arange(6) + 30,
        right_hand_keypoints_local=np.arange(18).reshape(6, 3) * -0.01,
    )


class OnlineHATServiceTest(unittest.TestCase):
    def test_lossless_inference_record_links_image_state_and_action(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            recorder = OnlineHATInputRecorder(
                run_dir, {"camera_names": ["top"], "checkpoint": "test.ckpt"}
            )
            images = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(1, 3, 2, 4)
            raw = np.arange(128, dtype=np.float32)
            normalized = raw / 10.0
            action = np.arange(3 * 128, dtype=np.float32).reshape(3, 128)
            recorder.record(
                sequence_id=7,
                source_state=state(),
                origin=np.array([0.1, 0.2, 1.3], dtype=np.float32),
                images=images,
                policy_state_raw=raw,
                policy_state_normalized=normalized,
                action=action,
                image_captured_ns=10,
                inference_started_ns=20,
                inference_finished_ns=30,
                chunk_generated_at_ns=31,
            )
            recorder.close()

            manifest = json.loads((run_dir / "manifest.json").read_text())
            record = json.loads(
                (run_dir / "inference_inputs.jsonl").read_text().strip()
            )
            self.assertEqual(manifest["format_version"], 1)
            self.assertEqual(record["sequence_id"], 7)
            self.assertEqual(record["source_state_sequence_id"], 4)
            np.testing.assert_array_equal(
                np.load(run_dir / record["images_file"])["images"], images
            )
            np.testing.assert_array_equal(
                np.load(run_dir / record["actions_file"]), action
            )
            np.testing.assert_allclose(record["policy_state_raw"], raw)
            np.testing.assert_allclose(
                record["policy_state_normalized"], normalized
            )

    def test_state_ablation_range_reaches_the_built_model(self):
        # The mask is a checkpoint buffer, so dropping it here loads cleanly
        # (strict=False) and then feeds the model the full state it never saw.
        # That failure is silent and looks like a regression, hence this test.
        config = _act_config(trainer_config(zero_state_dims="98:128"), 100)
        self.assertEqual(config["zero_state_dims"], (98, 128))
        self.assertEqual(
            _act_config(trainer_config(zero_state_dims=[98, 128]), 100)["zero_state_dims"],
            (98, 128),
        )
        self.assertNotIn("zero_state_dims", _act_config(trainer_config(), 100))

    def test_degenerate_rotation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "degenerate"):
            _rotation_6d_to_quat_wxyz([[0, 0, 0, 0, 0, 0]])

    def test_robot_state_maps_qpos_and_current_global_poses(self):
        output = robot_state_to_policy_state(state())
        expected = np.concatenate((
            [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
            np.arange(19) * 0.01,
        ))
        np.testing.assert_allclose(output[C.QPOS_INDICES], expected)
        identity6 = [1, 0, 0, 0, 1, 0]
        np.testing.assert_allclose(output[C.OUTPUT_HEAD_EEF[:3]], [0, 0, 1.2])
        np.testing.assert_allclose(output[C.OUTPUT_HEAD_EEF[3:]], identity6)
        np.testing.assert_allclose(output[C.OUTPUT_LEFT_EEF[:3]], [0.1, 0.2, 1.0])
        np.testing.assert_allclose(output[C.OUTPUT_RIGHT_EEF[:3]], [0.1, -0.2, 1.0])
        np.testing.assert_allclose(output[C.OUTPUT_WAIST[:3]], [0, 0, 0.8])
        np.testing.assert_allclose(
            output[C.OUTPUT_LEFT_KEYPOINTS], np.arange(18) * 0.01
        )
        np.testing.assert_allclose(
            output[C.OUTPUT_RIGHT_KEYPOINTS], np.arange(18) * -0.01
        )
        pose_indices = np.concatenate((
            C.OUTPUT_HEAD_EEF, C.OUTPUT_LEFT_EEF, C.OUTPUT_RIGHT_EEF,
            C.OUTPUT_WAIST, C.OUTPUT_LEFT_KEYPOINTS,
            C.OUTPUT_RIGHT_KEYPOINTS, C.QPOS_INDICES,
        ))
        untouched = np.delete(output, pose_indices)
        np.testing.assert_allclose(untouched, 0.0)

    def test_root_quaternion_tail_uses_training_sign(self):
        negative = state()
        negative["root_quat_world_wxyz"] = [
            -0.9238795, 0.0, 0.0, -0.3826834
        ]
        output = robot_state_to_policy_state(negative)
        np.testing.assert_allclose(
            output[103:107], [0.9238795, 0.0, 0.0, 0.3826834], atol=1e-7
        )
        # The sign-invariant 6D waist pose must still describe the same rotation.
        np.testing.assert_allclose(
            output[C.OUTPUT_WAIST[3:]],
            _quat_wxyz_to_rotation_6d(
                np.array([0.9238795, 0.0, 0.0, 0.3826834])
            ),
            atol=1e-6,
        )

    def test_quaternion_and_rotation_6d_round_trip(self):
        quat = np.array([0.9238795, 0.0, 0.0, 0.3826834], dtype=np.float32)
        recovered = _rotation_6d_to_quat_wxyz(
            _quat_wxyz_to_rotation_6d(quat)[None]
        )[0]
        self.assertAlmostEqual(abs(float(np.dot(quat, recovered))), 1.0, places=5)

    def test_inspire_hand_base_wrist_conversion_round_trip(self):
        poses = {
            "left": (
                np.array([0.1, 0.2, 1.0]),
                np.array([1.0, 0.0, 0.0, 0.0]),
            ),
            "right": (
                np.array([0.1, -0.2, 1.0]),
                np.array([0.9238795, 0.0, 0.3826834, 0.0]),
            ),
        }
        for side, (wrist_pos, wrist_quat) in poses.items():
            hand_pos, hand_quat = _g1_wrist_to_hand_base(
                wrist_pos, wrist_quat, side
            )
            recovered_pos, recovered_quat = _hand_base_to_g1_wrist(
                hand_pos, hand_quat, side
            )
            np.testing.assert_allclose(recovered_pos, wrist_pos, atol=1e-7)
            self.assertAlmostEqual(
                abs(float(np.dot(recovered_quat, wrist_quat))), 1.0, places=6
            )

    def test_action_chunk_extracts_palm_free_five_tip_order(self):
        action = np.zeros((64, 128), dtype=np.float32)
        # Valid identity 6D rows for every emitted pose.
        identity6 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        for indices in (
            C.OUTPUT_LEFT_EEF,
            C.OUTPUT_RIGHT_EEF,
            C.OUTPUT_HEAD_EEF,
            C.OUTPUT_WAIST,
        ):
            action[:, indices[3:9]] = identity6
        left = np.arange(18, dtype=np.float32).reshape(6, 3)
        right = left + 100
        action[:, C.OUTPUT_LEFT_KEYPOINTS] = left.reshape(-1)
        action[:, C.OUTPUT_RIGHT_KEYPOINTS] = right.reshape(-1)
        origin = np.array([-0.078, -0.001, 1.272], dtype=np.float32)
        result = action_chunk_to_targets(
            action, state(), 30.0, 9, 123,
            position_origin_world=origin,
        )
        self.assertEqual(result["frame_count"], 64)
        self.assertEqual(result["source_state_sequence_id"], 4)
        np.testing.assert_allclose(result["position_origin_world"], origin)
        np.testing.assert_allclose(result["left_fingertip_local"][0], left[1:])
        np.testing.assert_allclose(result["right_fingertip_local"][0], right[1:])


if __name__ == "__main__":
    unittest.main()
