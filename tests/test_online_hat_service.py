import numpy as np
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

from hdt import constants as C
from hdt.online_hat_service import (
    _act_config,
    _g1_wrist_to_hand_base,
    _hand_base_to_g1_wrist,
    _quat_wxyz_to_rotation_6d,
    _rotation_6d_to_quat_wxyz,
    action_chunk_to_targets,
    OnlineHATInputRecorder,
    OnlineHATPolicy,
    robot_state_to_policy_state,
    TimedObservationHistory,
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

    def test_history_config_reaches_the_built_model(self):
        config = _act_config(
            trainer_config(history={
                "enabled": True,
                "offsets": [5, 10],
                "token_pool": 2,
            }),
            100,
        )
        self.assertEqual(
            config["history_config"],
            {"offsets": [5, 10], "token_pool": 2},
        )
        self.assertNotIn("history_config", _act_config(trainer_config(), 100))

    def test_timed_history_selects_training_offsets_and_clamps_startup(self):
        history = TimedObservationHistory([5, 10], fps=30)
        samples = []
        for frame in range(13):
            samples.append(history.append(
                round(frame / 30 * 1e9),
                {"sequence_id": frame},
                np.full((1, 3, 2, 2), frame, dtype=np.uint8),
            ))
        self.assertEqual(
            [item["state"]["sequence_id"] for item in history.select(samples[-1])],
            [7, 2],
        )

        startup = TimedObservationHistory([5, 10], fps=30)
        first = startup.append(
            123, {"sequence_id": 9}, np.zeros((1, 3, 2, 2), dtype=np.uint8)
        )
        self.assertEqual(
            [item["state"]["sequence_id"] for item in startup.select(first)],
            [9, 9],
        )

    def test_history_infer_builds_training_tensor_layout(self):
        runner = OnlineHATPolicy.__new__(OnlineHATPolicy)
        runner.device = torch.device("cpu")
        runner.camera_names = ("top",)
        runner.history_offsets = (5, 10)
        runner.chunk_size = 3
        runner.is_torchscript = False
        runner.policy_class = "ACT"
        runner.config = {"model": {"use_language_conditioning": False}}
        runner.visual_preprocessor = lambda value: torch.from_numpy(value).float()
        runner.stats = {
            "action_mean": np.zeros(128, dtype=np.float32),
            "action_std": np.ones(128, dtype=np.float32),
        }
        runner.policy = Mock(return_value=torch.zeros((1, 3, 128)))

        output = runner.infer(
            np.zeros((1, 3, 2, 2), dtype=np.uint8),
            None,
            normalized_state=np.zeros(128, dtype=np.float32),
            history_images=np.zeros((2, 1, 3, 2, 2), dtype=np.uint8),
            normalized_history_states=np.zeros((2, 128), dtype=np.float32),
        )

        self.assertEqual(output.shape, (3, 128))
        conditioning = runner.policy.call_args.kwargs["conditioning_dict"]
        self.assertEqual(conditioning["hist_image"].shape, (1, 2, 1, 3, 2, 2))
        self.assertEqual(conditioning["hist_qpos"].shape, (1, 2, 128))

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
        # Without a checkpoint mean there is nothing to canonicalize against,
        # so the measured representative is passed through untouched.
        np.testing.assert_allclose(
            robot_state_to_policy_state(negative)[103:107],
            [-0.9238795, 0.0, 0.0, -0.3826834],
            atol=1e-7,
        )
        # b10/c30 all train on the negative-w representative: keep it.
        np.testing.assert_allclose(
            robot_state_to_policy_state(negative, root_quat_mean_w=-0.99)[103:107],
            [-0.9238795, 0.0, 0.0, -0.3826834],
            atol=1e-7,
        )
        # A checkpoint trained on the positive-w representative flips it.
        output = robot_state_to_policy_state(negative, root_quat_mean_w=0.99)
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

    def test_initial_heading_frame_matches_training_convention(self):
        # The training HDF5s declare coordinate_convention =
        # initial_heading_frame_x_forward_y_left_z_up, i.e. every episode's
        # opening root yaw is ~0.  A session whose whole world is yawed must
        # therefore produce the same policy state as the unrotated session,
        # once the opening heading is supplied.  Frame-local channels (finger
        # keypoints, joint angles) must not change at all.
        from hdt.online_hat_service import (
            _quat_mul_wxyz,
            _quat_rotate_wxyz,
            HeadOrigin,
        )

        yaw = np.deg2rad(-37.0)
        heading = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        base = state()
        rotated = state()
        for key in ("root", "head", "left_wrist", "right_wrist"):
            position = np.asarray(base[f"{key}_pos_world"], dtype=np.float64)
            quaternion = np.asarray(base[f"{key}_quat_world_wxyz"], dtype=np.float64)
            rotated[f"{key}_pos_world"] = _quat_rotate_wxyz(heading, position)
            rotated[f"{key}_quat_world_wxyz"] = _quat_mul_wxyz(heading, quaternion)
        origin = np.array([0.02, -0.01, 1.19])
        np.testing.assert_allclose(
            robot_state_to_policy_state(
                rotated,
                origin=_quat_rotate_wxyz(heading, origin),
                heading_quat_wxyz=heading,
            ),
            robot_state_to_policy_state(base, origin=origin),
            atol=1e-6,
        )

        # HeadOrigin locks both halves of the episode frame from the same
        # opening window; the live root stream carries w < 0, and heading
        # extraction must be invariant to that representative.
        head_origin = HeadOrigin(2)
        for _ in range(2):
            locked_origin, locked_heading = head_origin.observe(
                rotated["head_pos_world"],
                -np.asarray(rotated["root_quat_world_wxyz"]),
            )
        self.assertTrue(head_origin.locked)
        np.testing.assert_allclose(locked_origin, rotated["head_pos_world"], atol=1e-6)
        locked_yaw = 2.0 * np.arctan2(locked_heading[3], locked_heading[0])
        self.assertAlmostEqual(np.degrees(locked_yaw), -37.0, places=4)


if __name__ == "__main__":
    unittest.main()
