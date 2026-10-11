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
    ScheduledAnchorPlanner,
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


class EncoderLeftHandStateTests(unittest.TestCase):
    def test_train_mean_changes_only_encoder_left_points(self):
        policy = OnlineHATPolicy.__new__(OnlineHATPolicy)
        policy.left_hand_state_mode = "train_mean"
        policy.stats = {"qpos_mean": np.linspace(-1, 1, 128),
                        "qpos_std": np.full(128, .1)}
        raw = np.linspace(0, 2, 128).astype(np.float32)
        before = raw.copy()
        normalized = policy.normalize_policy_state(raw)
        expected = (raw - policy.stats["qpos_mean"]) / (.1 + 1e-6)
        expected[C.OUTPUT_LEFT_KEYPOINTS] = 0
        np.testing.assert_array_equal(raw, before)
        np.testing.assert_allclose(normalized, expected)
        np.testing.assert_allclose(normalized[C.OUTPUT_RIGHT_KEYPOINTS],
                                   np.full(18, 1 / (.1 + 1e-6)))

    def test_actual_retains_measured_left_points(self):
        policy = OnlineHATPolicy.__new__(OnlineHATPolicy)
        policy.left_hand_state_mode = "actual"
        policy.stats = {"qpos_mean": np.zeros(128), "qpos_std": np.ones(128)}
        raw = np.arange(128, dtype=np.float32)
        np.testing.assert_allclose(policy.normalize_policy_state(raw),
                                   raw.astype(np.float64) / (1 + 1e-6))


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
    def test_actual_anchor_discards_plan_bias_and_snapshots_observation(self):
        planner = ScheduledAnchorPlanner(30, anchor_mode="actual")
        observation = np.arange(128, dtype=np.float32)
        # Even a pre-existing plan must not contribute position or rotation.
        planner.previous_action = np.full((100, 128), 1000, dtype=np.float32)
        planner.previous_source_timestamp_ns = 1_000_000_000
        anchor, info = planner.select(observation, 1_333_333_333)
        np.testing.assert_array_equal(anchor, observation)
        self.assertEqual(info["mode"], "actual_observation")
        self.assertEqual(info["source_state_timestamp_ns"], 1_333_333_333)
        observation[:] = -1000  # A subsequent observation cannot mutate it.
        np.testing.assert_array_equal(anchor, np.arange(128, dtype=np.float32))
        planner.update(np.ones((100, 128), dtype=np.float32), 1_333_333_333)
        self.assertIsNone(planner.previous_action)
        next_observation = np.full(128, .25, dtype=np.float32)
        next_anchor, _ = planner.select(next_observation, 1_666_666_667)
        np.testing.assert_array_equal(next_anchor, next_observation)

    def test_actual_decode_uses_input_pose_when_newer_state_arrives_during_inference(self):
        from scipy.spatial.transform import Rotation
        from hdt.action_representation import POSE_BLOCKS

        measured = state()
        measured["timestamp_ns"] = 1_000_000_000
        raw = robot_state_to_policy_state(measured)
        planner = ScheduledAnchorPlanner(30, anchor_mode="actual")
        anchor, _ = planner.select(raw, measured["timestamp_ns"])
        relative = np.zeros((100, 128), dtype=np.float32)
        for indices in POSE_BLOCKS:
            relative[:, indices[3:9]] = [1, 0, 0, 0, 1, 0]
        relative[:, C.OUTPUT_RIGHT_EEF[:3]] = [.02, -.01, .03]
        delta = Rotation.from_euler("z", 15, degrees=True)
        relative[:, C.OUTPUT_RIGHT_EEF[3:9]] = delta.as_matrix()[:2].reshape(6)

        policy = OnlineHATPolicy.__new__(OnlineHATPolicy)
        policy.device = torch.device("cpu")
        policy.camera_names = ["top"]
        policy.history_offsets = []
        policy.visual_preprocessor = torch.as_tensor
        policy.is_torchscript = False
        policy.policy_class = "ACT"
        policy.config = {"model": {}}
        policy.chunk_size = 100
        policy.action_representation = "relative_chunk"
        policy.stats = {"action_std": np.ones(128), "action_mean": np.zeros(128)}

        def predict(image, qpos, conditioning_dict):
            np.testing.assert_array_equal(qpos.numpy()[0], raw)
            # Simulate a newer measured pose arriving before output decoding.
            measured["timestamp_ns"] = 1_200_000_000
            measured["right_wrist_pos_world"] = [9., 8., 7.]
            measured["right_wrist_quat_world_wxyz"] = [0., 0., 0., 1.]
            return torch.from_numpy(relative[None])

        policy.policy = predict
        output = policy.infer(
            np.zeros((1, 3, 2, 2), dtype=np.uint8), measured,
            normalized_state=raw.copy(), action_anchor=anchor,
        )
        np.testing.assert_allclose(output[0, C.OUTPUT_RIGHT_EEF[:3]], [.12, -.21, 1.03], atol=1e-7)
        rotation = _rotation_6d_to_quat_wxyz(output[0, C.OUTPUT_RIGHT_EEF[3:9]])
        expected = delta.as_quat()[[3, 0, 1, 2]]
        self.assertAlmostEqual(abs(float(np.dot(rotation, expected))), 1.0, places=6)

    def test_humi_world_frame_restores_measured_wrists_with_nonzero_heading(self):
        from scalebridge.online.reference import apply_alignment
        from scipy.spatial.transform import Rotation

        measured = state()
        origin = np.array([.37, -.18, 1.23], dtype=np.float32)
        heading = Rotation.from_euler("z", -43, degrees=True).as_quat()[[3, 0, 1, 2]]
        for side in ("left", "right"):
            measured[side + "_wrist_quat_world_wxyz"] = Rotation.from_euler(
                "xyz", [17, -26, 61], degrees=True
            ).as_quat()[[3, 0, 1, 2]]
        actual = robot_state_to_policy_state(
            measured, origin=origin, wrist_conversion="inspire_hand_base",
            heading_quat_wxyz=heading,
        )
        chunk = action_chunk_to_targets(
            np.repeat(actual[None], 100, axis=0), measured, 30, 0, 1,
            wrist_conversion="inspire_hand_base", position_origin_world=origin,
            action_representation="relative_chunk", frame_heading_world_wxyz=heading,
        )
        world = apply_alignment(chunk, chunk["position_origin_world"], chunk["frame_heading_world_wxyz"])
        for side in ("left", "right"):
            np.testing.assert_allclose(world[side + "_wrist_pos"][0], measured[side + "_wrist_pos_world"], atol=2e-7)
            q = world[side + "_wrist_quat_wxyz"][0]
            expected = measured[side + "_wrist_quat_world_wxyz"]
            self.assertAlmostEqual(abs(float(np.dot(q, expected))), 1.0, places=6)

    def test_humi_waiting_for_r1_never_accumulates_a_schedule(self):
        planner = ScheduledAnchorPlanner(fps=30.0)
        actual = np.zeros(128, dtype=np.float32)
        for i in range(30):
            timestamp = 1_000_000_000 + round(i / 3 * 1e9)
            anchor, info = planner.select(
                actual, timestamp, plan_active=False, plan_epoch=1
            )
            self.assertEqual(info["mode"], "actual_hold")
            np.testing.assert_array_equal(anchor, actual)
            prediction = np.repeat(anchor[None], 100, axis=0)
            prediction[:, C.OUTPUT_RIGHT_EEF[2]] += 0.02
            planner.update(prediction, timestamp, plan_active=False)
        self.assertIsNone(planner.previous_action)
        anchor, info = planner.select(actual, timestamp + 333_333_333,
                                      plan_active=True, plan_epoch=2)
        self.assertEqual(info["mode"], "actual_first")
        np.testing.assert_array_equal(anchor, actual)

    def test_humi_running_preserves_scheduled_pose_despite_actual_lag(self):
        planner = ScheduledAnchorPlanner(fps=30.0)
        actual = np.zeros(128, dtype=np.float32)
        planner.select(actual, 1_000_000_000, plan_active=True, plan_epoch=2)
        plan = np.repeat(actual[None], 100, axis=0)
        plan[:, C.OUTPUT_RIGHT_EEF[2]] = 0.3
        plan[:, C.OUTPUT_RIGHT_EEF[3:9]] = [0, -1, 0, 1, 0, 0]
        planner.update(plan, 1_000_000_000, plan_active=True)
        anchor, info = planner.select(actual, 1_333_333_333,
                                      plan_active=True, plan_epoch=2)
        self.assertEqual(info["mode"], "scheduled")
        np.testing.assert_array_equal(anchor, plan[9])

    def test_humi_new_start_epoch_discards_previous_schedule(self):
        planner = ScheduledAnchorPlanner(fps=30.0)
        actual = np.zeros(128, dtype=np.float32)
        planner.select(actual, 1_000_000_000, plan_epoch=1)
        planner.update(np.ones((100, 128), dtype=np.float32), 1_000_000_000)
        # Even if HAT missed every hold state, the next R1 epoch resets it.
        anchor, info = planner.select(actual, 1_333_333_333, plan_epoch=2)
        self.assertEqual(info["mode"], "actual_first")
        np.testing.assert_array_equal(anchor, actual)

    def test_humi_pause_clears_schedule_before_resume(self):
        planner = ScheduledAnchorPlanner(fps=30.0)
        actual = np.zeros(128, dtype=np.float32)
        planner.update(np.ones((100, 128), dtype=np.float32), 1_000_000_000)
        planner.select(actual, 1_333_333_333, plan_active=False)
        anchor, info = planner.select(actual, 1_666_666_666, plan_active=True)
        self.assertEqual(info["mode"], "actual_first")
        np.testing.assert_array_equal(anchor, actual)

    def test_scheduled_anchor_uses_robot_state_time_not_fixed_replan_row(self):
        planner = ScheduledAnchorPlanner(fps=30.0)
        actual = np.zeros(128, dtype=np.float32)
        action = np.repeat(actual[None], 20, axis=0)
        action[:, C.OUTPUT_RIGHT_EEF[0]] = np.arange(20, dtype=np.float32)

        first, first_info = planner.select(actual, 1_000_000_000)
        np.testing.assert_array_equal(first, actual)
        self.assertEqual(first_info["mode"], "actual_first")
        planner.update(action, 1_000_000_000)

        # 11 source frames elapsed, so row 10 (t+11) is the scheduled pose.
        anchor, info = planner.select(actual, 1_366_666_667)
        self.assertEqual(info["mode"], "scheduled")
        self.assertEqual(info["scheduled_row"], 10)
        self.assertEqual(anchor[C.OUTPUT_RIGHT_EEF[0]], 10.0)

    def test_scheduled_anchor_safety_reanchors_to_actual(self):
        planner = ScheduledAnchorPlanner(fps=30.0, max_position_error_m=0.10)
        actual = np.zeros(128, dtype=np.float32)
        action = np.repeat(actual[None], 20, axis=0)
        action[:, C.OUTPUT_RIGHT_EEF[0]] = 0.11
        planner.update(action, 1_000_000_000)

        anchor, info = planner.select(actual, 1_333_333_333)
        np.testing.assert_array_equal(anchor, actual)
        self.assertEqual(info["mode"], "actual_safety_reanchor")
        self.assertAlmostEqual(info["max_position_error_m"], 0.11, places=6)
        self.assertEqual(info["scheduled_row"], 9)

    def test_scheduled_anchor_timing_gap_reanchors_to_actual(self):
        planner = ScheduledAnchorPlanner(fps=30.0)
        actual = np.ones(128, dtype=np.float32)
        planner.update(np.zeros((5, 128), dtype=np.float32), 1_000_000_000)

        anchor, info = planner.select(actual, 2_000_000_000)
        np.testing.assert_array_equal(anchor, actual)
        self.assertEqual(info["mode"], "actual_timing_reanchor")

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
                action_anchor=raw,
                action_anchor_info={"mode": "actual_first"},
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
            np.testing.assert_allclose(record["action_anchor"], raw)
            self.assertEqual(record["action_anchor_info"]["mode"], "actual_first")

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
            action_representation="relative_chunk",
        )
        self.assertEqual(result["frame_count"], 64)
        self.assertEqual(result["source_state_sequence_id"], 4)
        self.assertEqual(result["action_representation"], "relative_chunk")
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


class RealAnchorTests(unittest.TestCase):
    """Check same-time continuity, bias cancellation, and retained plan progress."""

    @staticmethod
    def valid_state():
        """Build a state with valid published pose rotations and unused zero neck."""
        from hdt.action_representation import POSE_BLOCKS
        value = np.zeros(128, dtype=np.float32)
        for ids in POSE_BLOCKS:
            if not np.array_equal(ids, C.OUTPUT_NECK):
                value[ids[3:9]] = [1, 0, 0, 0, 1, 0]
        return value

    def test_constant_pose_and_finger_bias_does_not_integrate(self):
        """Pose bias is cancelled; local finger predictions retain their bias."""
        from hdt.action_representation import POSE_BLOCKS, POINT_BLOCKS
        actual = self.valid_state()
        planner = ScheduledAnchorPlanner(30, anchor_mode="real")
        prediction = np.repeat(actual[None], 100, axis=0)
        for ids in POSE_BLOCKS:
            prediction[:, ids[:3]] += .0067
        for ids in POINT_BLOCKS:
            prediction[:, ids] += .0067
        expected = np.repeat(actual[None], 100, axis=0)
        for ids in POINT_BLOCKS:
            expected[:, ids] = prediction[:, ids]
        for i in range(63):
            timestamp = 1_000_000_000 + i * 333_333_333
            anchor, _ = planner.select(actual, timestamp, plan_epoch=2)
            np.testing.assert_array_equal(anchor, actual)
            chunk, _, _ = planner.align_real(prediction, actual, timestamp,
                                              timestamp + 50_000_000)
            np.testing.assert_allclose(chunk, expected, atol=1e-7)
            planner.update(chunk, timestamp)

    def test_stalled_fingers_do_not_accumulate_future_shape_changes(self):
        """A moving finger forecast stays bounded when measured fingers stall."""
        from hdt.action_representation import POINT_BLOCKS
        actual = self.valid_state()
        prediction = np.repeat(actual[None], 100, axis=0)
        for ids in POINT_BLOCKS:
            prediction[:, ids] = np.arange(1, 101)[:, None] * .0005
        for mode in ('real', 'real_wrist'):
            with self.subTest(mode=mode):
                planner = ScheduledAnchorPlanner(30, anchor_mode=mode)
                for step in range(63):
                    timestamp = 1_000_000_000 + step * 333_333_333
                    planner.select(actual, timestamp, plan_epoch=2)
                    output, target, info = planner.align_real(
                        prediction, actual, timestamp, timestamp + 50_000_000)
                    seam = planner._sample_at(output, timestamp, timestamp + 50_000_000)
                    for ids in POINT_BLOCKS:
                        np.testing.assert_array_equal(output[:, ids], prediction[:, ids])
                        np.testing.assert_array_equal(target[ids], seam[ids])
                    self.assertEqual(info['finger_point_anchor'], 'decoded_prediction')
                    planner.update(output, timestamp)

    def test_real_wrist_preserves_hand_plan_and_replaces_entire_torso_pose(self):
        """Multiple replans retain real's hands while publishing measured torso poses."""
        from hdt.action_representation import POSE_BLOCKS, matrix_to_rotation_6d
        from scipy.spatial.transform import Rotation
        real = ScheduledAnchorPlanner(30, anchor_mode='real')
        hybrid = ScheduledAnchorPlanner(30, anchor_mode='real_wrist')
        torso = np.concatenate([C.OUTPUT_HEAD_EEF, C.OUTPUT_WAIST])
        other = np.setdiff1d(np.arange(128), torso)
        for step in range(4):
            actual = self.valid_state()
            for ids in (C.OUTPUT_HEAD_EEF, C.OUTPUT_WAIST):
                actual[ids[:3]] = [.1 * step, -.02 * step, .8 + .01 * step]
                actual[ids[3:9]] = matrix_to_rotation_6d(
                    Rotation.from_euler('xyz', [3 * step, 2 * step, 5 * step], degrees=True).as_matrix())
            prediction = np.repeat(actual[None], 100, axis=0)
            for ids in POSE_BLOCKS:
                prediction[:, ids[:3]] += np.arange(1, 101)[:, None] * .002 + .01
            timestamp = 1_000_000_000 + step * 333_333_333
            outputs = []
            for planner in (real, hybrid):
                planner.select(actual, timestamp, plan_epoch=2)
                output, target, info = planner.align_real(
                    prediction, actual, timestamp, timestamp + 50_000_000)
                planner.update(output, timestamp)
                outputs.append(output)
            np.testing.assert_array_equal(outputs[0][:, other], outputs[1][:, other])
            np.testing.assert_array_equal(outputs[1][:, torso],
                                           np.repeat(actual[None, torso], 100, axis=0))
            np.testing.assert_array_equal(hybrid.previous_action[:, torso], outputs[1][:, torso])
            np.testing.assert_array_equal(target[torso], actual[torso])
            self.assertEqual(info['mode'], 'real_wrist_first' if step == 0 else 'real_wrist')

    def test_real_wrist_wire_head_root_restore_measured_world_poses(self):
        """Published FK torso poses round-trip across origin and nonzero heading."""
        from scipy.spatial.transform import Rotation
        from scalebridge.online.reference import apply_alignment
        measured = state()
        origin = np.array([.37, -.18, 1.23], dtype=np.float32)
        heading = Rotation.from_euler('z', -43, degrees=True).as_quat()[[3, 0, 1, 2]]
        for name in ('head', 'root'):
            measured[name + '_quat_world_wxyz'] = Rotation.from_euler(
                'xyz', [17, -26, 61], degrees=True).as_quat()[[3, 0, 1, 2]]
        actual = robot_state_to_policy_state(measured, origin=origin,
                                            heading_quat_wxyz=heading)
        planner = ScheduledAnchorPlanner(30, anchor_mode='real_wrist')
        planner.select(actual, 1_000_000_000, plan_epoch=2)
        prediction = np.repeat(actual[None], 100, axis=0)
        prediction[:, C.OUTPUT_HEAD_EEF[:3]] += .4
        prediction[:, C.OUTPUT_WAIST[:3]] += .4
        output, _, _ = planner.align_real(prediction, actual, 1_000_000_000, 1_050_000_000)
        wire = action_chunk_to_targets(
            output, measured, 30, 0, 1_050_000_000,
            position_origin_world=origin, action_representation='relative_chunk',
            frame_heading_world_wxyz=heading)
        world = apply_alignment(wire, wire['position_origin_world'], wire['frame_heading_world_wxyz'])
        for name in ('head', 'root'):
            np.testing.assert_allclose(world[name + '_pos'],
                np.repeat(np.asarray(measured[name + '_pos_world'])[None], 100, axis=0), atol=2e-7)
            expected = np.asarray(measured[name + '_quat_world_wxyz'])
            np.testing.assert_allclose(np.abs(world[name + '_quat_wxyz'] @ expected), 1, atol=1e-6)

    def test_same_time_alignment_preserves_old_progress_without_rotating_motion(self):
        """Keep planned position progress but use measured wrist attitude.

        The seam attitude differs by 35 deg here, and the published displacement
        must stay the prediction's own world-frame step regardless: the labels
        store positions as world differences, so rotating them by the seam error
        is what sent the 2026-10-05 `real` runs off in the wrong direction.
        """
        from scipy.spatial.transform import Rotation
        from hdt.action_representation import matrix_to_rotation_6d, rotation_6d_to_matrix
        actual = self.valid_state()
        planner = ScheduledAnchorPlanner(10, anchor_mode="real")
        planner.select(actual, 1_000_000_000, plan_epoch=2)
        previous = np.repeat(actual[None], 10, axis=0)
        ids = C.OUTPUT_RIGHT_EEF
        previous[:, ids[0]] = np.arange(1, 11) * .01
        previous[:, ids[3:9]] = matrix_to_rotation_6d(
            Rotation.from_euler('z', np.arange(1, 11) * 10, degrees=True).as_matrix())
        planner.update(previous, 1_000_000_000)
        # Same actual state; the old plan must keep its .035 m progress at 1.35s.
        planner.select(actual, 1_200_000_000, plan_epoch=2)
        prediction = np.repeat(actual[None], 10, axis=0)
        prediction[:, ids[0]] = .0067 + np.arange(1, 11) * .01
        output, target, info = planner.align_real(
            prediction, actual, 1_200_000_000, 1_350_000_000)
        self.assertEqual(info['mode'], 'real')
        self.assertAlmostEqual(float(target[ids[0]]), .035, places=6)
        seam = planner._sample_at(output, 1_200_000_000, 1_350_000_000)
        np.testing.assert_allclose(seam[ids], target[ids], atol=1e-6)
        np.testing.assert_allclose(output[2, ids[:3]] - output[1, ids[:3]],
                                   [.01, 0, 0], atol=1e-7)
        np.testing.assert_allclose(output[:, ids[:3]] - output[0, ids[:3]],
                                   prediction[:, ids[:3]] - prediction[0, ids[:3]], atol=1e-7)
        np.testing.assert_allclose(rotation_6d_to_matrix(output[2, ids[3:9]]),
                                   np.eye(3), atol=1e-6)

    def test_wrist_rotation_replans_do_not_accumulate_unexecuted_motion(self):
        """Both real modes keep wrist increments near a stalled measured pose."""
        from scipy.spatial.transform import Rotation
        from hdt.action_representation import matrix_to_rotation_6d, rotation_6d_to_matrix
        wrists = (C.OUTPUT_LEFT_EEF, C.OUTPUT_RIGHT_EEF)
        for mode in ('real', 'real_wrist'):
            with self.subTest(mode=mode):
                actual = self.valid_state()
                measured = Rotation.from_euler('xyz', [12, -8, 10], degrees=True).as_matrix()
                increment = Rotation.from_euler('z', 5, degrees=True).as_matrix()
                old = Rotation.from_euler('xyz', [-20, 15, 30], degrees=True).as_matrix()
                for ids in wrists:
                    actual[ids[3:9]] = matrix_to_rotation_6d(measured)
                planner = ScheduledAnchorPlanner(10, anchor_mode=mode)
                planner.select(actual, 1_000_000_000, plan_epoch=2)
                previous = np.repeat(actual[None], 10, axis=0)
                for ids in wrists:
                    previous[:, ids[3:9]] = matrix_to_rotation_6d(old)
                planner.update(previous, 1_000_000_000)
                prediction = np.repeat(actual[None], 10, axis=0)
                for ids in wrists:
                    # Receipt at t+0.1s samples row0; row1 requests 5 deg more.
                    prediction[1:, ids[3:9]] = matrix_to_rotation_6d(measured @ increment)
                    prediction[:, ids[0]] += np.arange(1, 11) * .01
                for step in range(60):
                    timestamp = 1_200_000_000 + step * 200_000_000
                    planner.select(actual, timestamp, plan_epoch=2)
                    output, target, info = planner.align_real(
                        prediction, actual, timestamp, timestamp + 100_000_000)
                    for ids in wrists:
                        np.testing.assert_allclose(rotation_6d_to_matrix(output[0, ids[3:9]]),
                                                   measured, atol=1e-6)
                        np.testing.assert_allclose(rotation_6d_to_matrix(output[1, ids[3:9]]),
                                                   measured @ increment, atol=1e-6)
                        np.testing.assert_array_equal(target[ids[3:9]], actual[ids[3:9]])
                        np.testing.assert_allclose(output[1:, ids[:3]] - output[:-1, ids[:3]],
                                                   prediction[1:, ids[:3]] - prediction[:-1, ids[:3]],
                                                   atol=2e-6)
                    self.assertEqual(info['wrist_rotation_anchor'], 'measured_input')
                    planner.update(output, timestamp)

    def test_pause_and_new_epoch_clear_real_plan(self):
        """A paused or restarted episode cannot inherit an earlier plan."""
        actual = self.valid_state()
        planner = ScheduledAnchorPlanner(30, anchor_mode='real')
        for active, epoch in ((True, 2), (False, 2), (True, 3)):
            planner.select(actual, 1_000_000_000, plan_active=active, plan_epoch=epoch)
            prediction = np.repeat(actual[None], 10, axis=0)
            output, _, info = planner.align_real(prediction, actual, 1_000_000_000,
                                                 1_050_000_000, plan_active=active)
            self.assertEqual(info['mode'], 'real_first' if active else 'real_hold')
            planner.update(output, 1_000_000_000, plan_active=active)


if __name__ == "__main__":
    unittest.main()
