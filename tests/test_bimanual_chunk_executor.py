import numpy as np

from nero_vla.bimanual_chunk_executor import BimanualFeedbackProgressActionChunk
from nero_vla.bimanual_chunk_executor import BimanualFeedbackRtcActionQueue
from nero_vla.bimanual_chunk_executor import BimanualFixedHorizonActionChunk
from nero_vla.bimanual_chunk_executor import BimanualRtcActionQueue
from nero_vla.bimanual_chunk_executor import align_bimanual_chunk_to_state
from nero_vla.bimanual_chunk_executor import arm_matrix


def actions() -> np.ndarray:
    values = np.zeros((16, 16), dtype=np.float64)
    phase = np.arange(16, dtype=np.float64)
    values[:, 1] = 0.01 * phase
    values[:, 9] = 0.02 * phase
    values[:, 7] = 1.0 - 0.02 * phase
    values[:, 15] = 1.0 - 0.01 * phase
    return values


def test_arm_matrix_uses_left_then_right_joints() -> None:
    matrix = arm_matrix(actions())
    assert matrix.shape == (16, 14)
    np.testing.assert_allclose(matrix[:, 1], 0.01 * np.arange(16))
    np.testing.assert_allclose(matrix[:, 8], 0.02 * np.arange(16))


def test_alignment_uses_both_arms_to_choose_one_phase() -> None:
    values = actions()
    feedback = arm_matrix(values)[4]
    alignment = align_bimanual_chunk_to_state(
        values,
        feedback=feedback,
        command=feedback,
        latency_steps=5.0,
        max_alignment_error_rad=0.05,
    )
    assert 3.9 <= alignment.offset_steps <= 4.1
    # The small latency regularizer may pull the optimum slightly forward even
    # when feedback lies exactly on a sampled action.
    np.testing.assert_allclose(alignment.action, feedback, atol=2e-4)


def test_progress_projects_one_shared_phase_and_grippers_follow_it() -> None:
    values = actions()
    buffer = BimanualFeedbackProgressActionChunk(
        arm_lead_steps=1.0,
        gripper_lead_steps=0.0,
        blend_duration_sec=0.0,
    )
    buffer.push(values, initial_phase_steps=0.0, observed_at=1.0, received_at=1.0)
    feedback = arm_matrix(values)[3]
    buffer.sample(1.01, feedback=feedback)
    buffer.sample(1.02, feedback=feedback)
    sample = buffer.sample(1.03, feedback=feedback)
    assert 2.9 <= sample.phase_steps <= 3.1
    assert 3.9 <= sample.target_phase_steps <= 4.1
    assert np.isclose(sample.left_target[1], 0.04)
    assert np.isclose(sample.right_target[1], 0.08)
    assert np.isclose(sample.left_gripper_target, 0.94)
    assert np.isclose(sample.right_gripper_target, 0.97)


def test_progress_handoff_preserves_new_chunk_motion_while_offset_decays() -> None:
    first = actions()
    second = actions()
    second[:, 1] += 0.02
    buffer = BimanualFeedbackProgressActionChunk(
        action_hz=30.0,
        arm_lead_steps=1.0,
        blend_duration_sec=0.1,
    )
    buffer.push(first, initial_phase_steps=0.0, observed_at=1.0, received_at=1.0)
    before = buffer.sample(1.2, feedback=arm_matrix(first)[0])

    buffer.push(second, initial_phase_steps=0.0, observed_at=1.2, received_at=1.2)
    handoff = buffer.sample(1.2, feedback=arm_matrix(second)[0])
    moving = buffer.sample(1.225, feedback=arm_matrix(second)[1])

    assert handoff.status == "blending"
    np.testing.assert_allclose(handoff.left_target, before.left_target)
    # Raw joint1 target advances from 0.03 to 0.04 while the -0.02 rad
    # handoff offset decays to 75%, yielding 0.025 without a velocity reset.
    assert np.isclose(moving.left_target[1], 0.025)


def test_stale_chunk_holds_both_arms() -> None:
    values = actions()
    buffer = BimanualFeedbackProgressActionChunk(stale_after_sec=0.0)
    buffer.push(values, initial_phase_steps=14.0, observed_at=1.0, received_at=1.0)
    sample = buffer.sample(2.0, feedback=arm_matrix(values)[14])
    assert sample.status == "stale_chunk_hold"
    assert sample.left_target is None
    assert sample.right_target is None


def test_close_event_advances_only_the_gripper_that_predicts_closure() -> None:
    values = np.zeros((16, 16), dtype=np.float64)
    values[:, 7] = 1.0
    values[:, 15] = 1.0
    values[9:, 15] = np.linspace(0.95, 0.65, 7)
    buffer = BimanualFeedbackProgressActionChunk(
        gripper_event_lookahead_steps=10.0,
        gripper_event_activation_delta=0.03,
        blend_duration_sec=0.0,
    )
    buffer.push(values, initial_phase_steps=3.0, observed_at=1.0, received_at=1.0)

    sample = buffer.sample(1.01, feedback=arm_matrix(values)[3])

    assert np.isclose(sample.left_gripper_target, 1.0)
    assert np.isclose(sample.right_gripper_target, 0.75)


def test_close_event_outside_window_does_not_advance_gripper() -> None:
    values = np.zeros((16, 16), dtype=np.float64)
    values[:, [7, 15]] = 1.0
    values[14:, 15] = 0.5
    buffer = BimanualFeedbackProgressActionChunk(
        gripper_event_lookahead_steps=4.0,
        gripper_event_activation_delta=0.03,
        blend_duration_sec=0.0,
    )
    buffer.push(values, initial_phase_steps=3.0, observed_at=1.0, received_at=1.0)

    sample = buffer.sample(1.01, feedback=arm_matrix(values)[3])

    assert np.isclose(sample.right_gripper_target, 1.0)


def test_fixed_horizon_advances_one_action_per_control_tick() -> None:
    values = actions()
    buffer = BimanualFixedHorizonActionChunk(
        action_hz=30.0, execution_horizon_steps=8, blend_duration_sec=0.0
    )
    buffer.push(values, observed_at=1.0, received_at=1.0)

    # Deliberately irregular timestamps must not duplicate or skip an action.
    for index, now in enumerate((1.0, 1.061, 1.067, 1.124)):
        sample = buffer.sample(now, feedback=np.zeros(14))
        assert sample.status == "fixed_horizon_tracking"
        assert sample.phase_steps == float(index)
        assert np.isclose(sample.left_target[1], values[index, 1])
        assert np.isclose(sample.right_target[1], values[index, 9])
        assert np.isclose(sample.left_gripper_target, values[index, 7])
        assert np.isclose(sample.right_gripper_target, values[index, 15])


def test_fixed_horizon_holds_last_executed_action_after_prefix() -> None:
    values = actions()
    buffer = BimanualFixedHorizonActionChunk(
        action_hz=30.0, execution_horizon_steps=8, blend_duration_sec=0.0
    )
    buffer.push(values, observed_at=1.0, received_at=1.0)
    for index in range(8):
        buffer.sample(1.0 + index / 30.0, feedback=np.zeros(14))

    sample = buffer.sample(1.0 + 9.0 / 30.0, feedback=np.zeros(14))

    assert sample.status == "fixed_horizon_complete_hold"
    assert sample.phase_steps == 7.0
    assert np.isclose(sample.right_gripper_target, values[7, 15])


def test_fixed_horizon_starts_from_shared_aligned_index() -> None:
    values = actions()
    buffer = BimanualFixedHorizonActionChunk(
        action_hz=30.0, execution_horizon_steps=12, blend_duration_sec=0.0
    )
    buffer.push(
        values,
        initial_phase_steps=4.2,
        observed_at=1.0,
        received_at=1.0,
    )

    first = buffer.sample(1.0, feedback=np.zeros(14))

    assert first.phase_steps == 5.0
    assert np.isclose(first.left_target[1], values[5, 1])
    assert np.isclose(first.right_target[1], values[5, 9])


def test_fixed_horizon_blends_only_arm_handoff_not_grippers() -> None:
    first = np.zeros((16, 16), dtype=np.float64)
    second = np.ones((16, 16), dtype=np.float64)
    buffer = BimanualFixedHorizonActionChunk(
        action_hz=30.0, execution_horizon_steps=8, blend_duration_sec=0.1
    )
    buffer.push(first, observed_at=1.0, received_at=1.0)
    first_sample = buffer.sample(1.0, feedback=np.zeros(14))
    assert np.isclose(first_sample.left_target[0], 0.0)
    buffer.push(second, observed_at=2.0, received_at=2.0)

    handoff = buffer.sample(2.0, feedback=np.zeros(14))

    assert handoff.status == "fixed_horizon_blending"
    assert np.isclose(handoff.left_target[0], 0.0)
    assert np.isclose(handoff.right_target[0], 0.0)
    assert np.isclose(handoff.left_gripper_target, 1.0)
    assert np.isclose(handoff.right_gripper_target, 1.0)


def test_fixed_horizon_handoff_preserves_new_chunk_motion() -> None:
    first = actions()
    second = actions()
    second[:, 1] += 0.02
    buffer = BimanualFixedHorizonActionChunk(
        action_hz=30.0, execution_horizon_steps=8, blend_duration_sec=0.1
    )
    buffer.push(first, observed_at=1.0, received_at=1.0)
    before = buffer.sample(1.0, feedback=np.zeros(14))
    buffer.push(second, observed_at=2.0, received_at=2.0)

    handoff = buffer.sample(2.0, feedback=np.zeros(14))
    moving = buffer.sample(2.025, feedback=np.zeros(14))

    np.testing.assert_allclose(handoff.left_target, before.left_target)
    # Raw joint1 advances by 0.01 rad while the -0.02 rad handoff offset
    # decays to 75%, so motion continues to 0.015 rad instead of restarting.
    assert np.isclose(moving.left_target[1], 0.015)


def test_rtc_queue_replaces_chunk_after_consumed_delay() -> None:
    first = actions()
    second = actions() + 1.0
    queue = BimanualRtcActionQueue(action_hz=30.0)
    queue.load(first)
    for tick in range(4):
        queue.sample(tick / 30.0, feedback=np.zeros(14))
    request = queue.make_request(now=4 / 30.0, execution_horizon=12, predicted_delay_steps=7)
    assert request.valid_previous_steps == 12
    for tick in range(4, 11):
        queue.sample(tick / 30.0, feedback=np.zeros(14))
    assert queue.consumed_since(request) == 7

    queue.load(second, skip_steps=7)
    sample = queue.sample(11 / 30.0, feedback=np.zeros(14))

    np.testing.assert_allclose(sample.left_target, second[7, :7])
    np.testing.assert_allclose(sample.right_target, second[7, 8:15])
    assert sample.left_gripper_target == second[7, 7]
    assert sample.right_gripper_target == second[7, 15]


def test_rtc_request_padding_repeats_last_valid_target() -> None:
    queue = BimanualRtcActionQueue()
    queue.load(actions(), skip_steps=10)

    request = queue.make_request(now=1.0, execution_horizon=12, predicted_delay_steps=7)

    assert request.previous_actions.shape == (12, 16)
    assert request.valid_previous_steps == 6
    np.testing.assert_allclose(
        request.previous_actions[5:], np.repeat(actions()[15:16], 7, axis=0)
    )


def test_rtc_empty_queue_holds_last_action() -> None:
    queue = BimanualRtcActionQueue()
    queue.load(actions(), skip_steps=15)
    queue.sample(0.0, feedback=np.zeros(14))

    held = queue.sample(1 / 30.0, feedback=np.zeros(14))

    assert held.status == "rtc_queue_hold"
    np.testing.assert_allclose(held.left_target, actions()[15, :7])


def test_rtc_empty_queue_requests_from_safe_hold_trajectory() -> None:
    queue = BimanualRtcActionQueue()
    queue.load(actions(), skip_steps=15)
    final = queue.sample(0.0, feedback=np.zeros(14))
    assert queue.remaining_steps == 0

    request = queue.make_request(
        now=1 / 30.0,
        execution_horizon=4,
        predicted_delay_steps=7,
    )

    assert request.valid_previous_steps == 4
    expected = np.concatenate(
        (final.left_target, [final.left_gripper_target], final.right_target, [final.right_gripper_target])
    )
    np.testing.assert_allclose(
        request.previous_actions, np.repeat(expected[None, :], 4, axis=0)
    )


def test_rtc_reanchor_replaces_stale_tail_with_live_hold() -> None:
    queue = BimanualRtcActionQueue(action_hz=30.0, handoff_decay_steps=3)
    queue.load(actions())
    queue.sample(0.0, feedback=np.zeros(14))
    anchor = np.full(16, 0.25, dtype=np.float64)

    queue.reanchor_hold(anchor, now=0.1)
    request = queue.make_request(
        now=0.1,
        execution_horizon=4,
        predicted_delay_steps=7,
    )
    sample = queue.sample(0.1, feedback=np.zeros(14))

    np.testing.assert_allclose(request.previous_actions, np.repeat(anchor[None, :], 4, axis=0))
    np.testing.assert_allclose(sample.left_target, anchor[:7])
    np.testing.assert_allclose(sample.right_target, anchor[8:15])
    assert sample.left_gripper_target == anchor[7]
    assert sample.right_gripper_target == anchor[15]


def test_rtc_handoff_errors_keep_the_two_arms_separate() -> None:
    queue = BimanualRtcActionQueue()
    queue.load(actions())
    queue.sample(0.0, feedback=np.zeros(14))
    replacement = actions().copy()
    replacement[0, 0] += 0.01
    replacement[0, 8] += 0.04

    errors = queue.handoff_errors_rad(replacement)

    np.testing.assert_allclose(errors, [0.01, 0.04])
    assert np.isclose(queue.handoff_error_rad(replacement), 0.04)


def test_rtc_policy_rate_can_be_lower_than_control_rate() -> None:
    queue = BimanualRtcActionQueue(action_hz=25.0)
    queue.load(actions())

    samples = [queue.sample(tick / 30.0, feedback=np.zeros(14)) for tick in range(7)]

    assert queue.emitted_steps == 6
    assert sum(sample.status == "rtc_rate_hold" for sample in samples) == 1


def test_rtc_handoff_offset_decays_without_stopping_new_chunk_motion() -> None:
    first = actions()
    second = actions()
    second[:, 1] += 0.03
    queue = BimanualRtcActionQueue(
        action_hz=30.0,
        handoff_decay_steps=3,
        max_handoff_error_rad=0.05,
    )
    queue.load(first)
    before = queue.sample(0.0, feedback=np.zeros(14))
    queue.load(second)

    handoff = queue.sample(1 / 30.0, feedback=np.zeros(14))
    moving = queue.sample(2 / 30.0, feedback=np.zeros(14))

    np.testing.assert_allclose(handoff.left_target, before.left_target)
    assert np.isclose(moving.left_target[1], 0.02)
    assert np.isclose(moving.left_gripper_target, second[1, 7])


def test_feedback_rtc_does_not_advance_from_wall_time() -> None:
    values = actions()
    queue = BimanualFeedbackRtcActionQueue(
        arm_lead_steps=1.0,
        blend_duration_sec=0.0,
        stale_after_sec=0.0,
    )
    queue.push(values, initial_phase_steps=0.0, observed_at=0.0, received_at=0.0)
    feedback = arm_matrix(values)[0]

    samples = [queue.sample(float(tick), feedback=feedback) for tick in (0, 1, 100)]

    assert all(sample.phase_steps == 0.0 for sample in samples)
    assert queue.remaining_steps == 15
    assert samples[-1].status == "rtc_feedback_tracking"


def test_feedback_rtc_counts_motion_not_inference_wall_time() -> None:
    values = actions()
    queue = BimanualFeedbackRtcActionQueue(
        arm_lead_steps=1.0,
        max_progress_steps_per_tick=1.0,
        blend_duration_sec=0.0,
        stale_after_sec=0.0,
    )
    queue.push(values, initial_phase_steps=0.0, observed_at=0.0, received_at=0.0)
    request = queue.make_request(
        now=0.0, execution_horizon=12, predicted_delay_steps=7
    )

    queue.sample(0.0, feedback=arm_matrix(values)[0])
    queue.sample(0.6, feedback=arm_matrix(values)[1])

    assert queue.consumed_since(request) == 1
    assert queue.remaining_steps == 14


def test_feedback_rtc_advances_only_with_measured_progress() -> None:
    values = actions()
    queue = BimanualFeedbackRtcActionQueue(
        arm_lead_steps=1.0,
        max_progress_steps_per_tick=1.0,
        blend_duration_sec=0.0,
        stale_after_sec=10.0,
    )
    queue.push(values, initial_phase_steps=0.0, observed_at=0.0, received_at=0.0)
    request = queue.make_request(
        now=0.0, execution_horizon=12, predicted_delay_steps=1
    )

    queue.sample(0.01, feedback=arm_matrix(values)[0])
    queue.sample(0.02, feedback=arm_matrix(values)[1])
    queue.sample(0.03, feedback=arm_matrix(values)[2])

    assert queue.consumed_since(request) == 2
    assert queue.remaining_steps == 13
    assert queue.progress_rate_hz > 0.0


def test_feedback_rtc_request_starts_after_live_target_phase() -> None:
    values = actions()
    queue = BimanualFeedbackRtcActionQueue(
        arm_lead_steps=1.0,
        blend_duration_sec=0.0,
        stale_after_sec=10.0,
    )
    queue.push(values, initial_phase_steps=3.25, observed_at=0.0, received_at=0.0)

    request = queue.make_request(
        now=0.0, execution_horizon=12, predicted_delay_steps=1
    )

    np.testing.assert_allclose(request.previous_actions[0], values[5])
    assert request.valid_previous_steps == 11
