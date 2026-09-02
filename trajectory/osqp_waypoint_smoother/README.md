# OSQP Fixed-Horizon Waypoint Smoother

This is the current experimental physical runtime entry point. It does not
modify the maintained Pi0.5, RTC, follower, or CPV source: its A/B adapter
replaces the queue and follower only inside the launched process.

For the supported physical command and parameter ownership, read
[`../docs/CURRENT_CONTROL_STACK.md`](../docs/CURRENT_CONTROL_STACK.md).

## Contract

For one `H x 16` policy chunk, the smoother:

- optimizes only the 14 arm joint columns;
- preserves both gripper columns exactly;
- preserves `H` and the fixed 30 Hz timeline;
- keeps every arm waypoint inside a configurable trust region around the
  policy output;
- applies linear velocity, acceleration, jerk, and optional joint-position
  constraints;
- supports an exact start position and start velocity boundary;
- does not impose zero terminal velocity;
- returns the original input chunk on infeasibility or failed verification.

The convex objective is:

```text
tracking + velocity tracking + acceleration regularization
         + jerk regularization + terminal velocity tracking
```

OSQP changes the joint waypoints. A later CasADi stage may then redistribute
phase along this bounded, smoother path without changing the global horizon.

## Setup

Dependencies are isolated in this directory:

```bash
cd /home/dev/ros2_project/osqp_waypoint_smoother
/home/dev/enter/envs/lerobot/bin/python -m pip install \
  --target vendor -r requirements.txt
```

## Test

```bash
cd /home/dev/ros2_project/osqp_waypoint_smoother
chmod +x run_tests.sh scripts/smooth_chunk.py
./run_tests.sh
```

## Offline chunk probe

```bash
./scripts/smooth_chunk.py /path/to/actions.npy \
  --output outputs/actions_smoothed.npy \
  --trust-deg 0.3 \
  --max-velocity-deg-s 28 \
  --max-acceleration-deg-s2 280 \
  --max-jerk-deg-s3 8000
```

The output report includes raw and smoothed velocity, acceleration, and jerk
ratios plus maximum waypoint distortion. This stage should be connected to
CasADi only after these offline metrics show a useful feasibility improvement
without excessive waypoint displacement.

## Isolated runtime A/B

The runtime path remains separate from the production follower:

```text
raw RTC chunk -> OSQP waypoint smoothing -> CasADi retiming -> existing handoff
```

Action Gain is bypassed in the normal path. Every non-initial request also
predicts the earliest real takeover tick from the active plan's commit and
reserve boundaries. A bounded Action Gain recovery candidate is generated from
the rate-limited follower's predicted q/v at that wall-clock tick and from the
corresponding future action index in the new chunk. The short follower rollout
uses the production streaming position gain and the same velocity,
acceleration, and jerk limits. At the actual commit boundary, the raw
candidate remains preferred. The recovery candidate is selected when the raw
candidate either exceeds the hard q/v handoff limits or cannot form a bounded
q/v/a correction, provided the recovery candidate can. This avoids judging
recovery against the earlier request-time state and then discovering a new
mismatch after the robot has moved for another commit window.

If bounded OSQP smoothing rejects the raw chunk itself, the adapter uses the
recovery candidate as the primary result. A second rejection fails closed
instead of forwarding an unsafe trajectory. Runtime logs distinguish
`gain=bypassed`, `gain=fallback:<value>`, and
`handoff_candidate=recovery`. Recovery selection logs also include the
predicted takeover source and wall-clock tick.

The OSQP A/B launcher uses an 8-tick minimum commit and preserves an 8-tick
replan reserve. If neither the raw nor recovery candidate supports a bounded
handoff, that candidate is discarded at the replan boundary and a fresh RTC
request is allowed. This path deliberately disables the legacy
`reserve_exhaustion_follower_handoff`: it fails closed and replans instead of
forcing an unbounded follower takeover after the old trajectory is exhausted.

The runtime adapter replaces the RTC queue only inside its own process. The
production stream source remains unchanged:

```bash
cd /home/dev/ros2_project
NERO_POLICY_DURATION=30 \
  ./osqp_waypoint_smoother/ab_runtime/run_30k_osqp_casadi.sh --preflight-only
```

After preflight, omit `--preflight-only` to run the guarded physical A/B. The
default runtime leaves
cross-chunk continuity to the existing q/v handoff. Set
`NERO_OSQP_ENFORCE_BOUNDARY=1` only for a separate strict boundary experiment.

## Recorded policy A/B

Compare the existing raw-to-CasADi path with OSQP-to-CasADi without robot
commands:

```bash
./scripts/replay_chunks_to_casadi.py \
  /home/dev/nero_ws/logs/bimanual_policy_stream/RUN/chunks.jsonl \
  --trust-deg 0.3 \
  --output outputs/replay_report.json
```
