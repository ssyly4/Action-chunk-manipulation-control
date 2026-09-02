# Current Control Stack

## Ownership

The running physical-policy path has three local layers:

```text
policy WebSocket output (24 x 16 action chunk)
  -> OSQP bounded waypoint smoother
  -> CasADi fixed-horizon phase retimer
  -> shared receding RTC queue and q/v/a handoff
  -> NERO streaming joint follower
  -> CPV backend -> CAN -> two NERO arms
```

`osqp_waypoint_smoother/ab_runtime/run_30k_osqp_casadi.sh` is the only active
physical launcher in this workspace. Its small set of environment variables is
the supported tuning surface:

| Variable | Default | Meaning |
|---|---:|---|
| `NERO_POLICY_DURATION` | `30` s | Trial duration |
| `NERO_FOLLOWER_MAX_VELOCITY_DEG_S` | `28` | Joint follower speed limit |
| `NERO_FOLLOWER_MAX_ACCELERATION_DEG_S2` | `280` | Joint follower acceleration limit |
| `NERO_FOLLOWER_GOVERNOR_ERROR_DEG` | `1.50` | Soft command-feedback lead limit |
| `NERO_FOLLOWER_HARD_ERROR_DEG` | `2.25` | Hard command-feedback stop limit |
| `NERO_TOPPRA_MIN_COMMIT_TICKS` | `8` | Minimum active-plan duration before normal replacement |
| `NERO_TOPPRA_REPLAN_RESERVE_TICKS` | `8` | Remaining trajectory reserve required for fail-closed replan |
| `NERO_TOPPRA_ALLOW_RESERVE_FOLLOWER_HANDOFF` | `0` | Must remain disabled for the current A/B |

The control clock always remains at 30 Hz. Retiming is allowed to redistribute
phase *inside* a policy horizon, but may not change the horizon boundary.

## Source Responsibilities

| Path | Responsibility |
|---|---|
| `osqp_waypoint_smoother/waypoint_smoother/smoother.py` | Convex waypoint smoothing with bounded deviation from policy output |
| `osqp_waypoint_smoother/ab_runtime/osqp_casadi_rtc_queue.py` | Active OSQP/CasADi queue, recovery candidate selection |
| `casadi_fixed_horizon_retimer/fixed_phase_optimizer/optimizer.py` | Fixed-duration phase optimization under velocity, acceleration, and jerk constraints |
| `toppra_fixed_horizon_retimer/ab_runtime/receding_toppra_queue.py` | Shared q/v/a continuous handoff, reserve, and safety decisions |
| `toppra_fixed_horizon_retimer/ab_runtime/follower_state_bridge.py` | Follower command-state bridge for runtime handoff boundaries |
| `/home/dev/nero_ws/scripts/bimanual_policy/bimanual_guarded_policy_stream.py` | Maintained NERO policy client, cameras, safety checks, CPV command output |
| `/home/dev/nero_ws/nero_vla/trajectory_executor.py` | Streaming follower and feedback governor |

## Operating Rules

- Do not edit the maintained `/home/dev/nero_ws` stream merely to test a
  retimer. The A/B adapters replace queue and follower behavior in-process.
- `reserve_exhaustion_follower_handoff` is disabled in the current launcher.
  An unsafe candidate is discarded early and a fresh RTC request is made.
- `rtc_queue_hold` is diagnostic evidence of inadequate reserve or inference
  latency. It should be investigated, not hidden by raising hard limits.
- Runtime diagnostics are written to
  `/home/dev/ros2_project/osqp_waypoint_smoother/outputs/` and policy ticks to
  `/home/dev/nero_ws/logs/bimanual_policy_stream/`.

## Verification

```bash
cd /home/dev/ros2_project
./toppra_fixed_horizon_retimer/run_tests.sh
./casadi_fixed_horizon_retimer/run_tests.sh
./osqp_waypoint_smoother/run_tests.sh
```

Use a hardware preflight before an executed trial:

```bash
./osqp_waypoint_smoother/ab_runtime/run_30k_osqp_casadi.sh --preflight-only
```
