# Shared Receding-Horizon Runtime Base

This directory now contains the shared execution base used by the active OSQP
+ CasADi physical runtime. Its retained code is responsible for fixed-rate plan
sampling, q/v/a continuous handoffs, command-state bridging, reserve handling,
and runtime diagnostics.

## Active API

- `fixed_path_retimer/`: reference-path and retiming data structures.
- `ab_runtime/receding_toppra_queue.py`: `RecedingToppraRtcQueue`, handoff
  safety contract, and reserve/replan behavior.
- `ab_runtime/follower_state_bridge.py`: command and desired-velocity state
  passed between queue and streaming follower.
- `tests/`: handoff and fixed-path regression coverage.

Run its tests from this directory:

```bash
./run_tests.sh
```

Standalone TOPPRA physical runners are historical material in `legacy_runtime/`.
The full prior design notes are retained in
`docs/archive/TOPPRA_fixed_horizon_retimer_design_notes.md`.
