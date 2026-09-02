# CasADi Fixed-Horizon Phase Optimizer

This directory provides the phase optimizer used by the active OSQP + CasADi
policy runtime. It never creates a new joint-space path. Given a fixed policy
path, it redistributes local phase while preserving the 30 Hz horizon boundary.

## Active API

- `fixed_phase_optimizer/optimizer.py`: `CasadiPhaseOptimizer` and
  `PhaseOptimizerConfig`.
- `ab_runtime/casadi_rtc_queue.py`: queue adapter inherited by the active OSQP
  runtime.
- `tests/`: fixed-horizon, feasibility, and warm-start coverage.

Run its tests from this directory:

```bash
./run_tests.sh
```

The direct CasADi-only physical runner is historical material under
`legacy_runtime/`. The full prior design notes are retained in
`docs/archive/CasADi_fixed_horizon_retimer_design_notes.md`.
