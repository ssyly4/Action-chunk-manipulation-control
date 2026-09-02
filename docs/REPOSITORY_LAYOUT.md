# Repository Layout

`nero_bimanual_control` is the single owner of inference-to-actuation code.

```text
nero_bimanual_control/
  nero_vla/                 CPV backend, CAN checks, follower, policy client
  scripts/bimanual_policy/  native policy server and real-arm launchers
  trajectory/
    osqp_waypoint_smoother/ waypoint-space smoothing A/B layer
    casadi_fixed_horizon_retimer/ fixed-duration phase optimizer
    toppra_fixed_horizon_retimer/ path retiming and continuous handoff
  server_staging/           remote OpenPI staging helpers
  config/                   non-secret local path templates
  artifacts/                ignored runtime logs
```

The recording implementation intentionally stays in `nero_neo_teleop`; its
output belongs under `nero_data`. Model checkpoints stay on the training
server. This separation prevents experiment artifacts from becoming runtime
source dependencies.
