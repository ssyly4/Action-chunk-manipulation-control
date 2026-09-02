# NERO Bimanual Control

This repository owns the host-side bimanual execution path:

```text
OpenPI server request
  -> action-chunk scheduler / RTC
  -> OSQP waypoint shaping
  -> CasADi fixed-horizon phase retiming
  -> CPV feedback governor
  -> NERO CAN arms
```

It deliberately does not contain PICO teleoperation code, collected data, or
model checkpoints.

## Neighbor repositories

| Responsibility | Location |
| --- | --- |
| PICO teleop, Home, CAN preparation, recording | `/home/dev/nero_neo_teleop` |
| Canonical raw and curated data | `/home/dev/nero_data` |
| Training datasets, checkpoints, server logs | training server |

## Main entry points

- Native guarded policy runtime: `scripts/run_policy_native.sh`
- Current OSQP + CasADi A/B runtime: `scripts/run_policy_osqp_casadi.sh`
- Dependency and path template: `config/paths.env.example`

Before commanding hardware, source a local `config/paths.env` and run the
launcher's preflight mode. The launchers only use external paths for teleop
hardware preparation, the official SDK, and the remote policy server.
