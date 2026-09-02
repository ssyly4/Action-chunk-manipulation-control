# Archive Index

Archived material is preserved for reproducibility and comparison. It is not
part of the active physical execution path.

| Location | Status | Reason |
|---|---|---|
| `archive/trajectory_retiming/ruckig_fixed_horizon_poc/` | Rejected PoC | Point-to-point Ruckig handoff did not preserve the fixed RTC horizon and produced repeated queue stalls. |
| `archive/legacy_launchers/` | Superseded launchers | Older feedback/native/RTC towel runners with task-specific guards and duplicated policy presets. |
| `server_staging/legacy_pipelines/` | Historical training variants | 50-episode, stage23, and multistage99 pipelines using earlier action definitions or observation contracts. |
| `casadi_fixed_horizon_retimer/legacy_runtime/` | Superseded A/B runner | Direct CasADi-only physical runner; its optimizer remains active as an OSQP dependency. |
| `toppra_fixed_horizon_retimer/legacy_runtime/` | Superseded A/B runner | Direct TOPPRA-only physical runner; its shared queue and bridge remain active dependencies. |
| `docs/archive/` | Historical documentation | Prior ROS container notes, the August 2026 system map, and detailed retimer design notes. |

Archived code is intentionally retained rather than deleted. Do not use an
archived launcher for physical execution without first moving it back through a
reviewed experiment branch.
