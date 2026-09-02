# External dependencies

This directory documents dependencies that are not NERO control code.

- `pyAgxArm` remains at the SDK checkout selected by `NERO_ARM_SDK_ROOT`.
- The OSQP, CasADi, and TOPPRA Python dependency bundles live next to their
  trajectory modules because their CPython 3.12 binary wheels must match the
  real-time policy environment.

None of these directories contain collected data or model checkpoints.
