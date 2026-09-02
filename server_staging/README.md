# Training Staging Utilities

This directory mirrors the small set of scripts needed to inspect, convert, and
validate NERO training data before synchronizing equivalent changes to the
training server.

Active utilities:

- `convert_nero_bimanual_v3_to_v21.py`: LeRobot V3 to V2.1 conversion.
- `compute_nero_bimanual_norm_stats_fast.py`: normalization statistics.
- `validate_nero_towel_fullflow70_source.py` and
  `validate_nero_towel_fullflow70_v21.py`: dataset validation.
- `nero_action_alignment.py`: action/observation alignment helpers.
- `openpi/`: NERO-specific OpenPI configuration and policy definitions.

`legacy_pipelines/` contains dated experiment launchers. They encode earlier
datasets and action contracts and must not be used as a starting point for a
new run.
