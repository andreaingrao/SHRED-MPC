# Research status and next steps

## Completed and inspectable

The retained July experiment provides one model, one partial rollout and 250 candidate archives. Offline figures and numerical summaries are reproducible from a small real sample. The existing software includes control-conditioned pressure prediction, finite-candidate MPC and a candidate-CFD validation runner.

## Implemented but not scientifically evaluated in this archive

The maintained temporal launcher builds four matched scenario families, saves models and preprocessing, records causal indices, and reuses reference CFD across training seeds. Focused tests validate software behavior. No completed temporal study with independent model/state variability was found.

## Scientific work before stronger claims

1. Establish the intended physical benchmark. Current steady-state dictionaries cannot substantiate transient dynamics. Any revised discretization or Reynolds number needs new data and experiment IDs, plus mesh/time and boundary-condition verification.
2. Synchronize pressure histories, actual controls and CFD checkpoints; validate the initial controller cycle and pressure-reference observation requirement.
3. Compare the historical ring observable with wall-pressure integration and full solver forces. Keep corrected force scales separate from historical targets/results.
4. Freeze a model and evaluate held-out branches from multiple incoming states, including zero-change references. Report paired response errors and candidate regret, not field error alone.
5. Compare meaningful baselines, independently trained models and control trajectories; measure actual computation cost before claiming speedup.

Force-aware training, sensor/noise robustness, alternative horizons, adaptive data collection and online learning are proposed extensions, not completed results. The old full [paper roadmap](context/PAPER_RESEARCH_ROADMAP.md) is retained as dated context; its assumptions about incomplete data and sensor selection are superseded by this audit.
