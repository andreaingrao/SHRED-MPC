# SHRED-MPC paper research roadmap

This is a proposed research protocol, not a record of completed experiments. It assumes the main contribution is a sparse-pressure, control-conditioned SHRED surrogate used in receding-horizon control. Existing experiment archives must retain their original meaning; corrected implementations need new experiment identifiers.

Current agreed scope: the contributions are SHRED's adaptation as a control-conditioned forward predictor and its use as an MPC surrogate. General SHRED reconstruction benchmarks are background. Prioritize the focused forward-prediction protocol below; broader architecture, sensor, and compression studies are supplementary unless required to explain a specific failure.

## Current first step: controlled forward-prediction evaluation

Use a fixed trained model initially. Separate the trained measurement offset D (`meas_offset`) from the operational forecast position h within the existing prediction block (`meas_delta_t`). Changing D requires new training; evaluating each h in a saved forecast block does not.

For full-history samples, the installed PySHRED window convention makes the present formulation:

`Cp_hat(t) = F_theta({[s(j-D), omega(j)] : j=t-L,...,t-1})`,

where L is `shred_lags` and s contains the pressure observations. Windows exclude the current row. At an operational forecast position h after current sample k, the latest pressure sample used is k+h-D-1. Consequently a within-block lead-time curve evaluates deployed performance, but does not compare separately trained offsets or predictions using an identical latest observation window. Audit physical sample labels before presenting this as the paper's definitive indexing.

The current statistical script saves pressure/lift predictions and CFD references across the prediction block in `shred_prediction.npz`. Check archive completeness and validity, then derive per-position errors without new CFD or training. Historical results retain their original configuration and must not be presented as results of a corrected implementation.

Primary experiments:

1. Per-position Cp and lift errors across the existing prediction block. Report ramp and hold portions separately, using the true control timestamps to locate the transition.
2. Transient errors aligned by elapsed time from control onset and ramp completion. Include bias, peak absolute lift error, and near-cylinder Cp error alongside whole-field error. Pair each pressure forecast with CFD at the same lead; do not replace the CFD transient with a settled reference.
3. Controlled parameter-response tests: branch identical CFD checkpoints to negative, zero, and positive delta-omega profiles. Compare absolute predictions and differences relative to the zero-change branch. Lift-response difference MAE and pressure-response difference norms test whether the model captures the effect of the control instead of only the shared background trajectory. Avoid relative metrics with near-zero response denominators.
4. Repeat those branches for a small held-out set of incoming rotations and shedding phases. Incoming rotation is part of a checkpoint's physical history; obtain consistent checkpoints rather than overwriting that control-history label. Sample operating states deliberately, as checkpoints encountered by one MPC policy may provide biased coverage.

Minimal additions beyond the proposed measurements: paired control-response error and a small repeat across independent starting phases/states. These are more directly relevant than a broad SHRED benchmark. Optional later additions are a control-blind predictor ablation, varied ramp duration, and two or three trained offsets if the fixed-model diagnostics identify a need.

Start with approximately five bounded candidate changes including zero, several independent checkpoints, and one ramp duration. Final counts depend on pilot variability and CFD cost. Keep the trained offset fixed for this stage. Separate delta-omega from ramp rate when interpreting results: for fixed duration they change together.

First implementation task: an offline evaluator of the existing prediction archives, producing per-position field/lift errors, transient-aligned summaries, and paired response errors where a zero-change branch exists. Extend the existing CFD branching runner only for missing controlled test cases. Log checkpoint ID, model/configuration ID, sensor and target timestamps, actual omega trajectory, initial-state descriptors, and any archive failures. Summarize uncertainty by independent checkpoint/model, not by treating every time sample or candidate as independent.

The first dedicated temporal launcher is now `shred_mpc_launcher_temporal.py`, documented in `TEMPORAL_EVALUATION.md`. It trains once by default, builds matched static/history-change cases, generates isolated CFD references, and records per-lead and window-level pressure/lift errors in CSV, XLSX, and full-field NPZ archives. Additional training seeds reuse the references. It checks that the new training dataset has completed export and supports dry-run and resume. General paired response-error postprocessing remains a later addition; this launcher preserves the matched prefixes and complete predictions needed for it.

## 1. Establish the question and contribution

Working question: Can a control-conditioned SHRED model use sparse pressure histories to choose effective cylinder-rotation actions, and what determines whether accurate pressure predictions lead to accurate MPC decisions?

Organize the study around three hypotheses:

- Temporal pressure history plus explicit control information predicts controlled flow better than models without one of those inputs.
- Field reconstruction error alone is an incomplete predictor of force error and control regret.
- A force-aware training objective or appropriately targeted data collection improves decisions at a comparable training and sensing budget.

These are hypotheses to test, not claims to put in the abstract before results exist. A well-supported negative result is useful, especially if it identifies a systematic limitation and a remedy.

Relevant primary sources:

- Williams, Zahn, and Kutz, Sensing with shallow recurrent decoder networks: https://doi.org/10.1098/rspa.2024.0054
- Tomasetto et al., Reduced order modeling with shallow recurrent decoder networks: https://www.nature.com/articles/s41467-025-65126-y
- Bieker et al., Deep Model Predictive Control with Online Learning for Complex Physical Systems: https://arxiv.org/abs/1905.10094
- Tomasetto et al., Real-time optimal control with shallow recurrent decoder networks, July 2026 preprint: https://arxiv.org/abs/2607.19302
- Manohar, Morton, and Ziade, Sparse sensor-based cylinder flow estimation using artificial neural networks: https://doi.org/10.1103/PhysRevFluids.7.024707

Build a comparison matrix covering observations, physical system, prediction target, control conditioning, optimizer, control constraints, force evaluation, robustness, and runtime. The July 2026 SHRED control preprint describes learning from expert demonstrations; examine its full method before asserting a distinction or novelty. Do not claim the first combination of SHRED and control without a broader literature search.

## 2. Freeze and verify the physical benchmark

The checked-in files specify U_inf=0.5, R=2.5, and nu=1.5e-5, implying Re=U_inf*(2R)/nu approximately 166,667. They also specify a two-dimensional case and a laminar simulation type. Resolve the intended physical interpretation before collecting a publication dataset. A clearly labelled numerical benchmark can support a methods study, but agreement with that solver alone does not establish physical validity at this Reynolds number.

Choose between retaining and justifying the present numerical benchmark, and generating a conventional lower-Reynolds-number two-dimensional benchmark. If changing the benchmark, regenerate the data and do not merge its statistics with the old case. At the present U and D, Re=100 would require nu=0.025; this is an illustrative configuration, not an automatic recommendation to change files.

Record geometry, confinement/blockage, inlet/outlet/wall conditions, cylinder rotation sign, solver and exact OpenFOAM distribution/version, schemes, tolerances, pressure reference, viscosity, mesh, and initial conditions. The checked-in solver headers and application names are mixed, so document the executable actually used.

Report nondimensional time t*=t U_inf/D and rotation ratio alpha=omega R/U_inf. With the checked-in geometry, the 10-second prediction horizon is one convective time and the 5-second application horizon is half a convective time. Measure the shedding period as well; determine how much of that period the observation and prediction windows cover.

Run a small verification campaign before the full dataset:

- At least three meshes, with near-cylinder and wake resolution reported.
- At least two solver time steps; broaden if convergence is unclear.
- Sufficient transient discard and averaging duration based on the observed flow.
- Zero rotation, both rotation signs, and one representative control transient.
- Mean lift/drag, lift RMS, dominant shedding frequency, and relevant pressure statistics.

Document how numerical uncertainty compares with differences claimed between controllers. Match validation references to the actual confinement and flow regime.

Gate: one fixed, documented CFD plant with interpretable coefficients and demonstrated numerical convergence.

## 3. Correct observation, force, and time conventions

Current sensor placement searches around R=0.5, whereas the stored mesh uses R=2.5. The sampled dataset has no cells in the configured sensor band. Also, assigning input_var.sensor_ids does not update the separately imported sensor_ids used for indexing. Return sensor IDs explicitly and pass them through training, inference, and plotting.

Decide whether the sensors are wall-pressure taps or near-wall numerical probes. The existing CSVs contain cell-centre data. If wall taps are intended, export obstacle-patch pressure and locations. Sensor spacing and radial offsets should be expressed relative to the declared geometry.

The existing normalization uses the mean pressure of all leftmost exported cell centres at each time. This is an observation requirement: for a sparse-sensing claim, provide a fixed reference or a specified upstream reference sensor, and count that sensor. Do not derive the online input normalization from hidden full-field data without disclosing this access.

Use explicit variable names and units: raw kinematic pressure, Cp, pressure lift coefficient, total lift coefficient, omega, and nondimensional rotation ratio. Preserve raw data and store normalization metadata.

For diameter-referenced cylinder lift, the pressure contribution is Cl_p=-0.5*integral(Cp*sin(theta) dtheta). The present code approximates the same integral without the factor 0.5. Establish the desired convention, update target/cost scales consistently if it changes, and label legacy results accordingly.

Export OpenFOAM obstacle-patch pressure and viscous force contributions with explicit reference area and directions. Compare:

1. Solver wall-force integration.
2. An independent integration of exported wall pressure.
3. The existing cell-centre-ring approximation.

This separates observable approximation error from surrogate error. If predicting Cp only, evaluate pressure lift separately from total lift. OpenFOAM forceCoeffs documentation describes separate pressure and viscous contributions: https://doc.openfoam.com/2306/tools/post-processing/function-objects/forces/forceCoeffs/

Replace implicit sample indexing with stored timestamps. Assert pressure/control/coordinate alignment, including initial snapshots and table endpoints. Initialize observations and control history from the same CFD trajectory and checkpoint. Store actual past controls instead of inventing a zero history unless it is verified.

Essential focused tests:

- Distinct synthetic timestamp labels prove exactly which samples enter each window and which target is predicted.
- Training and inference construct the same windows. The installed PySHRED lag routine excludes the current input row; account for that convention explicitly.
- No forecast input requires pressure after the controller's current time.
- Changing a future control segment cannot affect predictions preceding its physical application.
- Each candidate CFD branch starts from the same checkpoint and follows exactly the proposed control profile.
- Uniform Cp yields approximately zero pressure lift; an analytic angular Cp profile gives the expected coefficient and sign.
- Preprocessing reads raw pressure consistently and does not normalize an already normalized field a second time.

Gate: one complete controller cycle whose timestamps, observations, candidate controls, force calculation, and CFD restart are auditable.

## 4. Create reproducible experiment infrastructure

Train once per declared model seed; save and reload the model for evaluations. Save the data manager's sensor scaler, coefficient scaler, pre-SVD standardization, basis, coordinate ordering, sensor IDs, dataset IDs, split IDs, and complete configuration. A state_dict alone is insufficient to reproduce physical predictions.

Seed dataset generation, Python, NumPy, PyTorch, trajectory splitting, target generation, observation noise, and scenario sampling independently. Log determinism settings, dependency versions, hardware, source revision or source-file hashes, and CFD settings.

Separate configuration from module-level execution and wildcard imports. Separate training, prediction, plant advancement, candidate evaluation, and reporting. Run directories should be resumable and preserve completed experiments.

The present statistical retraining routine reloads the same dataset and creates a new model. Unless data collection changes elsewhere, it is repeated training on fixed data, not online learning. For the main evaluation, freeze each model throughout its control runs. Treat independent retraining as model-seed variability. A future adaptive experiment needs explicit data acquisition, replay, training, and evaluation separation.

Gate: a saved model reproduces its forecasts after a fresh process starts, with all preprocessing restored.

## 5. Design the dataset and split before model selection

Implemented 500-case redesign: 200 trajectories with successive 25-sample bounded ramps (matching applied MPC commands), 150 with 25-sample ramps plus 25-sample holds (matching candidate forecasts), 100 bounded trajectories with randomly chosen 25/50/100-sample ramps and optional 0/25/50-sample holds, and 50 with ramps followed by random 50/100/150-sample holds. At the user's request all cases start at omega0=0 and retain random history generation. Do not enforce uniform operating-point coverage or reject overlapping histories. The current operating interval is [-0.09, 0.10].

The four families are interleaved by case index with their exact configured counts, keeping all families represented in the existing sequential 400/50/50 partition. This is not an independent-checkpoint split or a coverage-balancing algorithm. Moves are sampled randomly from feasible grid values before generating ramps, rather than clipped into artificial holds. Occupancy and transition plots remain optional diagnostics, not acceptance criteria for histories.

The current MPC moves at most 0.01 in 25 samples, implying a sample-wise rotation increment of 0.0004. Main profiles should respect that limit. Wider changes in the 100 varied-profile cases require proportionately longer ramps (e.g. 0.02 over 50 samples or 0.04 over 100), subject to operating bounds. Immediate full-range jumps are a separate stress test, not coverage required by the present actuator family.

The generator retains the initial shred_lags constant samples within the configured TIMESTEPS intervals. It now exports TIMESTEPS+1 values including time zero and the CFD end time, instead of exporting an unused suffix beyond the end time. DATASET_SEED=None keeps fresh random histories; an integer or the CLI --seed option reproduces each run using base_seed+run_index. Only control schedules and focused tests have been run for this change; CFD must be generated separately. Initial-history length L+D is a data-alignment requirement, not proof of physical settling.

Do not split overlapping windows from the same trajectory across training and test. Fit every scaler and SVD basis on training trajectories only. Reserve a final test suite and do not use it to choose hyperparameters, targets, or loss weights.

The current sequential split places all ramp-profile cases in training and hold-profile cases in validation/test. Keep a profile-held-out study if intentional, but also create a stratified split with both profiles represented in each partition.

Assess coverage of incoming rotation, outgoing rotation, change magnitude, ramp rate, time since a control change, and shedding phase. Equal case counts do not imply equal state-action coverage. The zero-centred random walk is especially likely to concentrate data near zero rotation.

Include control profiles that match the deployed 25-sample ramp followed by 25-sample hold. Test unseen profiles separately: steps, sinusoidal/chirp or varied ramps, and multi-stage profiles within actuator bounds. Treat deliberate extrapolation as a distinct stress test.

Start with existing data after validation. A learning curve at roughly 10%, 25%, 50%, and 100% of the training trajectories helps decide whether more CFD is worthwhile. Keep profile balance and include selection variability when interpreting small subsets.

Establish target feasibility from settled CFD responses at several rotation values and from transient branches. Track the attainable mean-lift range and saturation. A poor response to an unreachable target is not sufficient evidence of surrogate failure.

Gate: documented training/validation/test manifests, coverage plots, and a feasible primary target suite.

## 6. Evaluate compression and predictions independently of control

First reconstruct held-out CFD snapshots directly through the trained SVD basis, bypassing the LSTM. Compare their field and lift errors. This measures the compression floor, including any loss of force information, before attributing errors to the network.

Use physically interpretable metrics:

- Domain Cp relative L2 error, with cell-volume weighting where available, alongside the existing unweighted metric.
- Near-cylinder, surface, and wake errors separately.
- Wall Cp angular profiles and bias, especially their force-producing angular component.
- Pressure-lift absolute MAE/RMSE, bias, peak error, mean error, and oscillation amplitude/frequency error.
- Per-lead-time error from the first to the last predicted sample, with uncertainty across independent trajectories.
- Errors conditioned on control change, incoming rotation, phase, and distance from training coverage.

Relative lift error can become large when reference lift is close to zero. Always report absolute coefficient error too. Normalization by a fixed training-set scale is an option if clearly defined; do not silently use a tiny epsilon to turn near-zero cases into dominant relative errors.

Measure counterfactual response: from the same observed state, vary candidate control and compare predicted changes in mean lift and cost against CFD. A useful controller needs the response to alternative actions, not merely accurate predictions for logged actions.

Useful initial comparisons:

- Pressure persistence and a simple sensor/control-history linear or autoregressive predictor.
- The existing SHRED predictor.
- A direct lift predictor using the same observation/control history and comparable temporal network.

The direct lift model tests the value of reconstructing a full field. If the field approach provides mainly interpretation or additional observables, report those benefits precisely.

Gate: prediction and compression errors are distinguishable, and the model's response to candidate actions is evaluated.

## 7. Evaluate decision quality against a CFD oracle

Retain the current candidate-branching machinery as a validation tool. For a fixed checkpoint, candidate family, objective, and horizon, define a*=argmin CFD_cost(a) and a_hat=argmin surrogate_cost(a).

Report choice agreement, cost-ranking agreement, candidate cost error, and absolute regret R=CFD_cost(a_hat)-min_a CFD_cost(a). Include the gap between the best and second-best CFD candidates. Near-ties can yield disagreements with little practical loss.

The CFD oracle is optimal only over the evaluated candidate set and parameterized profiles. It is not a globally optimal controller. Report the candidate grid, bounds, clipping, and duplicate removal. Repeat selected cases with a finer grid to check search discretization.

Compare mean-error and regret relationships with scatter plots. Analyze high-regret cases rather than only averaging them away. Separate force bias, oscillation error, compression, data coverage, and near-tie effects where evidence permits.

Candidate evaluations from one checkpoint are correlated. Summarize uncertainty over independent trajectories/checkpoints grouped by rollout and model seed, not by treating every candidate row as an independent experiment. Report sample counts at all these levels.

Gate: decision errors have a measured control consequence, with uncertainty that respects the experiment structure.

## 8. Run paired closed-loop experiments

Use identical initial CFD states, reference schedules, and disturbance/noise realizations across controllers. Include several initial shedding phases. Start with a pilot of three model seeds and a few matched scenarios; use observed variability and runtime to set the final replication budget. Five model seeds is a reasonable starting target, not a universal statistical requirement.

Primary controller comparisons:

- Zero or fixed rotation.
- A tuned PI/PID controller using a lift estimate obtainable from the allowed sensors. A PI controller using true solver lift is a privileged-observation comparison and must be labelled as such.
- SHRED-MPC.
- Direct-lift-surrogate MPC with the same candidate family and objective.
- CFD-candidate MPC on an affordable subset, labelled with its greater computational budget.

Evaluate positive and negative setpoints, switches, zero-lift regulation, and feasible targets near actuator limits. Separate tests for infeasible targets. If an upcoming reference change is known, pass the reference trajectory over the prediction horizon; the current scalar target freezes the reference for candidate evaluation.

Choose the control objective before final tests. The current absolute error of mean lift plus lift standard deviation expresses mean tracking and oscillation suppression. If the intended claim is instantaneous trajectory tracking, compare against a pointwise tracking term such as mean((Cl(h)-reference(h))^2). Keep the actual performance metrics identical across objective comparisons.

Report tracking RMSE/MAE and integrated absolute error, steady bias, residual lift oscillations, overshoot and settling for suitable step tests, control total variation, saturation and rate violations, solver/controller failures, and runtime. Control movement is a proxy for actuation effort; claims about energy need torque and mechanical power.

Compute paired differences between controllers and confidence intervals across rollouts/model seeds. Preserve per-scenario traces, rather than reducing everything to a single pooled mean.

Gate: the controller has a measurable advantage, tradeoff, or documented limitation against credible comparisons under matched conditions.

## 9. Select focused ablations and one extension

Use validation data for screening and avoid a full Cartesian product. Prioritize:

- With/without control input and short/long observation history.
- Sensor count (for example 1, 3, 5, 10), multiple placements, and physical reference requirements.
- SVD rank, assessed using both field and force compression errors.
- Pressure-field versus direct-lift targets.
- Original field loss versus force-aware loss.
- Prediction/application horizons expressed in shedding or convective times.
- Matched ramp-hold training versus the current mixed profiles.

Recommended first extension: force-aware learning. Use a differentiable pressure-force operator and train with a scaled combination of field error and lift error. Because force is linear in pressure for fixed geometry, it can often be computed from modal coefficients without constructing the dense field: Cl=b0+b^T a. Account for means, scaling, basis transformation, and wall interpolation. This offers a physically interpretable way to test why field accuracy and control quality differ.

Second extension if coverage dominates errors: acquire CFD branches where model disagreement or decision regret is high, and add those state-action trajectories to training. Compare against random acquisition using the same added CFD budget. Reserve untouched test scenarios throughout. This is a proper adaptive-data experiment if the added data and update schedule are explicit.

Third extension if reliability dominates: an ensemble, calibrated prediction intervals, and conservative candidate selection. Validate uncertainty against actual force/cost errors. Predicted lift standard deviation across a horizon is flow variability, not epistemic model uncertainty.

More expensive optional changes: a sequence model that separately encodes present observations and future control trajectories, richer multi-stage candidate controls, continuous optimization, and latent dynamical models. The current fixed-offset construction limits straightforward horizon changes; changing a horizon may require retraining or a new formulation. Add complexity only to address an observed failure mode.

## 10. Robustness and computational claims

Test sensor noise referenced to a declared signal scale (e.g. 0%, 1%, 5% of training-set sensor standard deviation), one biased/dropout sensor, and delayed observations. Ensure the evaluation noise is never used for hyperparameter tuning. Parameter changes in velocity or viscosity are distinct generalization tests and may invalidate a fixed Cp normalization if not handled consistently.

Measure complete controller decision time: preprocessing, all candidate inference, force evaluation, and optimization. Separately measure CFD integration and export, offline training, preprocessing/SVD, and disk I/O. GPU timings need synchronization and warm-up.

Report median and high-percentile latency with hardware, precision, candidate count, mesh size, and batching. A field decoder speedup alone does not establish controller speedup. Calling the method real-time requires an explicit deadline and a measured computation path that meets it.

Run expensive CFD-oracle validation at selected checkpoints first, then expand where uncertainty remains. Use inexpensive offline evaluations to screen ablations. Estimate runtime and storage from a pilot before launching a large factorial experiment.

## 11. Proposed coding work, in implementation order

These filenames are proposed modules, not files already implemented:

1. `experiment_config.py`: explicit typed configuration, physical scales, separate seeds, versioned experiment IDs.
2. `validate_pipeline.py`: dataset/timestamp/mesh/sensor/reference checks and focused causality/alignment tests.
3. `cfd_observables.py` plus export changes: obstacle wall Cp, pressure/viscous forces, cell volumes, timestamps, and independent force integration.
4. `train_surrogate.py`: explicit trajectory splits, train-only transforms, saved model/preprocessing artifacts, learning histories, and reload validation.
5. `evaluate_forecasts.py`: compression floor, regional and lead-time errors, force metrics, counterfactual response, and direct-lift baselines.
6. `evaluate_candidates.py`: reusable snapshot branching, cost ranks, regret, best-cost margins, and structured records. Refactor the existing statistical code rather than replace verified restart machinery.
7. `run_closed_loop.py`: matched scenarios and controller adapters, frozen models, failure records, computation timings, resumability.
8. `make_paper_figures.py`: vector/PDF figures and CSV tables from immutable result archives, with grouped/paired uncertainty estimates.
9. Only then: force-aware loss, calibrated ensembles, or adaptive data acquisition selected by the diagnostics.

Save large prediction arrays only where needed for diagnostics; retain sufficient raw data, metadata, and paths to reproduce every paper metric. Record missing and failed runs explicitly. Do not silently exclude inconvenient runs.

## 12. Writing milestones and publication evidence

Start the methods draft, notation table, literature comparison, and experiment protocol immediately after conventions are settled. Delay the abstract's performance claims and final conclusions until frozen tests finish.

Expected paper figures:

1. CFD geometry, allowed sensors/reference, and physical/nondimensional scales.
2. SHRED-MPC architecture with exact observation/control/target timestamps.
3. Dataset state-action coverage and held-out tests.
4. Compression and prediction errors versus lead time; surface Cp and force diagnostics.
5. Decision regret versus field/force error, including near-tie analysis.
6. Matched closed-loop trajectories and aggregate controller comparisons.
7. The strongest ablation/extension and its uncertainty.
8. Robustness and controller latency/accuracy tradeoff where space permits.

Expected tables: CFD verification and configuration; model/data/split settings; baseline controller settings; held-out prediction, control, and runtime results.

Minimum coherent paper: verified benchmark, correct sparse observation model and coefficients, reproducible frozen models, independent trajectory splits, field/force/decision evaluations, credible paired closed-loop baselines, and one focused contribution supported by ablation. Extensive architecture searches, online learning, three-dimensional flow, and experimental hardware are optional unless the paper's claims require them.

The first implementation milestone is a synchronized saved-model pilot with correct sensors and wall-force validation. Do that before buying a larger dataset or launching the full statistical campaign.
