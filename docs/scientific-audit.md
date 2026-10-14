# Scientific and software audit of the source workspace

Audit of the original source, before final export edits. Current contribution/affiliation and release decisions are recorded in [credits](credits.md) and [release](release.md); statements below about pending identities reflect the initial audit stage.
Audit date: 14 October 2026. Source inspected read-only at `SOURCE_WORKSPACE`. No CFD, training campaign, or source-file modification was performed. Python snippets used `PYTHONDONTWRITEBYTECODE=1`; numerical inspection loaded existing arrays only. Line numbers refer to the original source, before export refactoring.

## Defensible project description

The implementation adapts PySHRED as a control-conditioned, fixed-offset pressure predictor, and uses its decoded pressure field in a finite-candidate receding-horizon controller. It does implement sparse pressure sensing, control-history conditioning, LSTM/MLP decoding through a compressed field representation, a pressure-ring lift observable, ramp/hold candidate evaluation, and OpenFOAM advancement.

It is an exploratory CFD/surrogate-control prototype. Current CFD dictionaries specify **steady-state numerical iteration**, despite incrementing the OpenFOAM time label and applying a scheduled rotating-wall control. Neither physical transient-flow fidelity nor a real-time controller is established. Existing statistical archives support a descriptive single-rollout candidate analysis, not general closed-loop superiority, independent experiments for each candidate, or validated total lift.

## Entry points and relevance

| Original path | Verified purpose/status | Proposed disposition |
|---|---|---|
| `shred_MPC_test_p.py` | Current pressure MPC, 11 candidate endpoints by default, 50-position prediction/cost window, first 25 applied. Executes training/CFD on import. | Historical controller retained explicitly; safe guarded runner only after isolated copy and preflight. |
| `shred_test_p_stat.py` | Pressure candidate validation, five CFD branches per checkpoint, durable CSV/NPZ, repeated training every 50 completed cycles, resettable online simulations. Executes at import. | Useful historical validation machinery; preserve separately from reproducible maintained offline analysis. |
| `shred_mpc_launcher_temporal.py` | Guarded argparse launcher, seeded four-family forward-prediction study, dry-run/resume, saved models/transforms, cached CFD reference trajectories, manifest and fingerprints. | Best maintained foundation; portable imports/paths and explicit run directory. |
| `shred_utils/temporal_evaluation.py` | Dataclass config, causal input construction, metrics, dataset checks, model persistence, isolated CFD, XLSX. | Maintained package modules; separate generic metrics from optional ML/CFD imports. |
| `shred_utils/pressure_preprocessing.py` | Raw pressure preservation and normalization, standard-library implementation. | Maintained utility, preserving scientific convention. |
| `shred_utils/functions_dataSet.py` + `test2.py` | Actual currently used bounded control-schedule generation. `test2.py` is a CLI, not a unit test. | Rename to a clear schedule-generation entry point. |
| `OpenFoam/runDataSet.sh` + `OpenFoam/import_input.py` | Current generation driver writes template includes, deletes/recreates case/CSV directories, regenerates mesh, calls solver and VTK, deletes time directories, then exports. | Preserve as historical; safe new generation must require fresh output dirs. |
| `DataSet_to_CSV.py` | Current ParaView dataset exporter, then pressure normalization. Executes on import. | Guarded exporter or retained historical exporter operating on copied cases. |
| `MPC_OpenFoam/CSV_doer_p.py` | Current MPC pressure exporter; selects VTK range via env, exports cell centers, preserves raw p. | Required runtime asset for historical runner. |
| `plot_shred_statistics.py` | Guarded offline figure generator from existing candidate CSV; meaningful reusable analysis. | Maintained CLI or source for curated reporting. |
| `tests/test_dataset_sequences.py`, `tests/test_temporal_evaluation.py` | 19 focused unit/integration tests; schedule family counts/rates, actual PySHRED exclusion of current row, input indices, metrics/zero denominators, reload, dry-run/resume. | Keep/update imports and paths; test summaries must distinguish software checks from CFD validation. |
| `shred_train.py`, `shred_test.py`, `shred_MPC_test.py`, `shred_MPC_test2.py`, `test_2_cycle.py` | Velocity-based predecessor workflows using `States_x`/`States_y`, shared helpers. Scientific lineage useful, but not the pressure workflow. | Archive with explanation, no default entry points. |
| `shred_complete_v1.py`, `shred_complete_v2.py` | Earlier monolithic PySHRED velocity workflows with old paths (`DataSet/CSV_files`, `CSV_files`) and old parameter ranges. | Archive. Similar, **not byte-identical**. |
| `shred_plot.py`, `test.py` | Offline legacy velocity/Numpy archive plotting and optimization diagnostics. | Archive as useful context rather than tests. |
| `test_1.py`, `Other Scripts/bbbbb.py`, `Other Scripts/provamo.py` | Earlier velocity control experiments; unpack three values from the now four-return-value `DataLoad_Training`. `provamo.py` also unpacks only two at its first call. Not runnable against current helper API. | Archive with incompatibility note. |
| `Other Scripts/baboh.py` | Contains useful explicit history-index functions, but three-return-value mismatch at line 61 and obsolete `omega_extreme`/`N_omegas_decoded` assumptions; no import guard. | Archive for indexing lineage; not current maintained implementation. |
| `Other Scripts/erg.py`, `Other Scripts/brbrb.py` | Small data-load diagnostics executing at import. | Archive, not automated tests. |
| `Other Scripts/runParallel.sh`, `Other Scripts/runDataSetSampled.sh`, `varie_OpenFoamDataset/*` | Alternate older shell generation drivers; active driver is `OpenFoam/runDataSet.sh`. | Archive only, preserve source in complete private duplicate. |
| `utils/models.py`, `utils/processdata.py` | Local legacy SHRED implementation; no maintained Python file imports these. Close upstream SHRED-ROM correspondence verified; attribution needed. | Archive with upstream license/notice rather than duplicate as maintained model. |
| `shred_utils/inverse_pendulum.py` | Unrelated inverse-pendulum example. | Exclude from scientific source export or archive explicitly as unrelated example. |

SHA-256 grouping across top-level and `Other Scripts` Python found **no exact duplicate files**. Do not consolidate similarly named scripts as if identical. Most duplicate functionality is historical iteration, not interchangeable code.

## Scientific indexing and sensing

`shred_utils/functions.py:494–548` shifts target pressure and control forward by `meas_offset=50`, while pressure measurements retain their earlier indices. `ParametricDataManager` prepares lagged windows. The installed PySHRED 1.0.21 routine (`pyshred/processor/utils.py:generate_lagged_sensor_measurements`) excludes the current input row; this is also asserted in `tests/test_temporal_evaluation.py:77`.

For complete histories, the deployed formulation is:

`Cp_hat(t) = F_theta({[s(j-D), omega(j)] : j=t-L,...,t-1})`.

Here L=50, D=50 in current settings. The latest measurement entering position h after origin k is k+h-D-1. A curve over h=1,...,50 is a curve within one fixed-offset deployed prediction block; it is not a comparison of separately trained lead times or of identical latest observation windows. The final supplied pressure sample at origin is excluded from the final forecast's lag window.

`temporal_evaluation.py:176–192` constructs target stream indices explicitly and requires all pressure input to be observed by origin; `EvaluationSettings.validate:60–61` rejects horizons above offset. `shred_mpc_launcher_temporal.py:215–236` slices only observed pressure, compares the next horizon reference rows, and saves input/target indices. This is stronger provenance than historical MPC. The exporter checks expected timestamp-labelled VTK paths, but the training CSV validator checks table timestamps and file counts; it does **not** prove a timestamp present inside every historical CSV.

Historical MPC at `shred_MPC_test_p.py:71,108–111` and statistics at `shred_test_p_stat.py:998,1030–1037` use the last L+H measurements. `DataLoad_Training_p` returns the **entire first training case** as starting state history (`functions.py:2477`), while the controller initializes only L zero controls (`shred_MPC_test_p.py:34–41`). The live CFD checkpoint is independently chosen by latest time (`functions.py:2176`). Original synchronization of those three streams is not established. Archive metadata for cycle 0 records base CFD label 175.0, while controller label starts at sample 50. Do not silently equate controller sample/5 with absolute CFD time.

Current configured sensor IDs are `[837,2178,2279,3144,3214,3351,3455]` (`input_var.py:11`); `sensor_maker_p` at `functions.py:344–352` uses them directly. The `get_surface_sensors` helper at line 1818 has **no calls anywhere in current Python**, and its default R=0.5 conflicts with the actual mesh radius 2.5. Current sensors are volume-cell observations, not cylinder surface taps. Their radii are approximately 3.375, 5.735, 8.394, 8.629, 3.644, 8.517, 4.277.

July candidate NPZs have `sensor_input.shape=(100,13)`: twelve pressure inputs plus omega, not seven pressure inputs. Exact matching of the newest 25 pressure observations against the preceding selected CFD branch's first 25 fields recovers historical IDs `[567,719,745,837,1050,2178,2279,2890,3144,3214,3351,3455]`. Independently checked at cycles 1,8,26,49 with RMS difference zero for all channels. Label these **inferred from archived arrays**, since the metadata does not explicitly store sensor IDs; uniqueness should be checked by the figure provenance code.

## Pressure normalization, sparse observation assumption and force observable

`pressure_preprocessing.py:5–6,12–34`: q=0.5*(0.5)^2=0.125; reference pressure is the mean over the **leftmost cell-center x coordinate**, within x-range-dependent tolerance; Cp=(p-p_ref)/q. OpenFOAM p has kinematic dimensions [0 2 -2 0 0 0 0], so density does not appear. This is a prescribed normalization and does not prove that these cells constitute a physical freestream reference. Sensing therefore implicitly needs this reference in addition to selected sparse cell pressures.

`processed_pressure_from_csv` prefers `p_raw` if present; `preprocess_pressure_csv` creates/preserves that column, making its repeated processing idempotent when raw metadata exists. If only already-normalized `p` exists, raw/normalized ambiguity remains. Historical `data_load_p` reads `p` without normalization and trusts prior export. New temporal `read_pressure_csv` uses p_raw when present, normalizes raw exports, and otherwise assumes p is normalized unless explicitly told raw. Do not apply normalization twice without column provenance.

`functions.py:808–875` selects a nearest-center ring by a radial gap heuristic, sorts angularly, and applies trapezoidal angular weights. For saved 3600-cell geometry, 120 selected cells span radii 2.6242–2.6893 around wall radius 2.5. The reported observable is

`L_p* = -sum(Cp_i sin(theta_i) w_i)`.

It uses cell-center pressure, not exported wall pressure, and excludes viscous force. Positive sign convention is upward. For standard cylinder coefficient normalized by diameter, the geometric factor R/D=1/2 is missing: use `L_p*` or explicitly “legacy pressure-ring lift observable,” not unqualified total lift or validated C_L. Existing target/cost scales belong to this convention and must be preserved with historical results.

An analytic numerical check of the existing ring yielded uniform-Cp lift -9.13e-17, sum weights 2π, and Cp=sin(theta) lift -π. This confirms sign/integration implementation while illustrating factor-of-two relative to the diameter-based coefficient (-π/2). It is not wall-force validation. Geometry caching currently keys only point count, so a different mesh with the same cell count reuses stale geometry. Fixing that cache key is a scientifically neutral bug fix for multi-mesh use if tested; keep the historical force formula unchanged.

## Predictor, splitting and conditioning

Current PySHRED model uses default LSTM sequence model, default MLP decoder, no latent forecaster, 150 pressure compression modes, and a separate uncompressed Parameter output (`functions.py:502–543`, `temporal_evaluation.py:310–331`). The control is both input and decoded output; do not imply a learned control policy or independently predicted optimized action. Optimization is finite candidate enumeration.

Installed `pyshred/processor/parametric_data_manager.py:104–107` splits **whole trajectories sequentially** 80/10/10. Training data alone fit pre-SVD StandardScaler and randomized SVD (`162–179`), output MinMaxScaler (`238–245`), sensor MinMaxScaler (`251–267`). No cross-split overlapping-window leakage is evident from this implementation. However histories start from shared initial conditions, so independent checkpoints/physical initial states are not established. Original scripts do not persist split IDs, seeds, preprocessing or weights. The temporal launcher seeds Python/NumPy/Torch, persists preprocessing registries and weights, and fingerprints dataset files, but deterministic-algorithm settings and every split ID are not explicitly persisted.

The redesigned current generator (`functions_dataSet.py:19–41`) interleaves four families: 200 successive-ramp,150 ramp/hold,100 varied-ramp/optional-hold,50 extended-hold. The older roadmap sentence saying all ramp cases are training and hold cases validation/test is stale relative to current generator. The claim that dataset is currently incomplete is also stale: all 500 CSV case dirs currently have 451 CSV snapshots and 451 control rows. This count is not a full validation of every field's contents. Existing tests verify schedule generation, not the historical physical data campaign.

Current tables cover 450 intervals at nominal 5 outputs per second, including initial/end rows; max rotation increment .0004 per saved sample, omega bounds [-.09,.10]. Because current ddtScheme is steadyState, express these as nominal solver labels/cadence rather than validated physical temporal resolution.

## Candidate objective and repeated training

`shred_MPC_test_p.py:88–145` searches 11 clipped/unique endpoints within ±.01, each reached with 25-sample ramp then 25-sample hold. Cost is `max(|mean(L_p*)-target|-.05,0) + .25*std(L_p*) + .02*|omega_end-omega_start|/.01`. `std` is temporal variability over one predicted window, not model uncertainty. The chosen first 25 controls are applied; next sensor history updates and the search repeats.

The statistical variant uses five feasible endpoints (`shred_test_p_stat.py:470–493`), branches all candidates from the same checkpoint (`functions.py:2268–2425`), and restores current U/checkpoint/files after each branch. This machinery is useful, but acts on a mutable case and has no import guard. New explicit owned-case execution is essential.

`retrain_shred_predictor` (`shred_test_p_stat.py:347–369`) reloads the same training dataset and fits a new model. No new CFD observations are added to that dataset here. This is **repeated training on fixed data**, not online learning. Multiple training counts in the temporal launcher likewise measure model-seed variability against reused reference trajectories.

## Current physical CFD configuration

- Mesh: 3600 cells; 7440 mesh vertices; domain [-10,10]×[-10,10]×[-.5,.5] in dictionary length units with `convertToMeters 1`; cylinder radius2.5, diameter5. Exact mesh file bytes match OpenFoam and MPC_OpenFoam. `0/p` files match; `0/U` differs in boundary table include.
- Inlet U=(.5,0,0); outlet U zeroGradient, p fixed0; top/bottom walls U fixed0; rotating cylinder origin0, axis+z, omega table; frontAndBack empty (`OpenFoam/0/U:3625–3656`, p same region). Initial fields are nonuniform inherited solutions, not zero rest states; original generation history is not separately documented.
- nu=1.5e-5 (`constant/transportProperties:19`), `simulationType laminar` (`constant/turbulenceProperties:17`). The implied dimensional Re=U D/nu≈166667 is not a demonstrated physical turbulent benchmark, especially under 2D laminar steady iteration.
- **Both template `system/fvSchemes:17–20` specify `ddtSchemes { default steadyState; }`. Both fvSolution files specify SIMPLE and relaxation p=.5,U=.1.** All 500 archived training cases have identical fvSchemes/fvSolution/transportProperties hashes. Representative `OpenFoam/case_omega0/log.pimpleFoam:47–48` explicitly reports steady-state mode/SIMPLE mode. The scheduled changing control therefore does not establish a time-accurate unsteady flow simulation.
- `pimpleFoam` is the command executed (`OpenFoam/runDataSet.sh:52`, `functions.py:2211`). Actual current log says Foundation OpenFOAM13 build `13-18870c24d21c`, `Exec: foamRun -solver incompressibleFluid`; the pimpleFoam command is a compatibility wrapper. Do not infer foam-extend4 from stale dictionary banners.
- Statistical July branches retain pressure NPZ/CSV and minimal metadata, but no fvSchemes/fvSolution or solver logs were found under their archive. Today's template and October dataset logs do **not** prove the precise July physical setup. Retain this provenance distinction.
- Temporal `prepare_cfd_case` copies original system settings and rewrites controlDict only (`temporal_evaluation.py:369–394`), so it also inherits steadyState unless explicitly changed in a new scientific experiment.

## Available evidence and metadata limits

`shred_stat_results/run_20260713_121325_067474` is one available candidate run. Its `candidate_statistics.csv` is byte-identical to `shred_stat_results/candidate_statistics_all_runs.csv` (do not double-count). Each contains250 rows,210 eligible; one simulation0;50 checkpoints/cycles;42 eligible checkpoints;one model ID `run_20260713_121325_067474_training_000`. Eligibility is controller_timestep≥250. `training_events.csv` contains only training000 event. Configured15 simulations×100 cycles is a **plan**, not completion; no evidence of the first retraining being completed.

The July metadata has grid5,control horizon25,prediction horizon50,bounds/search radius, saved points and branch references. It lacks model weights, transforms, seed, code hash, sensor IDs, exact training data identity, dependency snapshot, model hyperparameters and full CFD dictionaries. No `.pt` model or transform files were found under this result run. The current config and October dataset must not be assigned retroactively as its exact training config.

Eligible candidate descriptive summaries (210 correlated rows, single trajectory/model): pressure relative L2 median .0305954294195 (Q1 .0238366,Q3 .03663969); lift trajectory relative L2 median .184260342265 (Q1 .14893897,Q3 .2027132). Fields are flattened across50×3600; lift across50. These are descriptive pooled values, not confidence intervals. At42 eligible decisions,27 selected endpoints differ from the minimum CFD-cost endpoint; median absolute regret .0193097409281,max .0857148763655. A disagreement may have small cost due to near-ties, so regret is more informative than binary choice error alone. CFD minimum is only an oracle over the **five evaluated candidates**.

Historical relative L2 divides by max(norm(reference),1e-12), which can inflate values at near-zero reference. In this eligible archive lift-reference norms range3.2511–10.6915, so these particular window denominators are not near zero; always pair with absolute L2/RMSE. Relative normalized regret also divides by an epsilon-clipped oracle cost and should not be featured without absolute regret. Temporal metrics correctly return blank/None for effectively zero reference norms and retain absolute metrics.

The slide figure's1,680 eligible candidates cannot be recovered from these250/210-row files. It must remain “unmatched slide evidence” until its source is found; neither extrapolate nor relabel these rows to match. Slide inspection belongs to the evidence agent.

No `temporal_eval*` output dirs were present at source root. Four-family evaluation and model persistence are implemented, but no completed scientific temporal study was found. Synthetic/reload tests documented in TEMPORAL_EVALUATION are software implementation checks, not performance results. The paper roadmap's baselines, force-aware training, paired controller comparisons, active learning, uncertainty, noise robustness, mesh/time validation and speedup remain proposals unless separately evidenced.

## Dependencies and upstream origin

Observed environment metadata: Python3.12.3 (`venv/pyvenv.cfg`), PySHRED1.0.21, Torch2.9.1, NumPy1.26.4,pandas2.3.3,Matplotlib3.10.8,SciPy1.16.3,scikit-learn1.8.0. These are observed installed versions, not invented minimum requirements. `pvpython --version` reports ParaView5.11.2 (read-only version command required sandbox escalation because MPI needs local sockets). `/opt/openfoam13/bin/pimpleFoam` and `/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin/foamRun` are present. Full training/CFD was not rerun.

`utils/models.py` and `utils/processdata.py` strongly match SHRED-ROM upstream implementations (same class structure, defaults, fit/forecast functions and preprocessing/plot helpers; local models adds an unused IPython import and whitespace). They are not the active PySHRED engine. Primary sources inspected:

- SHRED-ROM code: https://raw.githubusercontent.com/MatteoTomasetto/SHRED-ROM/main/utils/models.py and https://raw.githubusercontent.com/MatteoTomasetto/SHRED-ROM/main/utils/processdata.py . Exact historical commit not established; describe correspondence, not invented original authorship or immutable provenance.
- Upstream license: https://github.com/MatteoTomasetto/SHRED-ROM/blob/main/LICENSE : MIT, copyright2024 Matteo Tomasetto - Stefano Riva. Preserve its copyright and permission notice with copied substantial code. Do not assign this license automatically to Andrea's whole project.
- Active package: https://github.com/pyshred-dev/pyshred . Installed METADATA identifies MIT; authors David Ye,Jan Williams,Mars Gao,Matteo Tomasetto,Stefano Riva,Nathan Kutz. Installed license exists under package dist-info.
- Williams,Jan P.; Zahn,Olivia; Kutz,J.Nathan. *Sensing with shallow recurrent decoder networks*. Proceedings of the Royal Society A480(2298),20240054(2024), https://doi.org/10.1098/rspa.2024.0054 . Verified publisher search page and primary arXiv https://arxiv.org/abs/2301.12011 .
- Tomasetto,Matteo; Williams,Jan P.; Braghin,Francesco; Manzoni,Andrea; Kutz,J.Nathan. *Reduced order modeling with shallow recurrent decoder networks*. Nature Communications16,10260(2025), https://www.nature.com/articles/s41467-025-65126-y . Verified published21November2025, DOI10.1038/s41467-025-65126-y. Use published citation rather than now-stale PySHRED METADATA's preprint status.

These upstream authors are separate from project collaborators. No inference about Andrea's collaborators, supervision, coauthorship, institutional endorsement, funding or project-wide rights is justified by these sources. Project license, shared-code/data/slide permissions, exact personal contributions and collaborator details remain user decisions.

## Reproducibility boundaries and recommended publication claims

Immediately reproducible from curated existing data: figure regeneration, candidate descriptive metrics, regret analysis, historical pressure-field/lift-window comparisons, sensor-location recovery, pipeline and geometry diagrams. No trained network is needed for these.

With optional ML dependencies: causal-input/schedule/metric tests and small synthetic model persistence checks; call them implementation tests. Current dataset can support training after complete numeric/timestamp/coordinate validation, but its500×451 files are a different provenance item from July results. A float32 pressure tensor alone is500×451×3600×4≈3.25GB, before PySHRED arrays, tensors, SVD and copies, so avoid implying laptop-light full training memory.

With OpenFOAM/ParaView and fresh run-owned case dirs: regenerate the **preserved numerical setup** and evaluate models against its fields. Physical transient validation would be a separate, explicit scientific correction and new experiment series; never silently change steadyState to Euler/backward and reuse historical claims.

Publication should await collaborator/contribution identities, repo/contact URLs, license/rights choices, and approval of concrete public contents. Missing physical validation should be explained as a limitation, not a reason to hide existing data or inflate its meaning.
