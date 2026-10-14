# Evidence and slide audit

Audit of the original source, before final export edits. Current contribution/affiliation and release decisions are recorded in [credits](credits.md) and [release](release.md); statements below about pending identities reflect the initial audit stage.
Read-only inspection of `SOURCE_WORKSPACE` and the supplied PowerPoint. Nothing in either original was modified. Extraction and derived samples are in `private-review/slide-inspection/ (outside proposed public files)`. Do not publish the full slides automatically: ownership and collaborator permissions remain unresolved.

## Completed evidence versus configured intention

The sole saved statistical run is `run_20260713_121325_067474`. Its run-local `candidate_statistics.csv` and aggregate `candidate_statistics_all_runs.csv` are byte-identical (SHA256 `d21af1f35a8d2b907672869b2cd47894ec928105faae54c169213b2a7afc8008`). Each contains 250 rows, one model (`run_20260713_121325_067474_training_000`), simulation 0, cycles 0–49, five candidate branches per cycle. Exactly 210 rows / 42 cycles are eligible (`eligible_for_statistics == 1`; controller timestep 250–1275, cycles 8–49). The 40 earlier candidate rows are warm-up, not missing data.

Run metadata specifies 15 simulations × 100 cycles and retraining after 50 completed cycles; these are intended settings, not evidence that those runs completed. `training_events.csv` contains just generation 0. No model checkpoint (`.pt`, `.pth`, `.ckpt`, `.pkl`) was found outside the local environment. There are exactly 250 candidate directories, each with a compressed SHRED/reference archive, metadata, and 50 raw CFD CSV frames: 12,500 CFD CSV frames altogether. This constitutes a complete 50-cycle *partial campaign*, with all candidate branches retained for that prefix.

Repeated training in the statistical script reloads the fixed training data. It is not demonstrated online data assimilation or online learning. The one-generation archive does not support a distribution over independently trained models.

## Numerical results that can be reproduced now

Selection: all 210 eligible rows, same one model and one trajectory. These are correlated candidate evaluations, not 210 independent experiments. Quartiles describe their empirical spread; they are not confidence intervals or seed uncertainty.

| Metric | Median | 25th–75th percentile |
|---|---:|---:|
| Pressure relative L2 over 50×3600 values | 3.05954% | 2.38366–3.66397% |
| Pressure-derived lift trajectory relative L2 over 50 values | 18.42603% | 14.89390–20.27132% |

The rule is `||prediction − reference||₂ / max(||reference||₂, 1e−12)`, flattening the entire horizon (and all field points for pressure). Every one of the 250 stored errors was independently recomputed from its NPZ: maximum absolute discrepancy is `2.23e-14` for pressure ratios and `1.41e-8` for lift ratios (the latter follows float32 archive rounding). It is safe to reproduce these historical values without retraining.

At the *decision* level, 27 of 42 eligible cycles selected a different discrete candidate than the CFD candidate-cost minimum. Count one row with `shred_selected == 1` per cycle; `choice_mistake` repeats across all five rows. The resulting selected CFD cost regret has median 0.01930974, IQR 0–0.05137383, max 0.08571488, in the stored composite objective's units. This supports the distinction between field prediction error and control ranking accuracy. It does not establish closed-loop superiority over a baseline or robustness across seeds.

The archived objective can be verified from row values: `max(abs(mean(Lp)−target)−0.05,0) + 0.25*std(Lp) + 0.02*abs(Δomega)/0.01`. The candidate horizon is 50 samples, ramping for the first 25 and holding for the last 25; only 25 samples are applied before replanning. The statistical grid has five candidates; current normal MPC defaults to eleven. Do not conflate the two configurations.

Avoid pointwise relative lift errors near a zero reference. Use absolute lift error or horizon-relative L2 with a reported reference norm. Eligible reference lift L2 norms range 3.2511–10.6915, so this particular horizon-normalized archive does not contain near-zero denominator horizons.

## Historical sensor recovery and sample integrity

Saved `sensor_input` arrays have shape `(100, 13)`: twelve pressure channels and one control channel. Current `input_var.py` defines seven pressure sensors. The historical twelve indices were recovered from numerical data, not inferred from the slide dots:

`[567, 719, 745, 837, 1050, 2178, 2279, 2890, 3144, 3214, 3351, 3455]`.

At cycles 10, 20, 30, 39 and 40, each sensor input pressure column equals exactly (float32 RMS 0) the corresponding point from the previous four selected-candidate CFD branches, taking each branch's first 25 steps. Thus 6,000 input values across these checks establish the twelve sensor point indices and order. Current seven are a subset; keep this change explicit.

`points_vec.npy` is `(3600,2)` with cell-center bounds ±9.875. Its coordinate order exactly matches `CFD_0.csv` in the selected representative archive. Using the representative raw `p`, subtracting the mean over the leftmost-x cells and dividing by 0.125 produces exactly the archived float32 normalized reference frame 0. Thus field/sample ordering and this normalization are verified for the sample.

The current pressure-derived lift function was safely extracted through AST (no module import) and evaluated on the representative field. It matches saved lift to maximum absolute difference `5.70e-8`. It uses the nearest 120 cell-center points, radii 2.6242–2.6893, angular trapezoidal weights summing to 2π, and `Lp = −sum(Cp sin(theta) dtheta)`. This is a near-wall pressure observable, not viscous-plus-pressure total lift and not a verified dimensional force. A conventional diameter-normalized circular-cylinder lift coefficient would have a different geometric normalization; retain historical values and call the quantity `Lp` or “pressure-derived lift observable.”

## Curated real sample (ready for export)

`data/curated/` has 14 files, ~1.79 MB, including:

- full 250-row candidate CSV, run metadata, training events and target sequence;
- actual points and a full `(50,3600)` predicted/reference representative field archive;
- representative metadata and raw CFD frame 0, for ordering and preprocessing verification;
- recovered sensor indices, coordinates and exact-match checks;
- compact selected tracking arrays in NPZ and CSV;
- metric summary and provenance manifest with original relative paths, source SHA256 hashes, exact transformation rules and output hashes.

Representative selection is deterministic: among the 42 eligible SHRED-selected candidates, choose the one closest to their median pressure relative L2 error, breaking ties by earliest cycle. Result: simulation 0, cycle 39, candidate 0, controller timestep 1025, omega 0.05→0.04, target 0. Its pressure error is 3.03067%; lift trajectory error 14.27639%. Original archive SHA256: `4611f6416f90c3e83ea5e345c1401a05ece1085ea89c8182c189717b307486d6`.

The tracking trace concatenates each selected candidate's first 25 lift/reference/control samples from all 50 cycles, with candidate targets and sample indices 51–1300. All 49 available next-cycle inputs match both the preceding selected pressure branch at twelve sensors and the applied omega exactly. The last branch has no subsequent input for that check. Therefore label the resulting figure **selected candidate-branch replay**, or explain this provenance in its caption, rather than claiming a separately archived main-case lift trace. Target values are −0.5 at controller samples before 500, −1.0 from cycle 18 (controller timestep 500), and 0.0 from cycle 38 (timestep 1000). Each target is held throughout its applied branch.

Regenerate the curated export safely:

```bash
python scripts/curate_evidence.py --source SOURCE_WORKSPACE --output /tmp/new-empty-curated-output
```

The script rejects nonempty output directories, never executes project modules, and needs only NumPy plus the standard library. Copy this script into the clean repository, adapting its default/documented paths, if provenance extraction is part of the public workflow. Figure generation can consume the small curated data immediately without access to the large archive.

## Slides: text, diagrams, images and animations

The PowerPoint has nine slides. Text and relationships are fully extracted in `slide-index.json`. No PowerPoint timing-animation nodes occur; motion is in three embedded GIFs. Diagram geometry was inspected through OOXML and diagnostic raster renderings of slides 2–5. A full native slide rendering was attempted, but installed LibreOffice cannot start because `libreglo.so` is missing. Diagnostic renderings are not faithful theme/layout exports and should stay private.

| Slide | Content | Evidence status |
|---|---|---|
| 1 | “SHRED–MPC”, Andrea Ingrao, 13/07/2026; wake GIF image6 | Title/date confirmed; GIF has no experiment identity |
| 2 | SHRED → optimizer → applied control → CFD → feedback, with candidate controls fed back into SHRED | Useful conceptual loop; replace with readable code-backed diagram |
| 3 | Sensor-history window ending at t, same-time output | Reconstruction concept |
| 4 | Same sensor history, output shifted to t′ | Forward prediction concept |
| 5 | Adds omega conditioning to shifted-output concept | Useful methodological progression; do not treat as timing verification |
| 6 | image7.gif, velocity-like `|U|` reconstruction and error | Older/other workflow; no exact model/data provenance recovered |
| 7 | image8.gif, pressure fields and pressure-lift history | Different target schedule from saved statistical run; no matching raw trajectory identified |
| 8 | Three numerical plots (image9/10/11) | Reconciled individually below |
| 9 | “What’s next?” | No implemented result stated |

Slide 8 image9 says **1,680 eligible candidate simulations** and spans negative/positive omega over most bounds. The current archive plot says **210** and covers a narrower transition region. The two image hashes differ, and no 1,680-row source or additional candidate CSV was found. Do not claim that current CSVs reproduce this slide or speculate that 1,680 equals some multiple of the extant prefix.

Slide 8 image11 names exactly `run_20260713_121325_067474_training_000` and shows pressure median 3.11%, IQR 2.41–3.67%; lift median 18.20%, IQR 14.71–20.23%. Those rounded numbers match the current CSV **eligible prefix through cycle 47: 200 candidate rows** exactly: pressure quartiles 2.40860045%, 3.11327176%, 3.67306760%; lift quartiles 14.71410783%, 18.20359005%, 20.22839033%. Thus this panel is attributable to an earlier snapshot of the retained run, while the final 210-row summary is slightly different.

Slide 8 image10 says 500 omega tables, 250,000 samples, and a 0.01-width zero bin containing 71,564 samples. Current `OpenFoam/CSV_files/case_omega*/omega_t_table.txt` has **500 tables × 451 rows = 225,500 table samples**, with 86,694 samples in the same zero bin rule `floor(omega/.01 + .5)`. These files were regenerated later: case0 CFD log is dated Oct 02 2026. Do not present the slide histogram as a plot of the current dataset.

GIF details and visual inspection:

- `image6.gif`: 800×211, 134 frames, usually 100 ms/frame; unlabelled wake on a long horizontal domain. Not the square domain in the pressure archive; no matching provenance.
- `image7.gif`: 800×450, 150 frames, 100 ms/frame; CFD/reconstruction/error with velocity magnitude scale 0–0.8. Inspected frames include timesteps 25/250/472 and displayed predicted omega −0.088/−1.275/−1.532. These do not identify the pressure workflow. Do not reuse as current pressure evidence.
- `image8.gif`: 1920×1080, 626 frames, 80 ms/frame, 77 MB; CFD and SHRED normalized pressure, lift, target and error. Inspected first/middle/last frames show timesteps 50/673/1297. The target is −1 then 0 then approximately +0.2, whereas the archived statistical trace uses −0.5 then −1 then 0. Similar twelve sensor placement does not make it the same experiment. Do not reuse as a verified numerical trace.

Recommended animation: regenerate a short 50-frame pressure comparison from the curated cycle39 archive, with shared CFD/prediction scales, explicit absolute-error scale, controller-relative horizon index, and a static midpoint frame. This is real saved scientific data with unambiguous provenance and avoids publishing unexplained slide media.

## Proposed six presentation figures

1. Square-domain geometry and actual archived twelve sensors, using `points.npy` and `historical_sensors.json`; radius 2.5 from mesh dictionary, cell-center scatter understated. State current seven-sensor configuration differs. The inlet is left, outlet right, top/bottom walls; do not draw an unbounded external-flow domain.
2. Code-native SHRED-MPC pipeline diagram: historical sensor sequence + candidate control → conditioned decoder → 50-step normalized-pressure field → near-wall pressure observable → composite cost → chosen 25-step rotation prefix → CFD/measure/replan. This is conceptual, not simulation evidence.
3. Representative CFD/prediction/absolute-error fields at one predetermined horizon index (e.g. 24), shared field scale, diverging/accessible pressure colors, sequential error colors; caption full run/cycle/candidate and field metric.
4. Selected first-25 branch replay: predicted/reference Lp and target, applied omega below, sample index on x. Caption branch provenance and one model/one trajectory.
5. Error distribution or cumulative median/IQR over 210 eligible candidate rows; state horizon-relative norms and correlated variability unit. Do not recycle slide's 1,680 heatmap.
6. Candidate-control ranking evidence: cycle39's five CFD and SHRED objective values or all42-cycle regret/decision disagreement; source candidate CSV, configured objective and eligible rules. This makes control quality visible alongside prediction accuracy.

All can be regenerated from ~1.8 MB curated data with NumPy/Matplotlib; no training/CFD installation needed. PNG/WebP for web, SVG/PDF for vector curves, MP4/WebM or small animated WebP plus static fallback for motion.

## Physical and temporal cautions relevant to figures

Current mesh dictionary is square ±10 with cylinder radius 2.5 and 3,600 one-layer cells. Archived points are consistent with that geometry. Current `fvSchemes` uses `ddtSchemes/default steadyState`, and a real case0 solver log confirms SIMPLE steady-state mode. Solver identity is proven by that log: OpenFOAM Foundation 13, build `13-18870c24d21c`, `foamRun -solver incompressibleFluid` invoked via the superseded `pimpleFoam` wrapper. Old foam-extend dictionary banners are not executable-version evidence. A nonzero control update step does not convert a steadyState temporal discretization into validated unsteady flow physics.

The archived branch metadata gives a base CFD file time (175.0 for cycle0; 370.0 for cycle39), and output sampling corresponds to 0.2 increments. Prefer sample indices in public figures until the physical time interpretation and historical CFD settings are checked. There is no saved per-model training configuration/checkpoint proving current training hyperparameters for the July model. Distinguish current source settings from historical facts stored in arrays/metadata.
