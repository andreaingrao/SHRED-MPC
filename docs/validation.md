# Validation record

Prepared 14 October 2026. Verification separates software behavior from scientific validation. This record is finalized with the delivery; detailed logs remain in the adjacent private-review directory, outside proposed public files.

## Checks actually run

- Existing 19 focused tests passed after import/CLI migration: profile family counts, all 500 generated schedules, bounds/rates, actual installed PySHRED lag convention, causal indices, zero-reference metrics, model/preprocessing reload, XLSX output and dry-run/resume.
- **25 focused tests passed in total**, including the original 19. Additional tests passed for import safety, refusal of implicit expensive execution, output overwrite prevention, isolated dataset planning and pressure-ring cache correctness on equal-sized different geometries.
- A one-epoch, ten-trajectory synthetic training-only CLI test fitted and saved a small model, resumed without retraining, and did not invoke CFD. This is a software integration test, not research evidence.
- A three-snapshot isolated OpenFOAM/ParaView smoke test passed through mesh generation, solver, VTK and CSV export, yielding finite arrays of shape (3, 3600) and coordinates (3600, 2). It preserved steady-state settings.
- A local wheel build and installation into a separate temporary target succeeded without modifying the original environment. Source-checkout imports and documented safe CLI plans were exercised.
- Curated source hashes, 250/210 row counts, 42 decisions, pressure normalization/coordinate order, pressure-ring values and headline summaries were verified. The evidence audit independently recomputed all 250 full-archive pressure/lift ratios and matched all 49 available next-cycle histories.
- Six figures and a five-second MP4 were regenerated from real curated data. Outputs include vector curves/diagrams and raster field layers, with fixed comparable field scales.
- Website generation, relative internal links, fragment targets, image alternative text, non-autoplay video and responsive/reduced-motion CSS were checked. Firefox desktop (1440 px) and mobile (390 px) screenshots were inspected; 20 HTTP URLs passed below `/repository/website/`.

## Workflows not claimed as validated research

No full model training, 500-case CFD campaign, full historical MPC run, convergence study, physical transient validation, controller baseline comparison or GitHub deployment was performed. A model round-trip test uses synthetic software-test inputs and is not scientific performance evidence. The tiny isolated solver/export smoke test establishes integration only; it retains the steady-state dictionaries.

Only Python 3.12.3 and the observed dependency versions were exercised. An entirely fresh online installation of all ML dependencies was not attempted. GitHub Pages workflow settings were checked against GitHub documentation; deployment awaits explicit approval.

## Exactly what is reproducible

**Immediately from the export:** read all traceable results, verify the curated sample, regenerate plots/video (ffmpeg optional), create bounded control schedules and temporal plans, build/preview the website, and run focused tests with the listed optional dependencies.

**With additional local data/software:** train a new model on complete consistent CSV exports; generate new CFD/data in owned directories; run the temporal evaluator and historical MPC wrapper. Full training needs substantial RAM/disk. OpenFOAM/ParaView are separate dependencies. The historical wrapper preserves documented scientific limitations.

**Not recoverable from this export alone:** inference from the original July model, because no historical weights/transforms, complete model configuration or exact training-data fingerprint were saved; the slide’s 1,680-candidate figure; validated transient cylinder-flow physics; general performance across independent experiments.

**Needs Andrea’s input before publication:** approval of public release of the exact proposed files and preview, ownership/redistribution confirmation and project license choice. Private upload to `andreaingrao/SHRED-MPC` is authorized; public Pages deployment is not. The full dataset remains local by choice.
