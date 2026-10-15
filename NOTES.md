# Changes and checks

This export starts from Andrea’s original scripts, not the previous packaged rewrite. The original workspace, full dataset and previous export are preserved. Andrea authorized uploading this revision to the private GitHub repository. Website deployment and public visibility remain unapproved.

## What changed

- Renamed the main entry point to `shred_mpc_launcher.py`; its training and controller behavior is unchanged. Kept `test2.py`, the two case folders and the existing `shred_utils` settings/helpers at their original paths. The controller body and pressure-training function are unchanged.
- Reduced `shred_utils/functions.py` from 75 to 34 functions by retaining the pressure workflow’s dependencies. Removed velocity examples, alternative controllers, statistical/temporal evaluation, unused legacy SHRED code, scratch scripts and generated outputs from this export.
- Added `--train-only`, model saving/loading and `--run-dir` to the existing MPC script. Model artifacts include the fitted scalers/POD transforms and the original case-zero history, so MPC does not retrain or reload the full dataset.
- Added main guards and removed the import-time directory creation. Kept settings in `input_var.py`; no new package, configuration layer or framework.
- Dataset generation now refuses existing output directories instead of deleting them. Control tables are written directly into each newly copied case. The algorithm and default settings are unchanged.
- The supplied MPC `0/` contains `p` and `U`, without `phi`. At the initial time only, the checkpoint check accepts those fields so OpenFOAM can initialize `phi`; later restarts still require it.
- Removed unused ParaView display/style commands from the CSV exporters while keeping the reader → cell centres → pressure CSV operations. A single-frame MPC export is named `CFD_0.csv`, matching the feedback loader.
- CFD meshes, initial fields, boundary conditions and active dictionaries are unchanged. Unused parallel-decomposition/sampling dictionaries and the draft mesh dictionary are omitted. There is no change to the pressure normalization, lift formula, time offset, sensors, candidate commands or objective.

The original initialization takes observations from the end of training case zero while CFD starts from its supplied initial state. Their alignment is not established by these launch checks. The fixed-mesh lift cache and `steadyState` CFD setting are also retained. Changing these would be a separate scientific revision.

## Checks performed

- 13 focused tests: the existing dataset tests, safe imports/output refusal, the initial-state check and a one-epoch synthetic train/save/load comparison.
- Dependency consistency in the installed Python environment, script syntax, CLI help and the dataset dry-run command.
- The complete dataset launcher on one temporary case with two output intervals: meshing, solver, VTK and three pressure CSV snapshots. Original and shortened dataset exporters produced byte-identical CSVs from these snapshots.
- The MPC CFD helper on one temporary output interval, starting with the supplied initial fields and exporting pressure feedback.
- Local page/asset links, project-relative paths, desktop/mobile browser rendering and video metadata.

No full dataset campaign, full training run, complete closed-loop MPC run, scientific validation or GitHub deployment was performed. The short CFD checks changed only temporary test copies. The supplied setup remains at its original defaults. Installation checks used the existing environment rather than downloading a fresh environment.

## Visual material

- **Overview:** unmodified [`media/SHRED-ROM.png`](https://github.com/MatteoTomasetto/shred-rom/blob/main/media/SHRED-ROM.png), from the upstream repository. Its MIT notice is retained in `website/assets/LICENSE-SHRED-ROM.txt` (Matteo Tomasetto – Stefano Riva, 2024). This image illustrates upstream reconstruction, not a SHRED-MPC result. It was not found among the supplied presentation’s raster images.
- **Prediction window:** rendered directly from slide 5 of `SHRED_MPC 13_7_2026.pptx`, then cropped to remove slide furniture. No diagram was redrawn.
- **Video:** `Desktop/SHRED/0001-1249.mp4`, the closest located match to “0-1229.” The top 700 pixels retain the pressure fields, shared colorbar and lift-target history; the lower error panels are omitted. The full 52-second clip is converted to a 768×280 GIF at 5 fps (260 frames). It is embedded directly as the first README image and website visual, with no separate video player or external video link. GIF color quantization and frame-rate reduction affect visual playback only; the original values and timing were not edited. For visitors who prefer reduced motion, the site uses a static frame from the original video. These original media are illustrations, not validation of this export.

The source slides/video remain local. Permission to publicly distribute shared project material and a project-wide license still need Andrea’s decision; no project license has been chosen. The upstream image’s MIT notice does not license the rest of this project.

## Website

The site is static HTML/CSS with local assets, no build tool or external services. `website/index.html` opens directly, or use the localhost command in the README. `.github/workflows/pages.yml` prepares deployment of `website/` to GitHub Pages with a **manual-only** trigger. It does not run on push. Only after approval, configure Pages to use GitHub Actions and run that workflow. A private source repository does not itself guarantee a private Pages website.
