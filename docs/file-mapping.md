# Original-to-export mapping

The original project is unchanged. This is a clean review export, not a 135 GB backup. Full datasets and experiment archives remain local; no hard links join editable source files to the original. The export has its own Git boundary and no remote.

| Original path/family | Export destination | Disposition |
|---|---|---|
| `shred_utils/input_var.py` | `src/shred_mpc/config.py` | Preserved defaults; random target generation now explicit, no import-time random draw |
| `shred_utils/functions_dataSet.py` | `src/shred_mpc/dataset.py` | Maintained schedule generation; explicit package imports |
| `test2.py` | `src/shred_mpc/cli/control_table.py` | Actual dataset CLI renamed; existing output refused |
| `shred_utils/pressure_preprocessing.py` | `src/shred_mpc/pressure.py` | Historical normalization retained |
| `shred_utils/temporal_evaluation.py` | `src/shred_mpc/temporal.py` | Isolated evaluation; imports point to maintained modules |
| `shred_mpc_launcher_temporal.py` | `src/shred_mpc/cli/temporal.py` | Portable paths, explicit execution permission, train-only option, complete package source hashes |
| `shred_utils/export_temporal_pressure.py` | `src/shred_mpc/cli/export_pressure.py` | ParaView imported only on explicit execution |
| Pressure-ring functions in `shred_utils/functions.py` | `src/shred_mpc/observables.py` | Same force mathematics; explicit coordinates; cache includes coordinates |
| Boundary table functions in same file | `src/shred_mpc/foam.py` | Pure helpers extracted without importing legacy training/plotting code |
| Existing `tests/` | `tests/` | Same focused tests with maintained imports/CLI paths; added isolation/cache/evidence checks |
| `OpenFoam/{0,constant,system}` | `cfd/templates/training/` | Complete inputs copied, including mesh and control table; no solver output copied |
| `MPC_OpenFoam/{0,constant,system}` | `cfd/templates/mpc/` | Complete historical MPC template inputs copied |
| Pressure MPC/statistical scripts and required helper/export files | `archive/historical-pressure/**/*.txt` | Frozen readable source, restored only by explicit isolated runner |
| Older velocity scripts, `Other Scripts/`, alternative drivers | `archive/legacy/**/*.txt` | Historical lineage, not default runnable entry points; no exact Python duplicates found |
| Local `utils/` | `archive/legacy/utils/` and `LICENSES/SHRED-ROM-MIT.txt` | Unused by maintained workflow; upstream attribution retained |
| July result CSVs and selected NPZ/metadata | `data/curated/` | Duplicate aggregate CSV consolidated to one identical file; source hashes in provenance JSON |
| Original roadmap and temporal notes | `docs/context/` | Dated context, explicitly superseded by verified audit where stale |
| Supplied slides/GIFs | Private inspection notes outside repository | Reviewed, not included in proposed public release; new figures generated from traceable arrays |
| `venv`, caches, bulk CFD, dataset, scratch files | Remain in original only | Excluded from publication; unrelated source snippets also retained in private source snapshot |

`file-mapping.csv` adds exact retained source-to-destination entries and original SHA-256 values. The curated evidence manifest separately lists all input result paths and hashes. No similarly named script was assumed to be an interchangeable duplicate.

## Engineering changes versus scientific changes

Engineering changes: package naming, entry points, import isolation, explicit data/output paths, run ownership, train-only mode, safe plans, offline analysis and documentation. A new dataset CLI replaces the destructive original shell driver for maintained use, using the already-isolated CFD helpers.

One explicit numerical robustness fix keys pressure-ring geometry by coordinates, not only cell count. Tests verify the original convention and a second same-sized rotated grid. Historical fixed-grid values remain unchanged. The benchmark, force normalization, target scales, trained delay and historical result values were not scientifically corrected.

Historical files retain old imports as documentation. The historical wrapper restores their original tree inside a new run and redirects data/cache paths; maintained commands never import frozen text files. No legacy shell driver is a default entry point.
