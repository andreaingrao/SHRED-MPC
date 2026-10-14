# Historical source, deliberately separate from maintained code

Files end in `.txt` to prevent import-time training, filesystem mutation or CFD. They preserve the original source for review; they are not installed as a package.

- `historical-pressure/`: current-at-audit pressure MPC, candidate statistics, exports and required helpers. `shred-mpc-historical` restores these into a fresh run and explicitly executes them only with `--execute`. Scientific initialization/force limitations remain.
- `legacy/`: velocity-based predecessors, monolithic SHRED versions, old diagnostics and alternative dataset drivers. Similar names did not establish duplication: no byte-identical top-level/Other Scripts Python files were found. Several old scripts expect obsolete return signatures and are not supported commands.
- `legacy/utils/`: local SHRED-ROM-related implementation, unused by the active PySHRED workflow. Retained with `LICENSES/SHRED-ROM-MIT.txt`; upstream attribution is in `docs/credits.md`.

The unrelated inverse-pendulum example and trivial scratch files are retained only in the private source snapshot outside this proposed public repository. No original file was deleted.
