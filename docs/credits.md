# People and acknowledgments

**Andrea Ingrao — project design and implementation.** Andrea reports responsibility for the CFD/data workflow, control-conditioned pressure predictor, MPC integration, experiment code and analysis.

**Matteo Tomasetto — theoretical guidance.**

**Prof. Andrea Manzoni — theoretical guidance.**

Project context: **MOX, Politecnico di Milano**, as supplied by Andrea. No formal supervision, coauthorship order, institutional endorsement or funding is inferred. Individual public profiles and Andrea’s contact information remain omitted at his request to keep the repository URL private.

The research preparation and documentation distinguish original scientific implementation from repository engineering. The cleaned export reorganizes and guards existing code, extracts reproducible evidence and documents limitations; it does not add a new scientific validation campaign.

## Upstream research and software

SHRED’s architecture originates in Williams, Zahn and Kutz, *Sensing with shallow recurrent decoder networks*, Proc. R. Soc. A (2024), [DOI](https://doi.org/10.1098/rspa.2024.0054), with [primary preprint](https://arxiv.org/abs/2301.12011).

Parametric SHRED-ROM is described by Tomasetto, Williams, Braghin, Manzoni and Kutz, *Reduced order modeling with shallow recurrent decoder networks*, Nature Communications 16, 10260 (2025), [DOI](https://doi.org/10.1038/s41467-025-65126-y).

The active implementation imports [PySHRED](https://github.com/pyshred-dev/pyshred). Local `utils/models.py` and `utils/processdata.py` correspond closely to [SHRED-ROM](https://github.com/MatteoTomasetto/SHRED-ROM). Function AST comparisons matched `fit`, `forecast`, and all inspected preprocessing/plotting helpers; class/import formatting differs. The exact historical upstream commit is unknown. Copies remain archived with [upstream MIT notice](../LICENSES/SHRED-ROM-MIT.txt), copyright 2024 Matteo Tomasetto – Stefano Riva. That upstream notice does not select a license for Andrea’s project.

OpenFOAM and ParaView provide simulation/export; PyTorch, NumPy, SciPy, pandas, scikit-learn and Matplotlib provide learning and analysis. No institutional logos or unverified slide animations are included. No project DOI or formal citation metadata is invented; a stable release citation can be prepared after ownership and publication details are agreed.
