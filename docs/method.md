# Method and conventions

## Observation, conditioning and prediction

Let s(t) be selected cell-centre normalized pressure values, omega(t) the supplied rotation and Cp(t) the full pressure field. For a full-history sample, the inspected PySHRED 1.0.21 convention gives

```text
Cp_hat(t) = F_theta({[s(j-D), omega(j)] : j = t-L, ..., t-1})
```

Current settings use L=50 and D=50. The lag window excludes its current row. At forecast position h after origin k, its latest pressure is k+h-D-1. Positions 1…50 are therefore one fixed-offset model’s deployed prediction block, not separately trained offsets or predictions sharing an identical newest sensor window. The maintained evaluator stores explicit target and measurement indices and forbids future pressure in its inputs.

The pressure target is compressed to 150 components in current defaults, with an LSTM sequence encoder and MLP decoder. A separate uncompressed Parameter output is decoded too. Proposed control is an input supplied by candidate enumeration; the network is not itself an optimized control policy. Same-time reconstruction is the underlying SHRED concept, while the shifted measurements implement this project’s forward predictor.

The July archive contains 100 input rows with 12 pressure channels plus control. Sensor IDs were recovered by unique exact matching against archived preceding CFD branches. Current seven-sensor defaults are a different configuration. Missing historical model metadata prevents asserting all current hyperparameters for the July model.

## Pressure and sensing

OpenFOAM pressure is kinematic (m²/s²). Preprocessing uses `Cp=(p-p_ref)/0.125`, with `p_ref` the mean pressure at the leftmost exported cell centres and `0.125=0.5*(0.5 m/s)^2`. This reference requires additional upstream pressure access; a pure seven/12-sensor physical implementation has not been established. Sensors are volume-cell probes, not wall taps.

CSV `p_raw` is preferred when present. Otherwise maintained training assumes `p` is already Cp; solver exports explicitly declare raw pressure. The raw representative frame, point order and normalization match the curated reference exactly. Never preprocess an already normalized `p` as raw without provenance.

## Historical lift observable

The retained observable is `Lp = -sum(Cp_i sin(theta_i) w_i)` on a heuristically selected near-cylinder cell ring, with angular trapezoidal weights. The saved geometry selects 120 cells outside the radius-2.5 wall. Positive lift is upward. This excludes viscous forces and uses cell-centre rather than wall pressure. It omits the geometric R/D=1/2 factor of a diameter-normalized cylinder coefficient. Targets, errors and costs retain this historical convention; they are not relabelled as total lift.

A focused analytic test gives zero for uniform Cp and −π for Cp=sin(theta). It verifies the implemented convention, not physical wall-force accuracy. The export fixes the geometry cache to include coordinate values rather than only point count; same-grid mathematics is unchanged. A regression test prevents stale reuse for another equal-sized mesh. No force normalization or physical CFD setting was silently changed.

## Candidate objective and receding horizon

For each candidate, ramp from the incoming rotation to its endpoint over 25 samples, then hold to position 50. The archived objective is

```text
J = max(abs(mean(Lp)-target)-0.05, 0)
    + 0.25*std(Lp)
    + 0.02*abs(omega_end-omega_start)/0.01
```

`std` is temporal variation within one forecast, not predictive uncertainty. Historical statistics evaluate five candidate endpoints; normal current MPC uses eleven, clipped to [−0.09,0.10] rad/s within a ±0.01 search radius. The lowest predicted cost supplies only the first 25 samples before feedback and replanning. Regret compares its true CFD cost with the minimum among the same five archived candidates, not a globally optimal controller.

## Alignment and splitting

The maintained evaluator builds full causal histories and records indices. The original controller seeds pressure from the first training trajectory but initializes zero controls and resumes an independently chosen CFD checkpoint. Initial alignment is not proven. Later selected branch histories match all 49 available next-cycle inputs, which supports the curated replay but does not repair initialization retrospectively.

PySHRED partitions whole trajectories sequentially 80/10/10; inspected scalers and SVD fit on training data. The current generator interleaves four control-profile families across the split. Shared initial fields limit independence of incoming physical states. Repeated training reloads fixed data; it is not online learning.

## CFD configuration and physical interpretation

Current training/MPC templates describe a 20 m × 20 m square domain with a radius-2.5 m cylinder, one-cell-thick empty front/back direction and 3,600 cells. Inlet velocity is 0.5 m/s; top/bottom walls are no-slip; outlet pressure is zero; the cylinder has rotating-wall velocity about +z. Viscosity is 1.5e−5 m²/s with laminar simulation type. Nominal Re=UD/nu≈166,667 follows these inputs; physical validity is not demonstrated.

**Both templates use `ddtSchemes/default steadyState`, SIMPLE and relaxation.** October generation logs confirm steady-state iteration, despite increasing solver labels and prescribed control schedules. They do not establish time-accurate transient flow. The new temporal evaluator preserves these dictionaries, so physical transient validation needs an explicitly changed benchmark and new experiment IDs.

Observed command `pimpleFoam` is the OpenFOAM Foundation 13 compatibility wrapper for `foamRun -solver incompressibleFluid`, build `13-18870c24d21c`. Dictionary banners mentioning foam-extend 4 are stale. ParaView 5.11.2 is observed. July branches omit solver logs/dictionaries, so current settings cannot be asserted as their exact historical configuration.

See [scientific audit](scientific-audit.md) for original-source line references and [results](results.md) for archive-specific provenance.
