# Stochastic Rayleigh-wave Newtonian noise for a spherical cavity

This chapter documents `getSphNNStochRay.ipynb`. It extends the single-wave
Rayleigh validation to an isotropic stochastic field composed of many plane
Rayleigh waves with random propagation azimuths, amplitudes, and phases. The
notebook calculates realization-level displacement and Newtonian acceleration,
forms ensemble RMS spectra, compares the numerical coupling with an analytic
Rayleigh NN model, scales the result to measured site motion, and projects it
onto Einstein Telescope strain.

Read `docs/examples/getSphNNSingleRay.md` first for the underlying Rayleigh
eigenfunction, half-space coordinate convention, numerical integration terms,
and single-wave validation.

## What the example demonstrates

The notebook shows how to:

- synthesize a stochastic, azimuthally distributed Rayleigh-wave field;
- calculate independent realizations with frequency-dependent random phases
  and amplitudes;
- save displacement and NN output per realization;
- assemble ensemble RMS surface and underground displacement;
- calculate surface-to-depth attenuation;
- normalize NN to unit vertical surface displacement;
- compare horizontal NN with the analytic isotropic Rayleigh prediction;
- scale displacement and NN to measured site ASD; and
- estimate the corresponding ET strain ASD.

## Physical scope

This example is restricted to a homogeneous elastic half-space. The free
surface is at `z = 0`, and the medium occupies the region below it. Rayleigh
motion decays with depth according to the homogeneous P- and SV-potential
eigenfunctions.

The notebook sets:

```python
fullSpace = False
```

and this value should remain false. A layered surface-wave field must use the
Qseis full-field workflow instead of the analytic `if_ray` mode.

## Required files

The notebook expects:

- `configParseHalfSpaceRayHomo.ini` under the stochastic Rayleigh tutorial;
- `SingleSphCavityHalfSpace.ini`;
- site reference files under `TERZ/`;
- the ET-D design curve under `ETDesign/`; and
- writable metadata, temporary-displacement, and realization-output paths.

Update the absolute paths in both notebook cells when running elsewhere:

```python
configFile = "/path/to/configParseHalfSpaceRayHomo.ini"
allCavities = configLoader.loadCavities(
    "/path/to/SingleSphCavityHalfSpace.ini"
)
```

## Tutorial geometry

The active tutorial configuration describes:

- a homogeneous medium with `vp = 4000 m/s`, `vs = 2000 m/s`, and
  `rho = 2465 kg/m^3`;
- a horizontal domain from -3000 to +3000 m in x and y;
- a half-space integration depth of 3000 m;
- a spherical cavity with radius 20 m at 250 m depth; and
- one selected test mass at the cavity centre.

The main calculation reads all four configured ET test masses but selects only
`itm_ac`:

```python
itmAll = np.vstack((itmACUse,))
```

Its positive configured depth is converted to the internal negative Cartesian
z coordinate before NN integration.

## Wavefield flags and stochastic settings

The configuration selects analytic Rayleigh mode:

```ini
[simFlags]
if_body = False
if_ray = True
if_full_field = False
save_hv = False
decoupled_hv = True
on_fly_nn = True
```

The stochastic ensemble uses:

```ini
[simSource]
n_rea = 100
n_src = 100
src_distri = internal
scale_vh = 1.0
```

Thus, every realization contains 100 Rayleigh plane waves, and the final RMS
is assembled from 100 independent realizations. This calculation can take
considerable time because the full volume and boundary NN integrals are
repeated for each realization.

The required source and realization counts should be established through
convergence tests rather than assumed from these tutorial values.

## Stochastic Rayleigh-wave generation

At the beginning of every realization, the notebook calls:

```python
xSrc, ySrc, azSrc, srcMeta = simDisp.genAmbSrc(
    nSrc,
    mode=srcDistri,
    R1=R1,
    R2=R2,
    xMin=xMin,
    xMax=xMax,
    yMin=yMin,
    yMax=yMax,
    randomPhase=True,
    freqDependent=True,
    nFreq=nFreq,
    decoupledHV=config.decoupledHV,
    scaleVH=scaleVH,
)
```

For the Rayleigh field, the important outputs are:

- `azSrc`: propagation azimuths drawn uniformly from `0` to `2*pi`;
- `srcMeta["ampV"]`: amplitudes drawn uniformly between zero and one for
  every source and frequency; and
- `srcMeta["phiV"]`: phases drawn uniformly between zero and `2*pi` for every
  source and frequency.

The notebook passes these arrays to the analytic Rayleigh displacement model:

```python
phiSrc = srcMeta["phiV"]
ampSrc = srcMeta["ampV"]
```

`xSrc` and `ySrc` are part of the shared source interface but do not set the
spatial origin of the analytic plane waves. Their `internal` distribution is
therefore not the controlling geometry for this Rayleigh calculation.

The random wave azimuths produce an approximately isotropic horizontal field
as the source and realization counts increase. Each wave retains the coupled
horizontal and vertical displacement dictated by the Rayleigh eigenfunction;
these are not independent horizontal and vertical wavefields.

No random seed or source metadata is saved. Repeated executions produce
statistically equivalent but not identical ensembles, which is acceptable for
the intended RMS calculation.

## Rayleigh phase velocity and depth decay

The fundamental-mode phase velocity is computed with `pysurf96` at every
selected frequency:

```python
periods = 1 / freqOut
vR = surf96(
    thickness,
    vp,
    vs,
    rho,
    periods,
    wave="rayleigh",
    mode=1,
    velocity="phase",
    flat_earth=True,
)
```

`simDisp.computeRayleighDisp` uses `vR`, `vp`, and `vs` to construct the
frequency-dependent horizontal wavenumber and the evanescent P- and
SV-potential depth factors.

## Qseis geometry metadata

As in the other analytic-wave examples, the notebook prepares `splitAll` and
`splitAllNew` with:

```python
splitAll = createDB.createFinalInputs(config)
```

and `gfLoader.assembleQseisRuns(..., onlySplitAll=True)`. These objects provide
geometry metadata required by the shared numerical workflow. Qseis does not
generate the stochastic Rayleigh displacement field, and the Qseis execution
line remains commented out.

## Adaptive geometry and integration

The grid and spherical-cavity refinement follow the earlier examples:

```python
etaUse = 0.5
minRefine = 20.0
```

The calculation associates the selected test mass with the cavity, constructs
a refined region around the cavity wall, partitions the medium into layers and
blocks, and excludes points inside the cavity.

For each realization, it evaluates:

- `IBlockTot`: material-volume contribution;
- `IVertFaceTot`: vertical outer-boundary contribution;
- `IHorFaceTot`: horizontal-interface and free-/bottom-surface contribution;
  and
- `ITot = IBlockTot - IVertFaceTot - IHorFaceTot`.

After applying Newton's gravitational constant, the arrays and frequency
vector are saved as:

```text
NNFullRea<realization>.npz
```

## Surface and underground displacement output

For every realization, the notebook evaluates displacement at:

```python
zList = [0.0, cavDepthUse]
```

and saves:

```text
surfDeepDispRea<realization>.npz
```

For the supplied cavity, the two locations are the free surface and 250 m
depth. These outputs are later used for ensemble RMS, attenuation, and
frequency-dependent scaling.

## Separating simulation and post-processing

The first notebook cell performs the expensive realization loop. The second
cell reads the saved files and can be rerun without repeating the NN
integration. This separation is useful for changing plots, scaling choices, or
analytic comparisons after a completed simulation.

The post-processing cell must use the same main and cavity configurations as
the simulation cell. It reads `n_rea` from the configuration and expects a
complete numbered sequence of output files.

## Ensemble RMS assembly

The displacement ensemble is assembled with:

```python
dispPointAllRea, attnAllRea, freqDisp = (
    simDisp.assembleSurfDeepDispAllRea(
        outDispRea, zList, freqOut, nRea
    )
)
```

The NN ensemble is assembled with:

```python
nnTotAllReaAllTM, nnVolAllReaAllTM, nnSurfAllReaAllTM, fNN = (
    NNCompute.assembleNNAllReaAllTM(
        outDispRea, freqOut, nRea, nTM
    )
)
```

The routines combine realization magnitudes in quadrature and take their RMS.
They do not coherently average complex phases between independent
realizations.

The notebook then selects the first and only simulated test mass:

```python
nnTotAllRea = nnTotAllReaAllTM[:, :, 0]
```

## Displacement scaling

Two forms of displacement scaling are produced.

### Scaling to observed surface motion

```python
scaledSimDisp, *siteASDs = NNPP.scaleDisp(
    dispPointAllRea,
    freqOut,
    config,
    scaleComp="Z",
    scaleDepth="surface",
    scaleToTrue=True,
)
```

The vertical surface component is matched to the measured surface vertical ASD
at every frequency. The same scale factor is applied to all components and to
both depths, preserving the simulated component and depth ratios.

### Unit vertical surface displacement

The second call uses `scaleToTrue=False`. It normalizes the simulated vertical
surface component to unity and exposes relative polarization and depth
attenuation without imposing the measured site amplitude.

The notebook plots both scaled variants at the surface and at 250 m depth.

## Surface-to-depth attenuation

`NNPP.getAttn` calculates the simulated surface-to-depth ratios and loads the
observed attenuation data from `attnModel.mat`:

```python
attnSim, attnX, attnY, attnZ = NNPP.getAttn(
    dispPointAllRea, freqOut, config
)
```

The attenuation comparison checks whether the analytic homogeneous Rayleigh
field has a depth dependence resembling that observed at the selected site. It
is a diagnostic and does not alter the NN integration.

## Unit-displacement NN and analytic comparison

The total numerical NN is normalized to unit vertical surface displacement:

```python
scaledNN, *__ = NNPP.scaleNN(
    nnTotAllRea,
    dispPointAllRea,
    freqOut,
    config,
    scaleComp="Z",
    scaleDepth="surface",
    scaleToTrue=False,
)
```

The analytic stochastic Rayleigh benchmark is:

```python
NNRay, freqNNRay = NNCompute.getAnalyticRayNN(config, 250)
```

The depth should preferably be supplied as `cavDepthUse` rather than
hard-coded. Unlike the deterministic single-wave comparison, the stochastic
notebook does not apply an additional `sqrt(2)` factor.

The analytic Rayleigh curve is plotted against the x and y numerical
components. For a sufficiently converged isotropic ensemble, the two
horizontal components should be statistically equivalent. The notebook does
not plot the same analytic curve on the z panel.

Although `getAnalyticBodyNN` is also called, its returned body-wave curves are
not plotted in the current cell and are not part of the Rayleigh validation.

## Site-scaled NN and ET-D projection

The physical acceleration ASD is obtained by scaling the total NN with the
observed vertical surface motion:

```python
scaledNNToTrue, *__ = NNPP.scaleNN(
    nnTotAllRea,
    dispPointAllRea,
    freqOut,
    config,
    scaleComp="Z",
    scaleDepth="surface",
    scaleToTrue=True,
)
```

It is projected to an approximate interferometer strain ASD using:

```python
nnProj, ETD = NNPP.NNProjectToET(
    scaledNNToTrue, fNN, config
)
```

The default projection assumes a 10 km arm and accounts for four independent
test-mass contributions with a factor of two in ASD. This is an approximate
projection from one simulated test-mass spectrum, not a spatially correlated
four-test-mass calculation.

## Running the notebook

1. Select the homogeneous half-space stochastic Rayleigh configuration.
2. Select the matching spherical-cavity file.
3. Update all paths and verify the site ASD and ET-D reference files.
4. Confirm `if_ray = True`, `fullSpace = False`, `n_src = 100`, and
   `n_rea = 100` for the supplied tutorial.
5. Run the first cell and monitor every realization and integration block.
6. Verify that each realization produced both a displacement file and an NN
   file.
7. Run the second cell without changing its configuration paths.
8. Inspect unit-normalized displacement and surface-to-depth attenuation.
9. Compare x and y unit-normalized NN with the analytic Rayleigh curve.
10. Inspect the site-scaled displacement, acceleration, and ET-D projection.
11. Repeat with modified source, realization, domain, and grid parameters to
    establish convergence.

## Interpretation and convergence checks

Before accepting the result, verify that:

- x and y ensemble RMS displacement and NN approach statistical equivalence;
- increasing `n_src` improves azimuthal isotropy;
- increasing `n_rea` stabilizes the ensemble RMS spectra;
- the y/x differences are consistent with finite-ensemble fluctuations;
- `lambda_res`, `etaUse`, and `minRefine` resolve the cavity-region integral;
- larger horizontal and depth extents do not materially change the result;
- the analytic comparison depth matches the cavity/test-mass depth;
- surface vertical displacement is nonzero throughout the scaled band; and
- all expected realization files exist before post-processing.

## Outputs to retain

For reproducibility, retain:

- the exact main and cavity configurations;
- every `surfDeepDispRea*.npz` file;
- every `NNFullRea*.npz` file;
- the site ASD, attenuation, and ET-D reference data;
- the notebook or source revision; and
- the source count, realization count, domain size, grid resolution, and
  cavity-refinement settings.

Because random seeds and source metadata are intentionally not stored, these
files reproduce the resulting ensemble spectra but not the individual random
plane-wave populations.
