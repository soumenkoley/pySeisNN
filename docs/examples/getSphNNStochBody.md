# Stochastic body-wave Newtonian noise for a spherical cavity

This chapter documents `getSphNNStochBody.ipynb`. It extends the deterministic
single-wave validation in `getSphNNSingleBody.ipynb` to an ensemble of randomly
oriented P and S waves in a homogeneous medium. The notebook calculates the
Newtonian acceleration for every realization, forms ensemble RMS quantities,
scales the simulation to measured site displacement spectra, and projects the
result onto Einstein Telescope strain.

Read the single-wave chapter first for the shared geometry, adaptive
refinement, volume integration, truncation-surface treatment, and definitions
of the numerical NN contributions. This chapter emphasizes the stochastic and
post-processing steps that differ.

## Purpose of the example

The notebook demonstrates how to:

- generate stochastic, isotropically propagating P and S plane waves;
- combine many waves in one seismic realization;
- repeat the NN calculation over independent realizations;
- save displacement and NN results per realization;
- form ensemble RMS displacement and acceleration spectra;
- compare simulated surface-to-depth attenuation with site observations;
- normalize NN to unit reference displacement;
- scale NN to a measured displacement ASD; and
- convert acceleration ASD to Einstein Telescope strain ASD.

This remains an analytic body-wave model. Qseis waveforms do not generate the
displacement field, although `splitAll` and `splitAllNew` are still prepared
because their geometric metadata is used by the common NN workflow.

## Required inputs

The notebook expects:

- a homogeneous-medium main configuration, such as
  `configParseFullSpaceBodyHomo.ini`;
- a one-sphere cavity file, such as `SingleSphCavityFullSpace.ini`;
- site reference files `asdSurf.mat`, `asdBH.mat`, and `attnModel.mat`;
- the ET-D design curve `ETD.mat`; and
- writable Qseis-metadata, temporary-displacement, and realization-output
  directories.

Change the absolute paths near the beginning of the notebook:

```python
configFile = "/path/to/configParseFullSpaceBodyHomo.ini"
allCavities = configLoader.loadCavities(
    "/path/to/SingleSphCavityFullSpace.ini"
)
```

The complete parameter reference is in `docs/configuration_complete.md`.

## Shared physical geometry

The supplied tutorial files describe:

- a homogeneous medium with `vp = 4000 m/s`, `vs = 2000 m/s`, and
  `rho = 2465 kg/m^3`;
- a `6000 × 6000 × 6000 m` numerical domain;
- a spherical cavity of radius 20 m centred at `(0, 0, 3000 m depth)`; and
- one selected test mass, `itm_ac`, located at the cavity centre.

The notebook reads all four configured test masses but explicitly reduces the
calculation to one:

```python
itmAll = np.vstack((itmACUse,))
```

To calculate all four nearby ET test masses, replace this with:

```python
itmAll = np.vstack((itmACUse, itmABUse, etmCAUse, etmBAUse))
```

and ensure that each position belongs to an appropriate cavity or modeled
region.

The `fullSpace` variable controls the body-wave boundary treatment:

```python
fullSpace = True
```

Use matching main and cavity configurations when changing between full-space
and half-space calculations.

## Required wavefield flags

The main configuration must select body-wave mode:

```ini
[simFlags]
if_body = True
if_ray = False
if_full_field = False
save_hv = False
decoupled_hv = True
on_fly_nn = True
```

The model must be homogeneous. Setting `if_body = True` for a genuinely
layered medium does not produce a valid layered-wavefield calculation.

## Number of sources and realizations

The stochastic calculation is controlled primarily by:

```ini
[simSource]
n_rea = 100
n_src = 100
```

`n_src` is the number of simultaneously acting P waves and S waves represented
in one realization. `n_rea` is the number of independent realizations included
in the ensemble RMS.

The stochastic tutorial uses `n_rea = 100` and `n_src = 100`. This provides a
reasonable starting ensemble, but the required realization count remains a
convergence question. Production studies may require 100–500 realizations or
more depending on the stability required from the RMS spectra.

The source-position settings `r_1`, `r_2`, and `src_distri` are passed to
`simDisp.genAmbSrc` because the notebook shares its driver structure with
Qseis and Rayleigh-wave workflows. In analytic body-wave mode, the actual
plane-wave directions and polarizations used for displacement generation are
stored in `metaBody` and are not determined by the generated `(xSrc, ySrc)`
locations.

## Stochastic P/S-wave ensemble

Each realization calls:

```python
metaBody = simDisp.genBodySrc(
    nSrc,
    nFreq,
    eP=1/3,
    eS=2/3,
)
```

For both P and S populations, the routine draws:

- propagation azimuth uniformly from `0` to `2*pi`;
- direction cosine uniformly from `-1` to `1`, producing isotropic directions;
- amplitude uniformly between zero and one at every frequency;
- phase uniformly between zero and `2*pi` at every frequency; and
- for S waves, a random polarization angle in the plane perpendicular to the
  propagation direction.

The amplitude factors are multiplied by `sqrt(eP)` and `sqrt(eS)`. With
`eP = 1/3` and `eS = 2/3`, the requested mean energy weighting is one-third P
and two-thirds S.

The random metadata are regenerated inside the realization loop, so different
realizations are statistically independent. The code does not currently set or
save a random-number seed; an exact stochastic run is therefore not
reproducible solely from its configuration file.

## Geometric metadata and adaptive grid

As in the single-wave example, the notebook calls

```python
splitAll = createDB.createFinalInputs(config)
```

and

```python
splitAllNew = gfLoader.assembleQseisRuns(
    splitAll,
    fInpPath,
    components,
    fIndices=idxFreq,
    onlySplitAll=True,
    minVel=100.0,
)
```

`onlySplitAll=True` prepares the required geometry metadata without assembling
a Qseis waveform database. The Qseis execution line remains commented out.

The background grid, test-mass-to-cavity association, wall-refinement widths,
refinement boxes, and layer/block decomposition are constructed exactly as in
the single-wave example. The default refinement controls are:

```python
etaUse = 0.5
minRefine = 20.0
```

## Per-realization displacement reference

For every random body-wave population, the notebook evaluates displacement at
two depths:

```python
zList = [0.0, cavDepthUse]
simDisp.getSurfDeepBodyDispPerRea(
    zList,
    metaBody,
    outDispRea,
    reaNo,
    vp2[0],
    vs2[0],
    freqOut=freqOut,
)
```

The resulting file is named

```text
surfDeepDispRea<realization>.npz
```

and contains the displacement magnitudes and frequency vector used later for
RMS assembly, attenuation, and scaling.

The two locations are the surface and the cavity depth. The full-space example
uses a cavity depth of 3000 m, whereas the corresponding half-space example
uses 250 m. Plot labels should therefore be generated from `cavDepthUse` or
updated manually whenever the selected configuration changes. A hard-coded
`250 m deep` label is correct for the half-space case but not for the supplied
full-space case.

## Per-realization NN integration

The numerical integration is the same as in the deterministic example. For
each realization, the notebook accumulates arrays with shape
`(n_frequency, 3, n_test_mass)`:

| Array | Meaning |
| --- | --- |
| `IBlockTot` | Material-volume contribution. |
| `IVertFaceTot` | Vertical outer-boundary contribution. |
| `IHorFaceTot` | Horizontal-interface and horizontal-boundary contribution. |
| `ITot` | Total acceleration, `IBlockTot - IVertFaceTot - IHorFaceTot`. |

After multiplication by Newton's gravitational constant, the result is saved
as:

```text
NNFullRea<realization>.npz
```

Each file contains the four arrays and `freqOut`. These realization files are
the persistent inputs to ensemble post-processing.

## Ensemble RMS assembly

After the realization loop, the notebook assembles the displacement ensemble:

```python
dispPointAllRea, attnAllRea, freqDisp = (
    simDisp.assembleSurfDeepDispAllRea(
        outDispRea, zList, freqOut, nRea
    )
)
```

and the NN ensemble:

```python
nnAllRea, nnVolAllRea, nnSurfAllRea, fNN = (
    NNCompute.assembleNNAllReaAllTM(
        outDispRea, freqOut, nRea, nTM
    )
)
```

These routines combine realization magnitudes in quadrature and take the
square root of their mean. The resulting quantities are ensemble RMS spectra,
not a coherent complex average.

The principal output shapes are:

- `dispPointAllRea`: `(n_frequency, 3, 2)`, where the last axis is surface and
  cavity depth;
- `nnAllRea`: `(n_frequency, 3, n_test_mass)`;
- `nnVolAllRea`: volume contribution with the same NN shape; and
- `nnSurfAllRea`: combined surface contribution with the same NN shape.

## Scaling the simulated displacement

The notebook scales the ensemble displacement using:

```python
scaledSimDisp, *siteASDs = NNPP.scaleDisp(
    dispPointAllRea,
    freqOut,
    config,
    scaleComp="E",
    scaleDepth="depth",
    scaleToTrue=True,
)
```

This chooses the east/x component at the underground reference depth. The
frequency-dependent scale factor makes that simulated component match the
selected observed underground site ASD. The same factor is applied to the
other simulated components so that their relative component ratios are
preserved.

`plotPSDDeepSurfMulti` then shows scaled displacement at the surface and cavity
depth. `plotScaledVsSiteASD` compares the scaled simulation against the loaded
surface and borehole reference spectra.

The site-ASD files must cover the simulation frequency band. The scaling does
not make the unselected components independently match their corresponding
observations.

## Surface-to-depth attenuation

The notebook calls:

```python
attnSim, attnX, attnY, attnZ = NNPP.getAttn(
    dispPointAllRea, freqOut, config
)
```

to compare the simulated surface-to-depth displacement ratio with observed
attenuation data from `attnModel.mat`. This is a diagnostic comparison between
the homogeneous stochastic body-wave model and the selected site model; it is
not used directly in the NN integration.

## Unit-displacement NN response

To expose the simulated NN coupling independently of the absolute random
source amplitude, the notebook first normalizes NN without applying measured
site motion:

```python
scaledNN[:, :, i], *__ = NNPP.scaleNN(
    nnAllRea[:, :, i],
    dispPointAllRea,
    freqOut,
    config,
    scaleComp="E",
    scaleDepth="depth",
    scaleToTrue=False,
)
```

The reference east displacement at cavity depth is set to unity, and the same
scale factor is applied to all three acceleration components. The resulting
plot is therefore an NN acceleration response per unit selected displacement,
not a prediction using the measured site ASD.

## Analytic comparison curves

The unit-normalized numerical NN is plotted with:

- the analytic Rayleigh-wave NN curve from `getAnalyticRayNN`;
- the analytic stochastic body-wave curve from `getAnalyticBodyNN`; and
- the reference level `4*pi*G*rho/3`.

For this notebook, the stochastic body-wave curve is the directly relevant
benchmark. The Rayleigh curve is shown as a broader comparison and should not
be interpreted as the expected body-wave result.

The analytic body-wave call uses:

```python
NNBody, minNN, freqNNBody = NNCompute.getAnalyticBodyNN(
    config, p=1/3
)
```

where `p = 1/3` matches the selected P-wave energy fraction.

## Scaling to site motion and ET-D projection

The notebook next calculates a physical site-scaled acceleration ASD:

```python
scaledNNToTrue[:, :, i], *__ = NNPP.scaleNN(
    nnAllRea[:, :, i],
    dispPointAllRea,
    freqOut,
    config,
    scaleComp="E",
    scaleDepth="depth",
    scaleToTrue=True,
)
```

It then converts acceleration ASD to an equivalent interferometer strain ASD:

```python
nnProj[:, :, i], ETD = NNPP.NNProjectToET(
    scaledNNToTrue[:, :, i], fNN, config
)
```

`NNProjectToET` uses a default 10 km arm length and a factor of two in ASD for
four independent test-mass contributions added in quadrature. The plotted ET-D
curve comes from `ETD.mat`.

This projection is an approximate single-test-mass-to-interferometer scaling.
It is not the same as coherently calculating and projecting all four test-mass
accelerations with their spatial correlations.

## Running the notebook

1. Select matching full-space or half-space main and cavity configurations.
2. Replace all machine-specific paths.
3. Verify that the medium is homogeneous and body-wave mode is enabled.
4. Confirm the intended `n_src`, `n_rea`, and P/S energy weights. The supplied
   stochastic configuration runs 100 sources over 100 realizations and can
   take considerable time.
5. Run the main cell and monitor each realization and integration block.
6. Confirm that every realization produced both
   `surfDeepDispRea*.npz` and `NNFullRea*.npz`.
7. Inspect displacement scaling and attenuation before interpreting NN.
8. Compare unit-normalized NN with the analytic body-wave benchmark.
9. Inspect site-scaled acceleration and the ET-D projection.
10. Repeat with larger source and realization counts if required to verify
    convergence.

## Configuration-dependent details

Review the following details whenever switching between tutorial variants:

- The stochastic tutorial uses 100 realizations and dedicated `StochWave`
  output paths. Keep its outputs separate from the deterministic `SingleWave`
  example.
- Its `site_asd_path` and `et_d_path` point to the older
  `pySeis1DNNCavity` checkout. Update them to this repository's `TERZ/` and
  `ETDesign/` directories if those bundled files are intended.
- Use a `3000 m` underground label for the full-space configuration and a
  `250 m` label for the half-space configuration. Prefer constructing the
  label from `cavDepthUse` to prevent mismatches.

## Random realization policy

The notebook intentionally does not save a random seed or the generated
`metaBody` source metadata. Repeated executions therefore produce statistically
equivalent but not identical ensembles. This is acceptable for the intended
stochastic RMS calculation; retain source metadata only if exact realization
reproduction becomes a requirement for a particular study.

## Recommended convergence checks

A production result should be tested against:

- increasing `n_rea` until ensemble RMS spectra stabilize;
- increasing `n_src` until the stochastic field statistics stabilize;
- increasing `lambda_res` to refine the background integration grid;
- changing `etaUse` and `minRefine` to test cavity-wall convergence;
- increasing the simulation-domain extent to test truncation boundaries; and
- comparing full-space or half-space numerical NN with the appropriate
  analytic body-wave expectation.

## Outputs to retain

For reproducibility, retain:

- the exact main and cavity configurations;
- the notebook version or source revision;
- every `surfDeepDispRea*.npz` file;
- every `NNFullRea*.npz` file;
- the assembled displacement and NN products;
- the site ASD and ET-D reference files used for scaling; and
- the chosen source count, realization count, P/S energy fractions, grid
  resolution, refinement parameters, and full-/half-space setting.
