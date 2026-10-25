# Stochastic body-wave Newtonian noise for a cuboidal cavity

This chapter documents `getCubNNStochBody.ipynb`. It combines the adaptive
cuboid geometry introduced in `getCubNNSingleBody.ipynb` with a stochastic
ensemble of randomly oriented P and S plane waves. The notebook calculates NN
for each realization, assembles ensemble RMS displacement and acceleration,
scales the simulation to site displacement spectra, and projects the resulting
acceleration onto Einstein Telescope strain.

Read the single-cuboid chapter for the detailed cavity geometry and numerical
integration scheme, and the stochastic spherical body-wave chapter for the
general ensemble and post-processing concepts.

## What the example demonstrates

The notebook shows how to:

- generate a stochastic P/S body-wave field in a homogeneous medium;
- calculate NN around an elongated cuboidal cavity;
- apply face-dependent numerical refinement near the cavity wall;
- repeat and save the calculation for independent realizations;
- form RMS displacement and NN spectra;
- scale the results to an underground east-component ASD;
- compare with generic analytic homogeneous body-wave curves; and
- estimate ET strain ASD from one simulated test-mass spectrum.

The active notebook is configured as a full-space calculation. A corresponding
half-space variant can be used by changing `fullSpace` and selecting matching
configuration and cavity files.

## Required files

The notebook expects:

- `configParseFullSpaceBodyHomo.ini` under the stochastic cuboid tutorial;
- `SingleCubCavityFullSpace.ini`;
- site ASD and attenuation files;
- the ET-D sensitivity file; and
- writable Qseis-metadata, temporary-displacement, and realization-output
  directories.

The absolute configuration paths near the top of the notebook must be adapted
for another installation.

## Tutorial geometry

The supplied full-space tutorial describes:

- a homogeneous medium with `vp = 4000 m/s`, `vs = 2000 m/s`, and
  `rho = 2465 kg/m^3`;
- a domain with horizontal half-width 3000 m and total configured depth
  6000 m;
- a `400 × 20 × 20 m` cuboidal cavity centred at 3000 m depth;
- `angleDeg = 0`, so the 400 m length is aligned with global x; and
- one selected test mass at the cavity centre.

The notebook selects only `itm_ac`:

```python
itmAll = np.vstack((itmACUse,))
```

Although the remaining three ET positions are loaded, they are not included
in the present calculation.

The active space setting is:

```python
fullSpace = True
```

This value must agree with the main configuration, cavity depth, integration
bounds, and any later interpretation of surface/depth reference motion.

## Cuboid refinement

The background integration spacing is calculated from the minimum relevant
wavelength. The notebook then computes face-specific cavity-wall refinement:

```python
etaUse = 0.5
wallRefineWidthAll = NNGeometryNN.getRefineWidths(
    tmInCavity,
    allCavities,
    hBGAll,
    eta=etaUse,
    minRefine=20.0,
)
```

The current cuboid geometry expands each local cavity face by its calculated
refinement width, rotates the expanded horizontal footprint by `angleDeg`, and
constructs an enclosing global axis-aligned refinement box. The printed
`wallRefineWidthAll` values, rather than only the legacy refinement-factor
fields in the INI file, describe the effective local refinement used by this
run.

Nearby material points are sampled more finely, while points inside the cavity
are excluded from the volume integral.

## Wavefield flags

The configuration selects analytic body-wave mode:

```ini
[simFlags]
if_body = True
if_ray = False
if_full_field = False
save_hv = False
decoupled_hv = True
on_fly_nn = True
```

The body-wave model is valid only for a homogeneous medium. Qseis waveforms do
not generate the displacement field, although `splitAll` and `splitAllNew`
remain necessary for common geometry metadata.

## Source and realization counts

The configuration currently referenced by the notebook contains:

```ini
[simSource]
n_rea = 100
n_src = 100
```

Thus the configured ensemble contains 100 independent realizations, each with
100 P waves and 100 S waves. This is a reasonable production starting point,
but convergence must still be checked explicitly; some studies may require
more realizations or sources.

When `n_rea` is increased, use a dedicated output directory or remove stale
numbered realization files so post-processing cannot mix incompatible runs.

## Stochastic P/S population

Every realization calls:

```python
metaBody = simDisp.genBodySrc(
    nSrc,
    nFreq,
    eP=1/3,
    eS=2/3,
)
```

The routine draws isotropic propagation directions, frequency-dependent random
amplitudes, random phases, and random S-wave polarization angles. Amplitudes
are weighted by `sqrt(eP)` and `sqrt(eS)`, so the notebook requests a
one-third P and two-thirds S energy weighting. This matches the stochastic
spherical body-wave tutorial and the analytic benchmark used later in this
notebook.

The common ambient-source call also produces `(xSrc, ySrc, azSrc, srcMeta)`.
In analytic body-wave mode, `metaBody` determines the plane-wave propagation
and polarization; the generated source positions are not the physical origins
of these infinite plane waves.

Random seeds and `metaBody` are not saved. Repeated executions therefore
produce statistically equivalent but non-identical ensembles.

## Geometric metadata

The notebook calls `createDB.createFinalInputs` and
`gfLoader.assembleQseisRuns(..., onlySplitAll=True)` to construct `splitAll`
and `splitAllNew`. These contain required geometric metadata. The Qseis
execution line remains commented out because analytic body waves are used.

## Per-realization displacement

For every body-wave population, the notebook evaluates displacement at:

```python
zList = [0.0, cavDepthUse]
```

and writes:

```text
surfDeepDispRea<realization>.npz
```

In the supplied full-space configuration, `cavDepthUse = 3000 m`. A half-space
variant may instead use 250 m. Plot labels should be constructed from
`cavDepthUse`; the current notebook label `250 m deep` matches the half-space
variant but not the active full-space configuration.

## Per-realization NN integration

For every realization, the model is partitioned into refined layers and
blocks. The notebook accumulates:

| Array | Meaning |
| --- | --- |
| `IBlockTot` | Material-volume contribution outside the cuboid. |
| `IVertFaceTot` | Vertical outer-boundary contribution. |
| `IHorFaceTot` | Horizontal-interface and horizontal-boundary contribution. |
| `ITot` | Total acceleration, `IBlockTot - IVertFaceTot - IHorFaceTot`. |

The arrays have shape `(n_frequency, 3, n_test_mass)`. After multiplication by
Newton's gravitational constant, each realization is saved as:

```text
NNFullRea<realization>.npz
```

## Ensemble assembly

After all realizations have completed, the notebook calls:

```python
dispPointAllRea, attnAllRea, freqDisp = (
    simDisp.assembleSurfDeepDispAllRea(
        outDispRea, zList, freqOut, nRea
    )
)
```

and:

```python
nnAllRea, nnVolAllRea, nnSurfAllRea, fNN = (
    NNCompute.assembleNNAllReaAllTM(
        outDispRea, freqOut, nRea, nTM
    )
)
```

The routines calculate ensemble RMS magnitudes rather than coherent complex
averages. With the current `n_rea = 100`, they combine the complete configured
ensemble.

## Displacement scaling and attenuation

The notebook scales displacement using the underground east/x component:

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

The simulated east component at cavity depth is matched to the measured
underground east ASD at every frequency. That same frequency-dependent factor
is applied to all components and both reference depths.

The scaled surface and underground displacement are plotted and compared with
the site ASD files. `NNPP.getAttn` separately compares simulated and observed
surface-to-depth attenuation.

For the active full-space case, a physical surface at `z = 0` is not part of
the intended infinite-medium model in the same sense as a half-space free
surface. The surface/depth terminology is inherited from the shared
post-processing workflow and should be interpreted as two reference planes in
the full-space calculation.

## Unit-reference NN response

The notebook normalizes total NN to unit underground east displacement:

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

This removes the arbitrary overall amplitude of the random plane-wave
population while preserving the simulated ratios between acceleration
components.

## Analytic comparison curves

The notebook plots three reference families with the unit-normalized numerical
NN:

- analytic Rayleigh NN;
- analytic stochastic body-wave NN; and
- the reference `4*pi*G*rho/3` level.

The body-wave curve is the relevant general benchmark. The notebook calls:

```python
NNBody, minNN, freqNNBody = NNCompute.getAnalyticBodyNN(
    config,
    p=1/3,
)
```

which matches the simulated `eP = 1/3` P-wave energy fraction. The simulation
and analytic reference therefore use the same P/S mixture.

The analytic curves describe generic homogeneous-medium coupling and do not
include the detailed finite cuboid response in the same way as the
single-wave finite-box functions. The stochastic numerical result includes the
actual cuboid exclusion and test-mass geometry.

## Site-scaled NN and ET projection

The notebook obtains physical acceleration ASD by scaling total NN to the
measured underground east ASD:

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

It then calls `NNPP.NNProjectToET`. The default projection uses a 10 km arm and
a factor of two in ASD for four independent test-mass contributions. This is
an approximate projection from one simulated test mass, not a coherent
multi-test-mass calculation with spatial correlations.

## Running the notebook

1. Select matching full-space or half-space main and cuboid configurations.
2. Update all machine-specific and reference-data paths.
3. Verify that the medium is homogeneous and `if_body = True`.
4. Confirm the cavity dimensions, rotation, test-mass location, and
   `fullSpace` setting.
5. Choose `n_src`, `n_rea`, `eP`, and `eS` for the intended ensemble.
6. Run the notebook and monitor every realization and integration block.
7. Verify that every realization created both displacement and NN files.
8. Inspect ensemble displacement and attenuation before scaling NN.
9. Compare unit-reference NN with a body-wave curve using the same P fraction.
10. Inspect site-scaled acceleration and ET-D projection.
11. Repeat with increased source/realization counts and refined spatial grids
    to establish convergence.

## Current configuration details to review

The active tutorial configuration currently has:

- `n_rea = 100` and `n_src = 100`;
- a consistent one-third P and two-thirds S energy mixture in the simulation
  and analytic benchmark;
- correct dedicated `StochWave/FullSpace` output directories;
- reference-data paths pointing to this repository's `TERZ/` and `ETDesign/`
  directories; and
- a 3000 m cavity depth, while one plot label says 250 m.

These points do not alter the integration formulas, but they should be updated
or explicitly accepted before a production run.

## Convergence checks

A production result should be tested against:

- increasing `n_rea` until RMS spectra stabilize;
- increasing `n_src` until the stochastic direction and polarization
  statistics stabilize;
- changing the P/S energy fractions intentionally and matching the analytic
  benchmark parameter;
- increasing `lambda_res`;
- varying `etaUse`, `minRefine`, and the resulting face-specific refinement
  widths;
- enlarging the finite integration domain; and
- checking the effect of cuboid rotation and test-mass position.

## Outputs to retain

Retain the exact main and cavity configurations, realization displacement and
NN files, site reference data, ET-D curve, notebook/source revision, P/S energy
fractions, source and realization counts, and all grid/refinement parameters.
