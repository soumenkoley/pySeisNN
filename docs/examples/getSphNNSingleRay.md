# Single Rayleigh-wave Newtonian noise for a spherical cavity

This chapter documents `getSphNNSingleRay.ipynb`. The notebook calculates the
Newtonian acceleration produced by one plane Rayleigh wave propagating along
the positive x direction in a homogeneous elastic half-space containing a
spherical cavity. It then normalizes the numerical response to unit vertical
surface displacement and compares the bulk contribution with an analytic
Rayleigh-wave NN model.

This is the Rayleigh-wave counterpart of `getSphNNSingleBody.ipynb`. It is a
controlled validation example, not a stochastic ambient-noise prediction.

## What the example demonstrates

The notebook shows how to:

- configure analytic Rayleigh-wave mode in a homogeneous half-space;
- calculate the fundamental-mode Rayleigh phase velocity with `pysurf96`;
- prescribe one unit-amplitude, zero-phase wave propagating along +x;
- evaluate its frequency-dependent displacement and depth decay;
- integrate bulk and truncation-surface NN contributions;
- normalize NN to unit vertical displacement at the free surface; and
- compare the numerical dipole term with an analytic Rayleigh NN expression.

## Physical scope

Rayleigh waves require a free surface and decay with depth into the medium.
Accordingly, this notebook is a half-space calculation:

```python
fullSpace = False
```

The comment in the notebook describes `fullSpace` as applicable only to body
waves. For this Rayleigh example, it should remain false so that the numerical
surface treatment matches the analytic half-space wavefield.

The model must be homogeneous. A layered Rayleigh wavefield requires the Qseis
full-field workflow rather than `if_ray = True`.

## Required files

The notebook expects:

- `configParseHalfSpaceRayHomo.ini`, or an equivalent homogeneous half-space
  configuration;
- `SingleSphCavityHalfSpace.ini`, or an equivalent spherical-cavity file;
- the modules under `modules/`; and
- writable directories specified in the main configuration.

Change the absolute paths near the start of both notebook cells when running
on another system:

```python
configFile = "/path/to/configParseHalfSpaceRayHomo.ini"
allCavities = configLoader.loadCavities(
    "/path/to/SingleSphCavityHalfSpace.ini"
)
```

The full parameter reference is in `docs/configuration_complete.md`.

## Tutorial model and domain

The supplied tutorial configuration uses the homogeneous model

```ini
[model]
thickness = 10000.0, 0.0
vp = 4000.0, 4000.0
vs = 2000.0, 2000.0
rho = 2465.0, 2465.0
qp = 900, 900
qs = 450, 450
rhoAir = 1.2
```

Identical layer and half-space properties make the medium homogeneous. The
zero final thickness marks the bottom half-space.

The horizontal domain extends from -3000 to +3000 m in both x and y. The
vertical domain begins at the free surface and extends to 3000 m depth.

The frequency band is 2.0–7.9 Hz with a requested 0.1 Hz interval. The
integration grid uses six points per minimum relevant wavelength.

## Cavity and test mass

The cavity file defines:

```ini
[CavityA]
shape = sphere
xC = 0.0
yC = 0.0
zC = 250.0
radius = 20.0
xRefineFactor = 6.0
yRefineFactor = 6.0
zRefineFactor = 6.0
```

The sphere is centred 250 m below the free surface and has a 20 m radius. The
selected test mass `itm_ac` is also at `(0, 0, 250 m depth)`, so it lies at the
cavity centre.

As in the preceding examples, the notebook converts configured positive depth
to the internal Cartesian convention where the half-space occupies `z <= 0`:

```python
itmACUse[2] = -np.abs(itmACUse[2])
itmAll = np.vstack((itmACUse,))
```

The other three configured ET test masses are read but are not included in
this validation calculation.

## Rayleigh-wave flags

The configuration must select analytic Rayleigh mode:

```ini
[simFlags]
if_body = False
if_ray = True
if_full_field = False
save_hv = False
decoupled_hv = True
on_fly_nn = True
```

The tutorial uses one source and one realization:

```ini
[simSource]
n_rea = 1
n_src = 1
```

This is appropriate because the notebook overwrites the random source
parameters and constructs one deterministic wave.

## Qseis geometry metadata

Although Qseis does not generate the Rayleigh displacement field, the notebook
still creates:

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

These objects provide geometric metadata used by the shared NN integration
workflow. `onlySplitAll=True` avoids assembling Qseis waveform data. The actual
Qseis execution remains commented out.

## Rayleigh phase velocity

For every selected frequency, the notebook computes the fundamental-mode
Rayleigh phase velocity:

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

`vR` controls the horizontal wavenumber and the exponential P- and S-potential
decay factors of the analytic Rayleigh eigenfunction.

## Constructing one plane Rayleigh wave

The common source generator initially creates source metadata, but the
notebook overwrites the first source:

```python
srcMeta["phiV"][0, :] = 0.0
srcMeta["ampV"][0, :] = 1.0
azSrc = np.array((0,))
```

Thus the wave has:

- azimuth `theta = 0`, meaning propagation along +x;
- unit amplitude at every selected frequency; and
- zero initial phase at every selected frequency.

The displacement is evaluated by `simDisp.computeRayleighDisp`. Its coordinate
convention is:

- free surface at `z = 0`;
- half-space at `z <= 0`; and
- positive x and y in the horizontal plane.

The returned displacement components are ordered `(x, y, z)`. For propagation
along +x, the Rayleigh motion lies in the x-z plane, so the ideal y component
is zero.

The horizontal and vertical displacements contain different combinations of
the evanescent P and SV potentials. Their amplitudes and relative phase vary
with depth and frequency.

## Reference displacement at surface and depth

The notebook saves displacement at:

```python
zList = [0.0, cavDepthUse]
```

using:

```python
simDisp.getSurfDeepRayDispPerRea(
    zList,
    azSrc,
    phiSrc,
    ampSrc,
    outDispRea,
    reaNo,
    vR,
    vp2[0],
    vs2[0],
    freqOut=freqOut,
)
```

The result is stored as `surfDeepDispRea0.npz`. It contains displacement
magnitudes at the free surface and at 250 m depth. The second notebook cell
uses its vertical surface component to normalize the NN response.

## Adaptive numerical geometry

The background wavelength-based grid and cavity-wall refinement are built in
the same way as in the body-wave examples. The defaults are:

```python
etaUse = 0.5
minRefine = 20.0
```

The model is partitioned into layers and blocks around the spherical cavity.
The cavity volume is excluded, and the surrounding material is sampled more
finely near the cavity wall.

## Numerical NN contributions

For the single realization, the notebook calculates arrays with shape
`(n_frequency, 3, 1)`:

| Array | Meaning |
| --- | --- |
| `IBlockTot` | Bulk gravitational contribution of the displaced medium. |
| `IVertFaceTot` | Contribution associated with vertical outer truncation surfaces. |
| `IHorFaceTot` | Horizontal-interface and free-/bottom-surface contribution. |
| `ITot` | Total response, `IBlockTot - IVertFaceTot - IHorFaceTot`. |

The horizontal term is especially important in a half-space because it
includes treatment of the free surface shared with air. `rhoAir` provides the
outside density for that boundary contribution.

After multiplication by Newton's gravitational constant, the result is saved
as:

```text
NNFullRea0.npz
```

under `out_disp_rea`.

## Normalization to unit surface displacement

The comparison cell loads the saved surface/depth displacement and constructs
the vertical surface amplitude:

```python
asdZ += np.abs(dispOut[:, 2, 0])**2
asdZ = np.sqrt(asdZ)
```

Because this deterministic example contains one realization, this is simply
the vertical surface-displacement magnitude. The bulk and total NN responses
are divided by it:

```python
Iblock = Iblock / asdZ[:, None, None]
Itot = Itot / asdZ[:, None, None]
```

The plotted quantities are therefore Newtonian acceleration per unit vertical
surface displacement. The y component should be zero apart from numerical
error because the wave propagates in the x-z plane.

## Analytic Rayleigh NN comparison

The analytic benchmark is calculated with:

```python
NNRay, freqNNRay = NNCompute.getAnalyticRayNN(
    config,
    250,
    gamma=0.815,
)
```

The depth argument must match the cavity/test-mass depth. Prefer passing
`cavDepthUse` instead of hard-coding 250 m when adapting the example.

The analytic expression includes:

- the Rayleigh phase velocity;
- evanescent P- and SV-potential decay with depth;
- the medium density;
- Newton's gravitational constant; and
- the dimensionless Rayleigh coupling factor `gamma`.

The notebook plots the analytic curve only against the numerical x-component
bulk term. It applies an additional factor of `sqrt(2)` for the single-wave
normalization:

```python
ax1[0].plot(
    freqNNRay,
    NNRay * np.sqrt(2),
    label="analytical dipole term",
)
```

The factor distinguishes the deterministic single-plane-wave normalization
used here from the stochastic/isotropic convention represented by the analytic
reference function.

## Interpreting the plots

The second cell creates three panels for x, y, and z acceleration components.

- The x panel compares the numerical bulk term with the analytic Rayleigh
  prediction.
- The y panel should remain close to zero for a wave propagating along +x.
- The z panel shows the vertical bulk acceleration generated by the same
  elliptically polarized Rayleigh motion.

The current comparison plots `IBlockTot`, not `ITot`. It therefore validates
the dipole/bulk integration rather than the complete cavity-plus-boundary
Newtonian acceleration. `ITot` is normalized and loaded, but is not plotted in
the present notebook cell.

## Running the example

1. Select the homogeneous half-space Rayleigh configuration.
2. Select the matching spherical-cavity file.
3. Update all machine-specific paths.
4. Verify `if_ray = True`, with `if_body` and `if_full_field` false.
5. Keep `n_src = 1`, `n_rea = 1`, and `fullSpace = False`.
6. Run the first cell to create geometry metadata, calculate displacement, and
   integrate NN.
7. Confirm that `surfDeepDispRea0.npz` and `NNFullRea0.npz` exist.
8. Run the second cell using the same configuration path.
9. Check the x-component numerical/analytic agreement and the near-zero y
   component.
10. Repeat with finer grids or a larger domain to assess convergence.

## Consistency and convergence checks

Before interpreting the result, verify that:

- the medium is homogeneous;
- the cavity and selected test mass are both at the analytic comparison depth;
- the free surface is at `z = 0` and internal depth coordinates are negative;
- the wave azimuth is zero and only one source is present;
- the analytic and numerical frequency vectors cover the same bins;
- `gamma` uses the intended convention for the homogeneous model;
- `lambda_res`, `etaUse`, and `minRefine` provide a converged cavity-region
  integration;
- the horizontal and bottom domain boundaries are sufficiently distant; and
- the comparison consistently distinguishes the bulk term from the total NN.

## Scope and limitations

This notebook uses one coherent Rayleigh plane wave in a homogeneous
half-space. It does not represent an isotropic ambient surface-wave field or a
layered Rayleigh eigenfunction. The next example,
`getSphNNStochRay.ipynb`, introduces many random propagation azimuths, random
amplitudes and phases, ensemble RMS assembly, and site-based scaling.
