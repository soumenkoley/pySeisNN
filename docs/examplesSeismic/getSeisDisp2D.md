# Visualizing a stochastic displacement field on a 2D plane

This chapter documents `getSeisDisp2D.ipynb`. The notebook simulates the
frequency-domain seismic displacement on a horizontal two-dimensional receiver
grid and displays the x, y, and z component amplitudes as colour maps.

This is a displacement-field example, not a Newtonian-noise calculation. Its
main purpose is to inspect spatial structure produced by a stochastic set of
seismic sources at one selected depth and frequency.

## What the example shows

The notebook demonstrates how to:

- select an internal or ring distribution of stochastic force sources;
- generate a Qseis database at one target depth;
- assemble the required frequency-domain GF components;
- construct and flatten a regular two-dimensional receiver grid;
- simulate displacement over the complete grid;
- combine one or more realizations as an RMS magnitude; and
- visualize `|Ux|`, `|Uy|`, and `|Uz|` at a selected frequency.

The driver also contains analytic body- and Rayleigh-wave branches. The two
supplied configurations select Qseis full-field mode in a homogeneous
half-space.

## Supplied cases

The notebook contains two alternative configuration paths:

| Configuration | Source distribution | GF coverage |
| --- | --- | ---: |
| `configParseDispRingHomo.ini` | Ring from 6000 to 6500 m radius | 13000 m |
| `configParseDispInternalHomo.ini` | Uniform x and y within the domain | 6000 m |

The ring case is active by default:

```python
configFile = "/home/koley/CavityResults/DispTutorialConfig/configParseDispRingHomo.ini"
```

Activate only one configuration path at a time.

Both supplied cases use the homogeneous model:

```ini
thickness = 10000.0, 0.0
vp = 4000.0, 4000.0
vs = 2000.0, 2000.0
rho = 2465.0, 2465.0
```

They select:

```ini
if_body = False
if_ray = False
if_full_field = True
on_fly_nn = False
```

`on_fly_nn` must remain false because the complete receiver-grid displacement
is needed for visualization. No NN reduction is requested.

## Target depth

The visualization plane is selected with:

```python
zTar = 0.0
```

Depth is entered as a non-negative physical depth for the Qseis full-field
path. The default `0.0` therefore selects the free surface. The target-depth
list passed to database preparation contains only this plane:

```python
zList = np.array((zTar,))
```

`createFinalInputsTD` consequently generates only the receiver-depth sampling
needed for the selected visualization plane.

## Two-dimensional receiver grid

The grid is defined by:

```python
gridX = np.arange(-2000, 2000, 40)
gridY = np.arange(-2000, 2000, 40)
```

Because the upper bound of `np.arange` is excluded, both coordinate arrays
contain 100 values from -2000 m through 1960 m. The plane therefore contains:

```text
nX = 100
nY = 100
nGrid = 10000 receiver points
grid spacing = 40 m
```

The notebook constructs the grid with:

```python
gridXMat, gridYMat = np.meshgrid(gridX, gridY, indexing="ij")
gridX_flat = gridXMat.ravel()
gridY_flat = gridYMat.ravel()
```

The flattened arrays are passed to the displacement functions. The plotting
routine later reshapes each component to `(nX, nY)` using C order, consistent
with this construction.

To include `+2000 m` explicitly, the user would need an upper bound greater
than 2000, for example `np.arange(-2000, 2000 + 40, 40)`.

## Visualization frequency

The requested display frequency is:

```python
fSee = 4.0
```

`plotDispDepthSlice` selects the first available frequency satisfying:

```python
freqOut >= fSee
```

Thus the displayed frequency is the first selected FFT bin at or above 4 Hz,
not necessarily exactly 4 Hz for an arbitrary `t_max`, `n_samp`, and `df`.
The actual value is printed in the figure title.

## Qseis database generation

For the supplied full-field configurations, the notebook runs:

```python
splitAll = createDB.createFinalInputsTD(config, zList)
results = createDB.runMultiQseis(
    splitAll,
    config.inputPath,
    config.qseisExe,
    nWorkers=config.cpuCoresQseis,
)
```

The generated time-domain outputs are then assembled with
`onlySplitAll=False`. The component list is:

```text
fh-2.tz  fh-2.tr  fh-2.tt  fz-2.tz  fz-2.tr
```

These provide radial, transverse, and vertical receiver motion for horizontal
and vertical surface forces. The assembly uses `minVel = 100 m/s` for its
arrival window.

The notebook regenerates and reruns the database whenever `main()` is executed
in full-field mode. Inspect the returned Qseis statuses before trusting the
assembled arrays.

## Stochastic sources

Both configurations use:

```ini
n_rea = 1
n_src = 100
scale_vh = 1.0
decoupled_hv = True
```

One realization therefore superposes 100 surface-force sources. Horizontal
and vertical amplitudes and phases are independently randomized at each
frequency, with no systematic vertical/horizontal amplitude bias.

In the internal case, x and y source coordinates are drawn uniformly within
the configured domain. In the ring case, radii lie between 6000 and 6500 m and
source azimuths are random.

Random seeds and source metadata are not saved, so rerunning the notebook
produces a different spatial pattern.

## Displacement simulation

For Qseis full-field mode, each realization calls:

```python
dispTempSurf, _, _, shmPathsSurf = simDisp.computeFullDispF(
    zTar,
    gridX_flat,
    gridY_flat,
    itmAll,
    xSrc,
    ySrc,
    azSrc,
    srcMeta,
    ...,
    saveHV=False,
    reduceNN=onFlyNN,
    nnMode=None,
    nnMeta=None,
)
```

For the configured `on_fly_nn = False`, `dispTempSurf` has shape:

```text
(frequency, receiver point, component)
```

with component order `[x, y, z]`. It is complex because it retains the
frequency-domain phase at every receiver.

After its power has been accumulated, the temporary field is deleted and the
shared-memory files returned in `shmPathsSurf` are cleaned.

## Combining realizations

The accumulator is initialized as:

```python
dispTotalAllReaSurf = np.zeros((nFreq, nGrid, 3))
```

For each realization, the notebook adds the squared magnitude and finally
computes:

```text
U_RMS(f, r, i) = sqrt(mean_realizations(|U(f, r, i)|^2))
```

With the supplied `n_rea = 1`, this reduces to `|U|` for a single stochastic
realization. The map therefore shows one spatial interference pattern.

If `n_rea` is increased, the map becomes an ensemble RMS amplitude and loses
the complex phase and signed spatial pattern of any individual realization.

## Meaning and units of the plotted quantity

The plotting call is:

```python
plotUtils.plotDispDepthSlice(
    dispTotalAllReaSurf,
    freqOut,
    gridXMat,
    gridYMat,
    nX,
    nY,
    fSee,
)
```

It produces three `pcolormesh` panels labelled:

```text
|Ux|   |Uy|   |Uz|
```

and uses metres on the colour bars. The displayed value is a
frequency-domain displacement amplitude from the synthetic source model. It
should not automatically be called an amplitude spectral density: no ASD
normalization or scaling to measured seismic data is performed in this
notebook.

Absolute amplitudes depend on the stochastic source amplitudes, number of
sources, Qseis wavelet/FFT convention, and any later normalization. The maps
are especially useful for examining relative spatial structure.

## Shared tutorial output directory

Like `getSeisCCF.ipynb` and the other displacement tutorial, both supplied
configurations use:

```text
/home/koley/CavityResults/AllTutOut/DispTests/
```

with common `QseisInpN/`, `OutDisp/`, and `OutDispRea/` subdirectories. These
outputs are intentionally disposable and are overwritten as different tutorial
cases are run. With `qseis_out_del = yes`, database preparation can delete the
previous Qseis run directories.

Copy any database, numerical field, or figure that must be retained before
running another displacement tutorial.

The notebook itself does not save `dispTotalAllReaSurf` or the figure. The
common folders contain Qseis and temporary displacement products, not a named
persistent copy of the final plotted map.

## Alternative analytic modes

The driver includes branches for analytic homogeneous body and Rayleigh waves.
They can be selected by preparing a compatible configuration with exactly one
of `if_body` or `if_ray` true and `if_full_field` false.

These analytic modes do not require Qseis output; `onlySplitAll=True` creates
only consolidated geometry metadata. They remain restricted to homogeneous
media.

### Nonzero-depth convention in analytic modes

`computeRayleighDisp` uses internal Cartesian z, where underground positions
are negative. The body-wave plane-wave phase uses the same Cartesian
coordinate. The notebook converts the user-facing positive depth before both
analytic calculations:

```python
zUse = -abs(zTar)
```

It passes `zUse` to `computeFullBodyDisp_parallel` or
`computeRayleighDisp`. The Qseis full-field path instead receives `zTar`
directly because its GF interface uses non-negative receiver depth. Thus a
single positive `zTar` value now selects the same physical plane in all three
wavefield modes.

## Plot-depth title caveat

`plotDispDepthSlice` has a separate keyword `zdepth`, whose default is zero.
The notebook does not currently pass it. Therefore, after changing `zTar` to a
nonzero depth, the data plane changes but the figure title will still report
`z = 0 m`.

Use:

```python
plotUtils.plotDispDepthSlice(
    dispTotalAllReaSurf,
    freqOut,
    gridXMat,
    gridYMat,
    nX,
    nY,
    fSee,
    zdepth=zTar,
)
```

to keep the title consistent with the simulated plane.

## Running the example

1. Select the ring or internal configuration.
2. Confirm that overwriting `AllTutOut/DispTests/` is acceptable.
3. Choose `zTar`, the x/y grid vectors, and `fSee`.
4. Keep `on_fly_nn = False` for displacement-field output.
5. Confirm that `xmax_gf` covers every source-to-grid separation.
6. Run database generation and check all Qseis process statuses.
7. Inspect the actual selected frequency printed in the figure title.
8. Check all three component maps for boundary or interpolation artifacts.
9. Save the array and figure explicitly if they must be retained.
10. When using nonzero depth, pass `zdepth=zTar` to the plotter.

## Convergence and interpretation checks

Before interpreting spatial structure, vary:

- receiver-grid spacing and extent;
- number of stochastic sources;
- number of realizations;
- source distribution and ring radii;
- GF receiver-distance sampling;
- Qseis time window and arrival-window velocity; and
- random seed or repeated random draws.

Also verify that structures near the edges are not caused by insufficient GF
coverage or the finite plotted domain.

## Current limitations

- The supplied examples cover a homogeneous Qseis model only, although the
  full-field machinery can use a layered configuration.
- The default is one realization, so it is a stochastic snapshot in amplitude
  rather than a converged ensemble statistic.
- The final displacement array and figure are not saved automatically.
- Source metadata and random seeds are not retained.
- The plot title remains at zero depth unless `zdepth=zTar` is passed.
- The output tree is shared and overwritten by the seismic-field tutorials.
