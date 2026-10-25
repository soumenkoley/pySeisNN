# Basic usage

This guide describes the common workflow used by the example notebooks. It is
intended as the starting point before reading a notebook-specific chapter.

For definitions of every input field, see
[`configuration_complete.md`](configuration_complete.md). Coordinate and file
conventions are described in [`coordinates_and_units.md`](coordinates_and_units.md)
and [`outputs.md`](outputs.md).

## Files needed for a calculation

A run normally requires:

1. a primary INI configuration file;
2. a cavity INI file;
3. one driver notebook;
4. for full-field calculations, a Qseis template and executable; and
5. writable directories for the GF database, temporary data, and realization
   outputs.

Analytic body- and Rayleigh-wave examples use the geometrical metadata created
by the database-preparation code, but they do not run or load Qseis GFs.

## Select the physical model

Define the one-dimensional geology in `[model]`. Layers are ordered downward
from the free surface, and the last thickness must be zero to identify the
bottom half-space. All layer-property arrays must have the same length.

Choose the calculation mode before running:

| Intended field | Required setting | Medium |
| --- | --- | --- |
| Analytic P and S waves | `if_body = True` | Homogeneous only |
| Analytic Rayleigh waves | `if_ray = True` | Homogeneous only |
| Qseis Green's functions | `if_full_field = True` | Homogeneous or layered |

Only one mode should be enabled. See
[`simulation_modes.md`](simulation_modes.md) for the complete flag matrix.

## Define the frequency and time grids

The calculation frequencies are selected from the native FFT grid:

```text
dt = t_max / n_samp
df_native = 1 / t_max
f_Nyquist = 1 / (2 dt)
```

`fmin` and `fmax` define the analysis band. `df` requests a subsampling interval,
but the actual frequencies must coincide with native FFT bins. Choose `t_max`
long enough to contain the latest relevant arrival and `n_samp` large enough
that the Nyquist frequency exceeds `fmax`.

## Define the domain and GF coverage

The integration domain is centred at `(x_center, y_center)` and extends by
`x_extent` and `y_extent` on either side. It begins at the free surface and
continues to `z_extent` depth.

For full-field runs, `xmax_gf` and `zmax_gf` must cover every source–receiver
separation needed by the integration. Include sources lying outside the
integration domain when estimating the maximum horizontal distance.

## Define cavities and test masses

The separate cavity INI file can contain one or more spherical or cuboidal
cavities. Cavity and test-mass z coordinates use positive depth. The drivers
convert them to the internal Cartesian convention, where underground z is
negative.

All four Einstein Telescope test-mass entries are required by the loader. A
driver may select only one of them, or all four values may be repeated for a
single-test-mass study.

## Validate the input

Each driver loads and validates the primary configuration:

```python
config = configLoader.load_config(configFile)
allCavities = configLoader.loadCavities(cavityFile)
validateInputs.validateInputs(config)
```

The validator catches several structural errors, but it does not prove that a
calculation is physically or numerically converged. In particular, users must
still check the FFT grid, GF coverage, late arrivals, quality factors, cavity
placement, and spatial resolution.

## Prepare the geometry

The call

```python
splitAll = createDB.createFinalInputs(config)
```

constructs the required receiver-depth and distance sampling and writes Qseis
input metadata. The resulting `splitAll` geometry is used even by analytic
examples because it contains useful depth sampling information.

This operation cleans top-level Qseis input files. When `qseis_out_del = yes`,
it can also delete existing Qseis run directories. Use a dedicated
`input_folder`.

## Generate and assemble Qseis GFs

This stage is required only for `if_full_field = True`:

```python
results = createDB.runMultiQseis(
    splitAll,
    config.inputPath,
    config.qseisExe,
    nWorkers=config.cpuCoresQseis,
)
```

After checking the process results, assemble the time-domain component files:

```python
splitAllNew = gfLoader.assembleQseisRuns(
    splitAll,
    config.inputPath,
    components,
    fIndices=idxFreq,
    onlySplitAll=False,
    minVel=minVel,
)
```

`onlySplitAll=False` loads, windows, Fourier transforms, and saves the GFs.
Analytic examples use `onlySplitAll=True`; this creates only consolidated
geometrical metadata and does not require Qseis output.

A compatible assembled database can be reused. It is compatible only if its
model, time grid, frequency selection, depth/distance coverage, component list,
and Qseis template match the new calculation.

## Generate one source realization

A stochastic realization contains `n_src` simultaneous sources. The driver
draws positions and, depending on the selected mode, force amplitudes, phases,
directions, or wave polarizations. These values are redrawn for every
realization.

Single-source validation notebooks call the same generator and then overwrite
its output with a deterministic source. This makes the intended source
position, amplitude, and phase explicit.

Random seeds are not currently stored, so stochastic realizations are not
exactly reproducible unless seed handling is added by the user.

## Calculate displacement and Newtonian noise

The domain is split into geological blocks, outer faces, internal interfaces,
and refined cavity regions. For each realization, displacement is evaluated on
the required integration points and converted into:

- a volume or bulk/dipole contribution;
- vertical outer-boundary contributions;
- horizontal free-surface, layer-interface, and bottom-boundary contributions;
  and
- their signed total.

The preferred production setting is:

```ini
save_hv = False
on_fly_nn = True
```

This reduces displacement chunks directly into partial NN integrals and avoids
storing the full three-dimensional seismic field.

## Assemble realizations and scale spectra

Each realization is saved separately. Assembly routines calculate RMS
magnitudes across realizations:

```text
RMS(f) = sqrt(mean(|X_r(f)|^2))
```

The stochastic source field is not normalized by `n_src`. Post-processing
scales the simulated RMS field to a chosen observed surface or underground
displacement ASD component and applies the same factor to the NN spectrum.

For deterministic point-force examples, the finite Qseis wavelet is instead
deconvolved to produce a response per unit force impulse.

## Recommended execution checklist

1. Copy a configuration and cavity file from the closest example.
2. Update every absolute path;
3. verify units, coordinate conventions, and layer-array lengths;
4. select exactly one wavefield mode;
5. verify the FFT band and Qseis time window;
6. verify GF distance and depth coverage;
7. run input validation;
8. confirm that database cleanup is safe;
9. generate or load the GF metadata/database;
10. inspect Qseis return statuses when Qseis is used;
11. run a small number of realizations first;
12. inspect displacement and separate NN contributions;
13. perform grid, domain, time-window, and realization-count convergence tests;
14. retain configurations, GF metadata, outputs, and software revisions.

