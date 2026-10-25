# Qseis vertical-force NN for a homogeneous half-space

This chapter documents `getSphNNVertForceHomo.ipynb`. It is the first tutorial
that generates and uses a complete Qseis Green's-function database rather than
an analytic plane-wave displacement field. The notebook calculates the seismic
displacement and Newtonian acceleration caused by one vertical point force at
the free surface of a homogeneous half-space containing a spherical cavity.

The result is independently validated against a SPECFEM3D simulation of the
same source-receiver arrangement. Both calculations are deconvolved by their
respective source-time functions so that the comparison is made in units of
Newtonian acceleration response per unit force:

```text
m s^-2 N^-1
```

This example deserves particular care because it combines database generation,
frequency-domain interpolation, source-wavelet normalization, numerical NN
integration, and cross-solver validation.

## Workflow at a glance

```text
main configuration + Qseis template
              |
              v
generate per-depth/per-distance Qseis inputs
              |
              v
run Qseis jobs -> time-domain GF component files
              |
              v
window, FFT, assemble, and save GF database
              |
              v
simulate one vertical surface force at (-2000, 0, 0)
              |
              v
integrate bulk + truncation-surface NN at cavity/test mass
              |
              v
deconvolve Qseis default wavelet
              |
              +--------------------+
                                   |
SPECFEM gravity + Ricker STF        |
              |                    |
              v                    |
interpolate STF, FFT, deconvolve    |
              |                    |
              +----------> compare transfer magnitudes
```

## What the example demonstrates

The notebook shows how to:

- create Qseis input files for all required receiver depths and distances;
- execute the Qseis jobs in parallel;
- assemble selected GF components into frequency-domain layer files;
- interpolate Qseis GFs at arbitrary integration points;
- synthesize displacement from one vertical point force;
- integrate volume and truncation-surface NN contributions;
- remove the finite Qseis source wavelet;
- read and deconvolve SPECFEM3D gravity output;
- compare bulk, surface, and total NN transfer functions; and
- quantify the frequency-dependent discrepancy between the two solvers.

## Required files and external data

The pySeisNN calculation expects:

- `configParseHalfSpaceFullFieldHomo.ini`;
- `SingleSphCavityHalfSpace.ini`;
- a compatible Qseis input template;
- a compiled Qseis executable; and
- writable directories for Qseis inputs/GFs, temporary displacement, and
  realization results.

The SPECFEM3D comparison expects these files under one `OUTPUT_FILES`
directory:

- `stat1.grav` — bulk/dipole gravity acceleration;
- `stat1.surf.grav` — outer truncation-surface acceleration from the modified
  SPECFEM3D implementation; and
- `plot_source_time_function.txt` — the actual Ricker source-time function.

The comparison path is currently hard-coded:

```python
SPEC_OUTPUT = "/path/to/HomoSphereNNSolid/OUTPUT_FILES"
```

The SPECFEM surface-gravity file requires the modified SPECFEM3D version used
for this validation; it is not a standard output of an unmodified installation.
Installation and patching instructions will be documented separately.

## Physical model

The Qseis configuration uses the homogeneous model:

```ini
[model]
thickness = 10000.0, 0.0
vp = 5000.0, 5000.0
vs = 3000.0, 3000.0
rho = 2700.0, 2700.0
qp = 300, 300
qs = 150, 150
rhoAir = 0.0
```

Identical layer and half-space properties make the elastic medium homogeneous.
The zero final thickness denotes the bottom half-space. `rhoAir = 0` treats the
region above the free surface as vacuum for the NN surface-density contrast.

The analysed frequency band is 2.0–7.9 Hz with a requested 0.1 Hz spacing. The
integration grid uses `lambda_res = 9`, finer than the earlier introductory
examples.

## Domain, cavity, and test mass

The numerical domain spans:

```text
-3000 <= x <= 3000 m
-3000 <= y <= 3000 m
0 <= depth <= 3000 m
```

The spherical cavity is:

```ini
[CavityA]
shape = sphere
xC = 0.0
yC = 0.0
zC = 250.0
radius = 20.0
```

The selected test mass `itm_ac` is at the cavity centre. Its configured
positive depth is converted to the internal point `(0, 0, -250 m)`.

The SPECFEM gravity station file also places its station at:

```text
(0, 0, -250 m)
```

so the two calculations evaluate Newtonian acceleration at the same nominal
location.

The notebook sets:

```python
fullSpace = False
```

and uses the half-space free-surface treatment throughout.

## Full-field flags

The configuration must select Qseis full-field mode:

```ini
[simFlags]
if_body = False
if_ray = False
if_full_field = True
save_hv = False
decoupled_hv = True
on_fly_nn = True
```

Unlike the analytic examples, `if_full_field = True` causes the displacement
functions inside the NN integration to load and interpolate the assembled
Qseis GFs.

The tutorial uses one source and one realization:

```ini
[simSource]
n_rea = 1
n_src = 1
```

This is a deterministic impulse-response validation, not a stochastic
ensemble.

## Stage 1: creating Qseis input files

The notebook begins with:

```python
splitAll = createDB.createFinalInputs(config)
```

This performs the following operations:

1. calculates the required depth and horizontal-distance sampling;
2. partitions receiver distances into jobs limited by `max_receivers`;
3. formats the homogeneous velocity model in Qseis units;
4. copies the master template for every Qseis job;
5. substitutes receiver depth, distance range, receiver count, `t_max`, and
   `n_samp`; and
6. writes `splitAll.mat`, the metadata table describing all jobs.

The Qseis source depth in the tutorial template is zero, so the GF source is at
the free surface. Receiver depth varies through the database, and horizontal
receiver distance covers the range required by the numerical domain.

### Destructive cleanup behavior

`createFinalInputs` cleans `input_folder` before writing new inputs. Top-level
`.dat` files are removed, and with `qseis_out_del = yes`, existing Qseis run
subdirectories are recursively deleted. Use a directory dedicated to this
database.

## Stage 2: running Qseis

The notebook executes all jobs with:

```python
results = createDB.runMultiQseis(
    splitAll,
    config.inputPath,
    config.qseisExe,
    nWorkers=config.cpuCoresQseis,
)
```

Each job runs in its own directory and produces the selected vertical- and
horizontal-force GF components. The components used later are:

```python
components = [
    "fh-2.tz",
    "fh-2.tr",
    "fh-2.tt",
    "fz-2.tz",
    "fz-2.tr",
]
```

For this vertical-force source, the directly relevant Qseis outputs are the
vertical and radial receiver motions from `fz-2`, while the complete component
set is assembled because the database and shared displacement routines also
support horizontal forces.

Qseis database generation scales well across independent jobs but can create
hundreds of run directories and substantial output. Check the returned process
statuses before continuing.

### Reusing an existing database

As written, the notebook regenerates inputs and reruns every Qseis job whenever
the first cell is executed. Once a compatible database has been validated,
comment or guard the input-generation and execution lines to avoid unnecessary
destructive regeneration. A database is reusable only when its material model,
time sampling, receiver-depth grid, horizontal extent, and Qseis template are
compatible with the new calculation.

## Stage 3: assembling frequency-domain GFs

The output frequency bins are selected with:

```python
freqOut, idxFreq, df_native = simDisp.getFreqGrid(
    tMax, nSamp, fMin, fMax, df
)
```

The Qseis outputs are then assembled with:

```python
splitAllNew = gfLoader.assembleQseisRuns(
    splitAll,
    fInpPath,
    components,
    outPath=None,
    fIndices=idxFreq,
    onlySplitAll=False,
    minVel=200.0,
)
```

`onlySplitAll=False` is essential here. For every database depth,
`assembleQseisRuns`:

- loads and concatenates component files from all distance-split jobs;
- applies the GF arrival window controlled by `minVel`;
- transforms the time-domain GFs with a real FFT;
- keeps only `idxFreq`;
- writes `Layer<id>_gf_fft.npy`;
- writes `Layer<id>_distVec.npy`; and
- writes `splitAllNew.npy` describing the assembled layer files.

The tutorial uses `minVel = 200 m/s`. This arrival-window parameter should be
low enough to retain all physical arrivals needed by the comparison while
removing wraparound or unwanted late-time content.

During NN integration, `getInterpolatedGFFFT` loads the relevant layer files,
crops the distance range, and interpolates between receiver depths when the
requested depth is not an exact database plane.

## Stage 4: defining the vertical force

The common source generator is called once, then its result is overwritten to
represent a deterministic source:

```python
xSrc[0] = -2000.0
ySrc[0] = 0.0
azSrc[0] = np.pi / 2

srcMeta["ampV"][:] = 1.0
srcMeta["phiV"][:] = 0.0
srcMeta["ampH"] *= 0.0
srcMeta["phiH"] *= 0.0
```

The physical source is therefore:

- located at `(-2000, 0, 0)`;
- vertical and downward in the Qseis depth-positive convention;
- unit amplitude at every selected frequency before accounting for the Qseis
  wavelet; and
- zero phase.

The horizontal-force contribution is exactly zero. `azSrc` is immaterial for
an axisymmetric vertical point force, although it remains part of the shared
source interface.

The source-to-test-mass horizontal separation is 2000 m.

## Reference displacement locations

The notebook saves displacement at:

```python
zList = [0.0, cavDepthUse - radius]
```

For a cavity centred at 250 m with radius 20 m, this is:

```text
surface: 0 m
underground reference: 230 m depth
```

The second point is the depth of the cavity top, not its centre. This choice
should be retained when reproducing the existing validation, and it should be
stated explicitly if these displacement spectra are used for normalization.

The displacement output is saved as `surfDeepDispRea0.npz`.

## Stage 5: NN integration

The cavity association, background grid, wall-refinement region, and
layer/block decomposition follow the earlier spherical examples. The default
refinement settings are:

```python
etaUse = 0.5
minRefine = 20.0
```

At every integration depth, the full-field functions load the assembled GFs,
interpolate them in depth and horizontal distance, rotate the radial/vertical
components into Cartesian coordinates, sum the source contribution, and
immediately reduce receiver chunks to NN when `on_fly_nn = True`.

The realization output contains:

| Array | Meaning |
| --- | --- |
| `IBlockTot` | Bulk/dipole NN contribution of the material volume. |
| `IVertFaceTot` | Contribution from vertical outer truncation surfaces. |
| `IHorFaceTot` | Horizontal interfaces and free-/bottom-surface contribution. |
| `ITot` | Total, `IBlockTot - IVertFaceTot - IHorFaceTot`. |

After multiplication by Newton's gravitational constant, the arrays are saved
as `NNFullRea0.npz`.

The assembly functions are then called even though there is one realization.
They return magnitude spectra for displacement, total NN, volume NN, and
combined surface NN.

## Qseis source wavelet

The Qseis template selects the default normalized square half-sinusoid:

```text
w(t) = (2/tau) sin^2(pi t/tau),  0 < t < tau
```

with a duration of four time samples. For the configured values:

```text
t_max = 40 s
n_samp = 2048
dt = 40 / 2048 = 0.01953125 s
tau = 4 dt = 0.078125 s
```

The wavelet integrates to unity and approximates a physical delta impulse, but
it still has a finite frequency response. The notebook evaluates that response
analytically at `freqOut`:

```python
waveletF, tau = simDisp.qseisWaveletResponse(
    freqOut,
    tMax,
    nSamp,
    waveletDurationSamples=4.0,
)
```

The complex response is:

```text
W(f) = exp(-i pi nu) [sinc(nu)
       + 0.5 sinc(nu - 1)
       + 0.5 sinc(nu + 1)],

nu = f tau.
```

## Qseis wavelet deconvolution

After ensemble assembly, the notebook calculates:

```python
Wnn = np.abs(waveletF[:, None, None])

deconDisp = dispPointAllRea / Wnn * dt
deconFullNN = nnAllRea / Wnn * dt
deconVolNN = nnVolAllRea / Wnn * dt
deconSurfNN = nnSurfAllRea / Wnn * dt
```

The division removes the magnitude response of the four-sample Qseis wavelet;
the factor `dt` converts the discrete FFT normalization to the impulse-response
normalization used in the comparison. The deconvolved NN quantities are
reported as response per unit force impulse, in `m s^-2 N^-1`.

Because realization assembly has already converted the complex results to RMS
magnitudes, this is a magnitude-only deconvolution. It does not recover or
compare the complex pySeisNN phase response. Frequencies where `|waveletF|` is
small should be excluded or treated cautiously because deconvolution amplifies
numerical errors.

## Diagnostic pySeisNN plots

The second cell creates four three-component figures:

1. raw surface and underground displacement spectra;
2. deconvolved displacement response for a unit vertical force impulse;
3. raw total, volume, and truncation-surface NN, with analytic Rayleigh/body
   reference curves; and
4. deconvolved total, volume, and surface NN response.

The analytic Rayleigh and body-wave curves are qualitative context. The Qseis
point-force field contains the complete homogeneous half-space response and is
not a pure plane-Rayleigh or isotropic-body-wave field.

The plotting cell multiplies some raw spectra by `dt`; this is part of its
Fourier-amplitude convention and should not be confused with the explicit
wavelet deconvolution used for the unit-impulse response.

## SPECFEM3D validation setup

The checked SPECFEM setup uses:

- a force source at `(-2000, 0, 0)`;
- force factor `F0 = 1 N`;
- direction `(E, N, Z_UP) = (0, 0, -1)`, i.e. downward;
- a gravity station at `(0, 0, -250 m)`;
- solver time step `DT = 0.00005 s`; and
- gravity output sampled every `0.02 s`, corresponding to 50 Hz.

These source and station locations match the Qseis/pySeisNN validation geometry.
The SPECFEM material model, cavity mesh, absorbing boundaries, and modified
surface-gravity implementation must also match the intended pySeisNN model;
those mesh-generation details are external to this notebook.

SPECFEM uses a 1 N Ricker source-time function. The notebook reads the actual
saved STF instead of reconstructing it from nominal source parameters.

## Reading SPECFEM gravity output

Both gravity files are expected to contain four columns:

```text
time, acceleration_E, acceleration_N, acceleration_Z
```

The notebook removes rows containing non-finite values and verifies that bulk
and surface outputs have identical sample counts, time steps, and time vectors.
These checks must pass before their complex spectra can be combined.

The SPECFEM source file is sampled at the fine solver time step, while gravity
is written at the coarser 0.02 s interval. The notebook converts source-file
time to solver time and interpolates the Ricker wavelet onto the gravity time
grid using cubic interpolation:

```python
stfInterp = interp1d(
    tStfSolver,
    stfSpec,
    kind="cubic",
    bounds_error=False,
    fill_value=0.0,
)
stfOnGrav = stfInterp(tGrav)
```

This alignment is essential. The source and gravity FFTs must have the same
time step, length, and time origin for valid complex deconvolution.

## SPECFEM complex deconvolution

The notebook calculates:

```python
Wspec = np.fft.rfft(stfOnGrav)
ABulk = np.fft.rfft(accBulk, axis=0)
ASurf = np.fft.rfft(accSurf, axis=0)

HSpecBulk = ABulk / (F0 * Wspec[:, None])
HSpecSurf = ASurf / (F0 * Wspec[:, None])
HSpecTotal = HSpecBulk - HSpecSurf
```

Here:

```text
A(f) = H(f) F0 W(f)
```

so `H(f)` is the NN acceleration transfer function per newton. No additional
`dt` appears in the ratio because source and acceleration use the same discrete
FFT grid and their time-step factors cancel.

Bulk and surface spectra are kept complex until after subtraction. This is
mandatory:

```text
|H_bulk - H_surface| != |H_bulk| - |H_surface|
```

The sign convention matches pySeisNN's total:

```text
total = bulk - truncation surface.
```

Before trusting deconvolution, the notebook prints the minimum `|Wspec|` in the
2–8 Hz comparison band. A near-zero source spectrum would make the transfer
estimate unstable.

## Comparing pySeisNN and SPECFEM3D

The comparison is restricted to 2–8 Hz. SPECFEM and pySeisNN remain on their
native frequency grids for the main plots.

Three figures compare magnitudes of:

1. `deconVolNN` against `HSpecBulk`;
2. `deconSurfNN` against `HSpecSurf`; and
3. `deconFullNN` against `HSpecTotal`.

Each figure contains x, y, and z components. For this symmetric geometry and a
vertical source on the x axis, the y response should be extremely small; its
relative discrepancy is not a robust validation metric when both solvers are
near numerical zero.

The main validation compares magnitudes, not complex phase. Agreement therefore
tests the component transfer amplitudes and the combined surface correction,
but not the full phase response.

## Time-domain offset diagnostic

The notebook plots SPECFEM bulk x and z acceleration in time and calculates the
mean over the final 20% of samples. This checks for late-time DC offsets or
incomplete decay that could contaminate the low-frequency FFT.

If the offsets are appreciable relative to the signal, detrending, extending
the run, tapering, or revisiting boundary conditions may be necessary before a
frequency-domain comparison.

## Quantitative discrepancy plot

For the x-component bulk response, the notebook interpolates the SPECFEM
magnitude onto the pySeisNN frequency grid and calculates:

```python
diffX = |H_py| - |H_spec|
relative_difference = diffX / |H_py| * 100
```

It plots both the absolute and percentage discrepancy and limits the displayed
relative range to ±10%. This calculation is meaningful only where the
pySeisNN x response is safely nonzero.

Interpolation is performed on magnitudes, so this diagnostic does not measure
complex transfer-function error.

## Running the complete validation

1. Verify the Qseis template, executable, model, source wavelet, and output
   directory.
2. Verify that Qseis database deletion is intended before running
   `createFinalInputs`.
3. Run the database generation and check every Qseis return code.
4. Assemble frequency-domain GFs with `onlySplitAll=False`.
5. Confirm the source is the single vertical force at `(-2000, 0, 0)`.
6. Run the NN integration and confirm `surfDeepDispRea0.npz` and
   `NNFullRea0.npz` are created.
7. Inspect the Qseis wavelet magnitude across 2–8 Hz.
8. Run the Qseis deconvolution and diagnostic plots.
9. Point `SPEC_OUTPUT` at the matching SPECFEM3D run.
10. Verify the SPECFEM source, gravity station, material model, cavity, and
    output sampling.
11. Run the SPECFEM time-grid checks and source interpolation.
12. Inspect the minimum Ricker spectrum magnitude in the comparison band.
13. Compare bulk, surface, and total transfer magnitudes.
14. Inspect late-time offsets and the x-component discrepancy plots.

## Essential consistency checks

Before interpreting agreement, verify that both solvers use the same:

- P-wave velocity, S-wave velocity, density, and attenuation convention;
- free-surface and half-space geometry;
- spherical cavity centre and radius;
- test-mass/gravity-station coordinate;
- source coordinate and downward vertical direction;
- force normalization;
- bulk and truncation-surface sign convention;
- Cartesian component mapping; and
- comparison frequency band.

Also verify solver-specific numerical convergence with respect to:

- Qseis receiver depth/distance sampling and arrival window;
- pySeisNN volume grid and cavity refinement;
- finite integration-domain extent;
- SPECFEM mesh resolution and cavity representation;
- SPECFEM domain boundaries and absorbing layers;
- SPECFEM gravity output interval and total duration; and
- both source spectra used for deconvolution.

## Outputs to retain

For a reproducible validation, retain:

- the main and cavity configurations;
- the exact Qseis template and executable/version;
- `splitAll.mat` and `splitAllNew.npy`;
- all `Layer*_gf_fft.npy` and `Layer*_distVec.npy` files;
- the Qseis run logs or return statuses;
- `surfDeepDispRea0.npz` and `NNFullRea0.npz`;
- the Qseis wavelet parameters and deconvolution code;
- the SPECFEM `Par_file`, `FORCESOLUTION`, mesh, and gravity-station file;
- `stat1.grav`, `stat1.surf.grav`, and
  `plot_source_time_function.txt`; and
- the software revisions and comparison plots.

## Current limitations

- The notebook always regenerates the Qseis database unless the user manually
  disables those lines.
- The pySeisNN assembly produces magnitude spectra, so the current cross-solver
  plots do not validate pySeisNN phase.
- The SPECFEM surface term depends on a modified solver implementation.
- SPECFEM mesh construction is external and not reproduced by this notebook.
- The displacement reference depth is the cavity top at 230 m, while NN is
  evaluated at the cavity-centre test mass at 250 m.
- The analytic Rayleigh/body curves are contextual comparisons, not exact
  representations of the vertical point-force field.
