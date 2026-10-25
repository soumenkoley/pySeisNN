# Qseis vertical-force NN for a layered half-space

This chapter documents `getSphNNVertForceLayer.ipynb`. The notebook calculates
the seismic displacement and Newtonian acceleration produced by a single
vertical point force at the free surface of a layered elastic half-space. The
detector is represented by a spherical underground cavity containing one test
mass.

Like `getSphNNVertForceHomo.ipynb`, this example generates and uses a complete
Qseis Green's-function (GF) database. The important difference is that wave
propagation now occurs through a genuinely layered model. Consequently, the
full Qseis field must be used: the analytic homogeneous body- and Rayleigh-wave
modes are not valid substitutes.

The result is independently compared with a matching layered SPECFEM3D
simulation. Both calculations are deconvolved by their respective source-time
functions and expressed as Newtonian-acceleration response per unit force,
`m s^-2 N^-1`. The comparison treats bulk, truncation-surface, and total NN
separately.

## Workflow at a glance

```text
layered model + Qseis template
              |
              v
generate depth- and distance-dependent Qseis inputs
              |
              v
run Qseis jobs in parallel
              |
              v
window, FFT, and assemble the layered GF database
              |
              v
apply one vertical surface force at (-2000, 0, 0)
              |
              v
synthesize displacement throughout the integration domain
              |
              v
integrate bulk and truncation-surface NN
              |
              v
remove the Qseis source-wavelet response
              |
              +----------------------+
                                     |
SPECFEM bulk + surface gravity       |
              |                      |
              v                      |
interpolate Ricker STF to output grid|
              |                      |
              v                      |
complex FFT and source deconvolution |
              |                      |
              +-------------> compare bulk, surface,
                                and total responses
```

## What the example demonstrates

The notebook shows how to:

- define an attenuating layered elastic half-space;
- generate Qseis jobs over the required receiver depths and distances;
- execute and assemble a frequency-domain GF database;
- interpolate GFs across layer-dependent receiver grids;
- simulate the full displacement field from one vertical point force;
- calculate volume and truncation-surface NN contributions;
- evaluate displacement at the surface and test-mass depth;
- deconvolve the finite Qseis source wavelet to obtain a unit-force impulse
  response;
- read bulk and surface gravity from the modified SPECFEM3D solver;
- deconvolve the actual SPECFEM Ricker source-time function; and
- compare all three Cartesian components of the two calculations.

## Required inputs

The notebook uses:

- `configParseHalfSpaceFullFieldLayer.ini` for the material model, numerical
  domain, Qseis settings, and output paths;
- `SingleSphCavityHalfSpace.ini` for the spherical cavity;
- the Qseis input template selected by `template_file`; and
- the compiled Qseis executable selected by `qseis_exe`.

The SPECFEM3D comparison additionally reads:

- `stat1.grav`, containing the bulk/dipole gravity acceleration;
- `stat1.surf.grav`, containing the outer truncation-surface contribution from
  the modified SPECFEM3D implementation; and
- `plot_source_time_function.txt`, containing the actual Ricker source-time
  function used by the solver.

The current comparison path is:

```python
SPEC_OUTPUT = "/home/koley/specfemRuns/LayerHomoSphNNSolid/OUTPUT_FILES"
```

`stat1.surf.grav` is not a standard SPECFEM3D output; it requires the modified
surface-gravity implementation used for this study.

The example contains absolute paths. Change them to valid local paths before
running it. General configuration fields are described in
[`configuration_complete.md`](../configuration_complete.md).

## Layered physical model

The example configuration defines:

```ini
[model]
thickness = 50.0, 10000.0, 0.0
vp = 2000.0, 5000.0, 5000.0
vs = 1000.0, 3000.0, 3000.0
rho = 2500.0, 2700.0, 2700.0
qp = 100, 300, 300
qs = 50, 150, 150
rhoAir = 1.2
```

This represents:

| Depth range | `vp` | `vs` | Density | `Qp` | `Qs` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–50 m | 2000 m/s | 1000 m/s | 2500 kg/m³ | 100 | 50 |
| 50–10050 m | 5000 m/s | 3000 m/s | 2700 kg/m³ | 300 | 150 |
| Below 10050 m | 5000 m/s | 3000 m/s | 2700 kg/m³ | 300 | 150 |

The final zero thickness denotes the lower half-space. The second layer and
bottom half-space have identical properties, so the only physical subsurface
contrast is at 50 m depth. The 10 km second-layer thickness still forms part of
the Qseis model representation.

`rhoAir = 1.2 kg/m³` is used in the density contrast for the top truncation
surface. It is not an elastic air layer through which Qseis propagates waves.

The analysed band is 2.0–7.9 Hz with a requested spacing of 0.1 Hz. The
integration grid uses `lambda_res = 9` and is controlled locally by the
shortest relevant wavelength. In the shallow layer, its low wave speeds lead
to finer spatial sampling than in the deeper material.

## Domain, cavity, and test mass

The calculation domain is:

```text
-3000 <= x <= 3000 m
-3000 <= y <= 3000 m
0 <= depth <= 3000 m
```

The spherical cavity is centred at 250 m depth:

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

The notebook uses only `itm_ac`, located at the cavity centre. The positive
configured depth is converted internally to the Cartesian point
`(0, 0, -250 m)`. Although four test-mass positions are present in the
configuration, the line

```python
itmAll = np.vstack((itmACUse,))
```

selects only this test mass for the calculation.

The notebook sets `fullSpace = False`, so the free surface and half-space
surface terms are retained.

## Required simulation mode

The configuration selects:

```ini
[simFlags]
if_body = False
if_ray = False
if_full_field = True
save_hv = False
decoupled_hv = True
on_fly_nn = True
```

These choices are essential for a layered medium. `if_body` and `if_ray`
represent analytic homogeneous-medium fields and must remain false.
`if_full_field = True` makes the displacement routines load and interpolate
the Qseis GFs. With `on_fly_nn = True`, receiver-grid displacement is reduced
to NN during simulation rather than being written and read as a complete
wavefield.

The source configuration contains one source and one realization:

```ini
n_rea = 1
n_src = 1
```

This example computes a deterministic transfer response, not an ensemble of
stochastic seismic fields.

## Stage 1: preparing and running Qseis

The database workflow begins with:

```python
splitAll = createDB.createFinalInputs(config)

results = createDB.runMultiQseis(
    splitAll,
    config.inputPath,
    config.qseisExe,
    nWorkers=config.cpuCoresQseis,
)
```

`createFinalInputs` determines the depth and horizontal-distance sampling,
splits receiver distances into jobs limited by `max_receivers`, inserts the
layered model and run parameters into copies of the Qseis template, and writes
the job metadata. Every Qseis job is then executed in its own directory.

The configured GF limits are:

```ini
xmax_gf = 9000
zmax_gf = 4000
max_receivers = 50
t_max = 40
n_samp = 2048
```

The notebook executes every generated Qseis job with `runMultiQseis`. Inspect
the returned process statuses before assembling the results; the message
`Database generation complete` indicates that the parallel calls returned, but
does not by itself verify the contents of every run.

The 9000 m horizontal limit exceeds the maximum diagonal separation in the
6 km by 6 km numerical domain, and the 4000 m depth limit exceeds the 3000 m
integration depth. The time-domain sampling interval is
`40 / 2048 = 0.01953125 s`.

### Existing database deletion

The example sets `qseis_out_del = yes`. When `createFinalInputs` prepares the
input directory, an existing database under that path can be deleted and
replaced. Use a dedicated output directory and confirm that regeneration is
intended before executing the first cell.

As written, rerunning the first cell regenerates the input files and metadata
and executes all Qseis jobs again. To reuse an existing database, the user must
explicitly skip or guard the generation calls. Reuse is valid only if the
layered model, time sampling, receiver sampling, GF extents, and Qseis template
are unchanged.

## Stage 2: assembling the layered GF database

The notebook selects the requested FFT bins and assembles these components:

```python
components = [
    "fh-2.tz",
    "fh-2.tr",
    "fh-2.tt",
    "fz-2.tz",
    "fz-2.tr",
]
```

It then calls:

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

`onlySplitAll=False` is required because the actual Qseis output must be
assembled. For each receiver depth, the routine concatenates the
distance-split jobs, applies an arrival window, computes the real FFT, retains
the selected frequencies, and writes:

- `Layer<id>_gf_fft.npy`;
- `Layer<id>_distVec.npy`; and
- `splitAllNew.npy`.

`minVel = 200 m/s` controls the late-arrival window used during assembly. It is
not a layer velocity from the geological model. It should be low enough to
preserve all relevant layered arrivals while preventing unwanted late-time or
wraparound content.

## Stage 3: deterministic vertical-force source

The general source generator is called first, after which its result is
overwritten:

```python
xSrc[0] = -2000.0
ySrc[0] = 0.0
azSrc[0] = np.pi / 2

srcMeta["ampV"][:] = 1.0
srcMeta["phiV"][:] = 0.0
srcMeta["ampH"] *= 0.0
srcMeta["phiH"] *= 0.0
```

The physical source is therefore a unit-amplitude downward vertical force at
the free-surface point `(-2000, 0, 0)`, with zero phase and no horizontal-force
component. Its horizontal separation from the test mass is 2000 m.

The source array is constant across the selected frequency bins before the
finite Qseis source-wavelet response is taken into account.

## Displacement reference positions

This notebook evaluates displacement at:

```python
zList = [0.0, cavDepthUse]
```

which corresponds to:

```text
surface reference:   0 m depth
underground reference: 250 m depth
```

The underground reference is at the cavity centre and test-mass depth. This is
different from `getSphNNVertForceHomo.ipynb`, which currently uses the cavity
top at 230 m as its second displacement reference. The distinction affects the
plotted underground displacement but does not move the NN evaluation point.

The per-realization displacement result is stored in
`surfDeepDispRea0.npz`.

## Stage 4: displacement and NN integration

The notebook constructs a layer-dependent background grid, refines the region
around the spherical cavity, associates the selected test mass with the
cavity, and divides the work into blocks and boundary faces.

At every requested integration point, the full-field routines:

1. select the appropriate assembled GF depths;
2. interpolate in receiver depth and horizontal source–receiver distance;
3. rotate radial, transverse, and vertical responses into Cartesian
   components;
4. apply and sum the source contributions; and
5. reduce the displacement field into NN contributions.

The NN archive `NNFullRea0.npz` contains the gravitationally scaled results:

| Array | Meaning |
| --- | --- |
| `IBlockTot` | Volume, or bulk/dipole, contribution. |
| `IVertFaceTot` | Vertical outer truncation-surface contribution. |
| `IHorFaceTot` | Horizontal interfaces and free-/bottom-surface contribution. |
| `ITot` | Total contribution, `IBlockTot - IVertFaceTot - IHorFaceTot`. |

The layer boundary at 50 m changes both the elastic response and the density
structure used by the NN integration. The top surface additionally uses the
rock–air density contrast. These interface contributions are part of the
layered calculation and should not be interpreted as merely geometric
corrections around the cavity.

## Qseis source wavelet and deconvolution

The selected Qseis template uses the normalized square half-sinusoid wavelet
with a duration of four time samples. For this configuration:

```text
dt = 40 / 2048 = 0.01953125 s
tau = 4 dt = 0.078125 s
```

The notebook obtains its frequency response with:

```python
waveletF, tau = simDisp.qseisWaveletResponse(
    freqOut,
    tMax,
    nSamp,
    waveletDurationSamples=4.0,
)
```

After realization assembly has converted the results to magnitudes, it
calculates:

```python
Wnn = np.abs(waveletF[:, None, None])

deconDisp = dispPointAllRea / Wnn * dt
deconFullNN = nnAllRea / Wnn * dt
deconVolNN = nnVolAllRea / Wnn * dt
deconSurfNN = nnSurfAllRea / Wnn * dt
```

The deconvolved NN spectra represent acceleration response per unit force
impulse, with units `m s^-2 N^-1`. This is magnitude-only deconvolution: phase
has already been discarded by the realization-assembly step. Frequencies at
which the wavelet spectrum becomes small should be treated cautiously.

## Diagnostic figures

The second notebook cell plots, for the x, y, and z components:

1. surface and 250 m displacement magnitudes;
2. their deconvolved unit-impulse responses;
3. raw total, volume, and truncation-surface NN; and
4. deconvolved total, volume, and surface NN responses.

An analytic homogeneous body-wave estimate is included as qualitative context.
It is not an exact prediction for the layered point-force field. The analytic
Rayleigh curve is calculated but its plotting command is commented out.

### Fourier normalization

The raw NN spectra and the analytic Rayleigh/body reference curves are
multiplied by `dt`, consistently with the homogeneous vertical-force notebook.
The same factor appears in the wavelet-deconvolved displacement and NN arrays.
This converts the unnormalized discrete FFT convention toward the continuous
Fourier-amplitude convention used by these diagnostic plots.

## SPECFEM3D validation model

The third notebook cell compares pySeisNN with the SPECFEM3D run
`LayerHomoSphNNSolid`. The checked SPECFEM setup uses:

- a 1 N point force at `(-2000, 0, 0)`;
- a downward direction, specified by `Z_UP = -1`;
- a Ricker source-time function with `hdurorf0 = 5.0`;
- a gravity station at `(0, 0, -250 m)`;
- a spherical cavity centred at the gravity station;
- a 50 m shallow layer with `vp = 2000 m/s`, `vs = 1000 m/s`, and
  `rho = 2500 kg/m^3`; and
- deeper material with `vp = 5000 m/s`, `vs = 3000 m/s`, and
  `rho = 2700 kg/m^3`.

The source and gravity-station coordinates match the pySeisNN calculation.
The material file also contains an air region with density `1.225 kg/m^3`,
close to the `rhoAir = 1.2 kg/m^3` used by pySeisNN. This small density
difference should be remembered when interpreting very precise surface-term
agreement.

SPECFEM's material file specifies its attenuation parameters in the format
required by that solver. They should not be compared column-by-column with
Qseis `qp` and `qs` without accounting for the solver's quality-factor
convention.

The SPECFEM run uses:

```text
DT = 0.00005 s
NSTEP = 160000
simulation duration = 8 s
gravity-output interval = 0.02 s
```

The gravity files consequently contain 400 samples from 0.02 to 8.00 s. This
duration is shorter than the 40 s Qseis record. Agreement therefore also
depends on the relevant SPECFEM signal having sufficiently decayed within its
8 s window.

## Reading SPECFEM gravity output

The validation reads `stat1.grav` and `stat1.surf.grav`. Each file contains:

```text
time, acceleration_x, acceleration_y, acceleration_z
```

Rows containing non-finite values are removed independently. The notebook then
requires bulk and surface data to have identical lengths, sampling intervals,
and time vectors. A mismatch raises an error before Fourier transformation.

The bulk file represents the volume/dipole contribution. The surface file is
the outer truncation-surface contribution written by the modified solver.

## Aligning the SPECFEM source-time function

`plot_source_time_function.txt` is sampled on the fine solver time step, while
gravity is output on the coarser 0.02 s grid. The source file uses a coordinate
whose Ricker peak is at zero. The notebook converts it to solver time using:

```python
tStfSolver = tStfSpec - tStfSpec[0] + dtStfSpec
```

It then applies cubic interpolation onto the gravity time vector, with values
outside the source interval filled by zero. This produces an STF and gravity
series with the same time origin, sample count, and interval before the FFT.

## SPECFEM complex deconvolution

The aligned source and gravity arrays are transformed with `np.fft.rfft`:

```python
Wspec = np.fft.rfft(stfOnGrav)
ABulk = np.fft.rfft(accBulk, axis=0)
ASurf = np.fft.rfft(accSurf, axis=0)
```

Because numerator and denominator use the same discrete time grid, their `dt`
factors cancel. For a force amplitude `F0 = 1 N`, the transfer functions are:

```python
HSpecBulk = ABulk / (F0 * Wspec[:, None])
HSpecSurf = ASurf / (F0 * Wspec[:, None])
HSpecTotal = HSpecBulk - HSpecSurf
```

Bulk and surface quantities remain complex until after subtraction. This is
essential because phase controls cancellation:

```text
|H_bulk - H_surface| != |H_bulk| - |H_surface|
```

The minus sign matches pySeisNN's total convention. Before interpreting the
result, the notebook prints the minimum `|Wspec|` in the 2–8 Hz comparison
band. Frequencies near a source-spectrum zero would make the deconvolution
unstable.

## Comparison plots

The notebook compares 2–8 Hz and produces three figures, each with x, y, and z
panels:

1. `deconVolNN` against `HSpecBulk`;
2. `deconSurfNN` against `HSpecSurf`; and
3. `deconFullNN` against `HSpecTotal`.

Magnitudes are plotted only after each SPECFEM complex contribution has been
formed. The pySeisNN realization-assembly routines already return magnitudes,
so the present plots validate transfer amplitudes rather than the complex
pySeisNN phase response.

The y response is theoretically small for this geometry because the source,
cavity centre, and test mass lie in the x–z symmetry plane. It is therefore
more sensitive to numerical asymmetry and is plotted on a restricted scale in
the bulk comparison.

## Running the example

1. Update the configuration, cavity, template, executable, and output paths.
2. Confirm the three-layer arrays have equal lengths and end in a zero-thickness
   half-space.
3. Confirm `if_body = False`, `if_ray = False`, and `if_full_field = True`.
4. Check that `xmax_gf` and `zmax_gf` cover every source–receiver separation
   required by the integration domain.
5. Confirm that deletion of any existing Qseis database is intended.
6. Run `createFinalInputs` and inspect the number of generated jobs.
7. Run Qseis and check every returned process status.
8. Assemble the database with `onlySplitAll=False`.
9. Confirm that the source has been overwritten to the single vertical force
   at `(-2000, 0, 0)`.
10. Run the displacement and NN calculation.
11. Verify that `surfDeepDispRea0.npz` and `NNFullRea0.npz` were created.
12. Inspect the Qseis wavelet magnitude before interpreting deconvolved output.
13. Plot and compare the total, volume, and surface NN components.
14. Point `SPEC_OUTPUT` to the matching layered SPECFEM run.
15. Verify the SPECFEM source, station, layer boundary, material properties,
   cavity, attenuation convention, and output duration.
16. Run the time-grid and source-spectrum checks.
17. Compare the bulk, surface, and total transfer-response figures.

## Consistency and convergence checks

Before interpreting the result, verify:

- layer thicknesses, velocities, densities, and attenuation factors;
- the 50 m physical interface depth;
- the air-density value used for the top surface;
- source direction, coordinate, and normalization;
- cavity and test-mass coordinates;
- Qseis template source depth and wavelet selection;
- receiver-depth and distance coverage;
- frequency-bin selection and wavelet deconvolution;
- signs used when combining bulk and surface contributions; and
- Cartesian component conventions.

For the cross-solver comparison, also verify that Qseis/pySeisNN and SPECFEM
use equivalent:

- layer-interface depths and elastic properties;
- attenuation conventions;
- air and solid densities at the free surface;
- spherical-cavity geometry and gravity-evaluation point;
- downward force direction and 1 N normalization;
- bulk/surface sign convention; and
- frequency band and source-time-function normalization.

Numerical convergence should be checked with respect to:

- Qseis receiver-depth and horizontal-distance sampling;
- the `minVel` arrival-window parameter;
- background grid resolution in each geological layer;
- cavity refinement and the fixed buffer used by the gridding code;
- total integration-domain extent; and
- `t_max` and `n_samp`;
- SPECFEM mesh resolution near the cavity and 50 m interface;
- SPECFEM domain boundaries and absorbing layers; and
- the shorter 8 s SPECFEM output duration.

## Outputs to retain

For reproducibility, retain:

- the main and cavity configuration files;
- the exact Qseis template and executable version;
- `splitAll.mat` and `splitAllNew.npy`;
- every `Layer*_gf_fft.npy` and `Layer*_distVec.npy` file;
- Qseis run logs or returned statuses;
- `surfDeepDispRea0.npz` and `NNFullRea0.npz`;
- the source-wavelet parameters and deconvolution convention;
- the SPECFEM `Par_file`, `FORCESOLUTION`, gravity-station definition, mesh,
  and material assignments;
- `stat1.grav`, `stat1.surf.grav`, and
  `plot_source_time_function.txt`;
- the comparison plots; and
- the software revision used for the calculation.

## Current limitations

- The notebook regenerates and reruns the Qseis database whenever its first
  cell is executed.
- `createFinalInputs` rewrites Qseis inputs and metadata and may delete existing
  run directories according to `qseis_out_del`.
- The assembled realization outputs retain magnitudes rather than complex
  response phases, so the plots do not validate pySeisNN phase.
- The analytic body-wave reference assumes a homogeneous field and is only a
  qualitative comparison for this layered calculation.
- Several absolute paths must be edited for another installation.
- The SPECFEM surface term requires a modified solver implementation.
- The SPECFEM record is 8 s long, whereas the Qseis record is 40 s; late-time
  truncation should be checked explicitly.
- The configured pySeisNN air density is `1.2 kg/m^3`, while the inspected
  SPECFEM material file uses `1.225 kg/m^3`.
