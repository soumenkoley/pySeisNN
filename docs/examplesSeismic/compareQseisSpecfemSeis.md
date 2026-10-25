# Comparing Qseis and SPECFEM3D seismic Green functions

This chapter documents `compareQseisSpecfemSeis.ipynb`. The notebook compares
surface displacement produced by Qseis and SPECFEM3D for the same homogeneous
half-space, source, and receiver geometry.

The comparison goes beyond plotting the raw seismograms. It:

- creates a dedicated Qseis run at the exact SPECFEM receiver offsets;
- checks raw and 2–8 Hz filtered displacement traces;
- examines both source-time functions in time and frequency;
- deconvolves each solver by its own source-time function;
- compares vertical and radial Green-function amplitudes;
- calculates Qseis/SPECFEM amplitude ratios;
- reconstructs hard-band-limited impulse responses; and
- performs a second, stabilized common-grid deconvolution with a smooth
  zero-phase Butterworth response.

This is a seismic-wavefield validation. It does not calculate Newtonian noise.

## Purpose of the comparison

pySeisNN uses Qseis Green functions to synthesize stochastic displacement
fields. Comparing those GFs with an independent three-dimensional solver tests
several assumptions at once:

- material and source normalization;
- radial and vertical component mapping;
- coordinate signs;
- arrival times and waveform shape;
- source-time-function removal;
- frequency-domain amplitude; and
- consistency after reconstruction in the time domain.

Agreement after deconvolution is more informative than agreement between raw
traces because Qseis and SPECFEM use different source wavelets and sampling
grids.

## Physical model

Both calculations represent a homogeneous elastic half-space with:

```text
vp  = 5000 m/s
vs  = 3000 m/s
rho = 2700 kg/m³
```

The Qseis configuration represents this with identical finite-layer and
half-space properties:

```ini
thickness = 10000.0, 0.0
vp = 5000.0, 5000.0
vs = 3000.0, 3000.0
rho = 2700.0, 2700.0
qp = 300, 300
qs = 150, 150
```

The inspected SPECFEM material file uses the same elastic velocities and
density. SPECFEM and Qseis attenuation parameters must be interpreted using
their respective quality-factor conventions rather than compared blindly
column by column.

## Source and receiver geometry

The SPECFEM point force is located at:

```text
source = (-2000, 0, 0) m
```

`FORCESOLUTION` specifies:

```text
force amplitude = 1 N
E component = 0
N component = 0
Z_UP component = -1
```

The source is therefore a 1 N downward vertical force at the surface.

Eight SPECFEM receivers lie on the x axis:

```text
x = -1500, -1000, -500, 0, 500, 1000, 1500, 2000 m
y = 0 m
depth = 0 m
```

Their source–receiver offsets are consequently:

```text
r = 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000 m
```

The primary comparison uses station `S007` at `(2000, 0, 0)`, giving
`R_TARGET = 4000 m`.

## Component mapping and signs

The receiver is on the positive x side of the source, so SPECFEM's x component
is the radial component for this geometry.

| Physical component | SPECFEM file component | Qseis component |
| --- | --- | --- |
| Radial displacement | `HXX` | `fz-2.tr` |
| Vertical displacement | `HXZ` | `fz-2.tz` |

The source is vertical, so an ideal axisymmetric solution has no transverse
motion along the source–receiver line.

SPECFEM uses z positive upward, while Qseis uses z positive downward. The
notebook therefore applies:

```python
qZ_raw = -qZ_raw
```

when `FLIP_QSEIS_Z = True`. No corresponding sign flip is needed for the
outward radial component in this geometry.

## Comparison band and filter

The principal band is:

```text
FMIN = 2 Hz
FMAX = 8 Hz
Butterworth order = 4
```

The helper `bandpass_zero_phase` constructs a fourth-order Butterworth
bandpass and applies `scipy.signal.sosfiltfilt`. Forward-and-reverse filtering
removes the causal phase shift and produces an effective zero-phase amplitude
response equal to the squared magnitude of the one-pass response.

## Stage 1: dedicated Qseis receiver run

Rather than interpolate a large general-purpose database, the notebook creates
one Qseis input containing exactly the eight SPECFEM offsets:

```python
rExact = np.array([
    500.0, 1000.0, 1500.0, 2000.0,
    2500.0, 3000.0, 3500.0, 4000.0,
])
zRec = 0.0
```

It verifies uniform 500 m spacing because the Qseis template uses an
equidistant interval defined by the first distance, last distance, and number
of receivers.

The velocity-model block is generated with `createDB.formatVelModel`. The
source-radius/integration parameter `srN` is obtained from
`createDB.getNXDP(config)`, ensuring consistency with ordinary pySeisNN
database generation.

The template selected by `config.templateFile` is copied and filled with:

- the velocity model and number of model lines;
- zero receiver depth;
- eight receiver offsets from 0.5 to 4.0 km;
- Qseis duration and sample count; and
- `srN`.

The job is run with:

```python
result = createDB.runSingleQseis(
    layerName,
    inputPath,
    config.qseisExe,
)
```

The return code must be zero before the outputs are used.

### Dedicated-output consistency check

The corrected validation run writes to:

```text
/home/koley/CavityResults/AllTutOut/QseisSpecfemSeisTest/QseisInpN/
```

The inspected generated input now requests eight receivers at zero depth from
0.5 to 4.0 km, and both `fz-2.tr` and `fz-2.tz` contain the expected eight
receiver columns plus time. All 2048 loaded samples are finite.

The notebook variables `rQCheck` and `rExact` verify the requested equidistant
coordinates, but they do not themselves parse the GF headers. Because the
validation directory is reusable, checking the generated input and output
column count remains a useful safeguard before extracting `R_TARGET`.

A robust future code check could require the number of GF data columns to equal
`len(rExact)` before indexing the target trace. Using `np.isclose` rather than
exact equality in the target-index lookup would also make the check less
sensitive to floating-point representation.

## Solver sampling

The Qseis configuration uses:

```text
t_max = 40 s
n_samp = 2048
dt approximately 0.01954 s
```

The inspected SPECFEM run uses:

```text
solver DT = 0.001 s
NSTEP = 10000
saved displacement dt approximately 0.01 s
saved trace time range = -0.24 to 9.75 s
```

The actual SPECFEM source-time function is retained at the fine 0.001 s solver
sampling. These differences are handled explicitly during deconvolution and
common-grid reconstruction.

## Stage 2: loading and windowing Qseis traces

The notebook loads `fz-2.tz` and `fz-2.tr` from the dedicated run and selects
the column corresponding to `R_TARGET`.

It applies the same late-time window used by GF database assembly:

```python
W = gfLoader.genWin(
    tQ,
    np.array([R_TARGET]),
    minVel=200.0,
)[:, 0]
```

The window is applied before Fourier transformation. At 4 km and
`minVel = 200 m/s`, its late-time scale begins around 20 s, well after the
principal direct arrivals. The window reduces late-time wraparound or unwanted
content but means the Qseis and SPECFEM records are not treated identically:
SPECFEM is used without this taper.

The notebook then plots raw-windowed and zero-phase-filtered Qseis vertical and
radial traces, together with their FFT magnitudes.

## Stage 3: loading SPECFEM traces

`load_specfem_trace` searches the SPECFEM output directory for exactly one
file matching the selected station and component, for example:

```text
DB.S007.HXX.semd
DB.S007.HXZ.semd
```

The radial and vertical time vectors must match. The notebook applies the same
2–8 Hz zero-phase Butterworth filter and plots raw/filtered time traces and FFT
magnitudes.

These first two plot groups are diagnostic. Raw or filtered solver traces
should not be expected to overlay directly because their source-time functions
are different.

## Source-time functions

### SPECFEM Ricker wavelet

The SPECFEM source uses a Ricker function with:

```text
hdurorf0 = 5.0
force amplitude = 1 N
```

The exact sampled function is read from
`plot_source_time_function.txt`. Its source-time peak is at `t = 0`, and the
file is sampled at the 0.001 s solver step.

### Qseis wavelet

The bundled Qseis template selects its default normalized square
half-sinusoid:

```text
w(t) = (2/tau) sin²(pi t/tau),  0 < t < tau
tau = 4 dt_Qseis.
```

With the configured Qseis sampling, `tau` is approximately 0.0781 s. The
notebook constructs the wavelet on the native Qseis time vector and prints
`dtQ * sum(stfQ)` as a normalization check; it should be close to one.

### STF diagnostic plot

The notebook plots both source functions and their FFT magnitudes on their own
native grids. Comments in the plotting cell state that the curves are
normalized for a shape comparison, but the current code plots `stfSpec`,
`stfQ`, and their raw FFT magnitudes directly without explicit normalization.
The plot therefore shows both shape and native amplitude/FFT-normalization
differences. This does not affect the later deconvolution, which uses the
unmodified source arrays intentionally.

## Frequency-domain source deconvolution

The first GF comparison keeps each solver on its native time and frequency
grid.

The SPECFEM STF is interpolated from its 0.001 s source grid onto the 0.01 s
seismogram grid. The Qseis wavelet is constructed directly on the Qseis trace
grid. For each solver:

```text
U(f) = G(f) S(f)
G(f) = U(f) / S(f).
```

The code calculates:

```python
GZ_spec = UZ_spec / S_spec
GR_spec = UR_spec / S_spec

GZ_q = UZ_q / S_q
GR_q = UR_q / S_q
```

Because displacement and STF within each solver use the same sample count and
interval, their discrete-transform scaling factors cancel. Simple division is
used without regularization in this first comparison. It is restricted to
2–8 Hz, where both source spectra are expected to remain safely nonzero.

The source spectrum should nevertheless be inspected explicitly; division by
a small value amplifies noise and numerical error.

## Common frequency grid

The solver FFT bins differ. The notebook first finds the available Qseis bins
inside 2–8 Hz, then keeps SPECFEM bins strictly inside that Qseis interpolation
range.

Complex Qseis Green functions are interpolated onto the SPECFEM frequencies by
interpolating real and imaginary parts separately:

```python
G_q_common = interp(real(G_q)) + i interp(imag(G_q)).
```

This preserves complex information better than interpolating magnitude alone,
although ordinary Cartesian interpolation can still be inaccurate if phase
wraps rapidly between Qseis bins.

## Frequency-domain comparison products

For vertical and radial components, the notebook plots:

- `|G_SPECFEM(f)|`;
- `|G_Qseis(f)|`; and
- the amplitude ratio `|G_Qseis| / |G_SPECFEM|`.

The Green-function units are labelled `m/N`. Perfect amplitude agreement would
produce a ratio of one.

This figure tests amplitudes. The complex interpolation and later time-domain
reconstruction also retain phase information, but no separate phase-difference
plot or complex error metric is currently produced.

## Hard-band-limited waveform reconstruction

The first reconstruction uses the SPECFEM FFT grid. Empty spectra are created,
and only the common 2–8 Hz GF bins are inserted. An inverse real FFT then
produces vertical and radial band-limited impulse responses for both solvers.

This is equivalent to applying a sharp rectangular frequency window. Such a
window can introduce ringing in time. After source deconvolution, the notebook
uses a zero-based `tGF` axis and compares waveform arrival time, polarity,
phase, and amplitude.

## Stabilized common-grid comparison

The final cell provides a more controlled waveform comparison.

### Common time interval

The common interval uses SPECFEM's finer saved-trace step:

```text
dtCommon = dtSpec approximately 0.01 s
tMinCommon = min(tSpec[0], tQ[0])
tMaxCommon = min(tSpec[-1], tQ[-1])
```

For the inspected data, this is approximately −0.24 to 9.75 s. SPECFEM and
Qseis radial/vertical traces are interpolated onto this grid. Qseis is filled
with zero at negative time and is truncated to the shorter SPECFEM duration.

Each source-time function is also evaluated on this common grid. The duration
of the Qseis pulse remains based on the original Qseis `dt`, not `dtCommon`.

### Common zero padding

Both solvers use the same FFT length:

```python
nfft = 2**ceil(log2(2 * nCommon))
```

This supplies identical frequency bins and reduces circular wraparound during
the reconstructed time interval.

### Stabilized deconvolution

Instead of direct division, the final method uses:

```text
G = U S* / (|S|² + epsilon)
```

with solver-specific stabilization:

```text
epsilon = 10^-8 max(|S|²).
```

This behaves like inverse filtering where the source spectrum is strong while
limiting amplification near spectral zeros. Because each solver has a
different STF, each receives its own denominator and epsilon.

### Smooth zero-phase bandpass

The notebook constructs the one-pass Butterworth response with `sosfreqz` and
uses:

```python
B = abs(H)**2
```

`|H|²` is the amplitude response corresponding to forward-and-reverse
`sosfiltfilt` filtering. Applying the same `B` to both deconvolved GFs produces
a smooth, zero-phase 2–8 Hz band limitation without the sharp edges of the
earlier rectangular window.

The inverse FFT gives the final vertical and radial waveform overlays.

## What should agree

For a successful validation, Qseis and SPECFEM should show consistent:

- P- and S-wave arrival times;
- surface-wave arrival and dispersion;
- vertical and radial polarity after the z sign conversion;
- waveform phase after source removal;
- band-limited waveform shape;
- Green-function amplitude in `m/N`; and
- Qseis/SPECFEM amplitude ratios close to one in the reliable band.

Small differences can result from finite SPECFEM boundaries and mesh,
attenuation conventions, Qseis wavenumber integration, trace interpolation,
source representation, the unequal record lengths, and the Qseis late-time
window.

## Running the notebook

1. Update the Qseis configuration, validation-output, SPECFEM output, and
   station-file paths.
2. Confirm that the homogeneous material properties agree in both solvers.
3. Confirm the 1 N downward source and all receiver coordinates.
4. Run the dedicated Qseis-generation cell and require a zero return code.
5. Verify that the generated input and GF headers contain eight receivers at
   the requested 0.5–4.0 km offsets.
6. Confirm that `S007` corresponds to the 4 km target distance.
7. Inspect raw and filtered Qseis traces.
8. Inspect raw and filtered SPECFEM traces.
9. Check both source spectra throughout 2–8 Hz.
10. Compare the deconvolved frequency-domain amplitudes and ratios.
11. Compare the hard-band-limited time-domain GFs.
12. Compare the stabilized common-grid Butterworth waveforms.
13. Save figures and numerical comparison metrics if the validation must be
    retained.

## Recommended additional diagnostics

The notebook currently relies mainly on visual comparison. A more quantitative
validation could also save:

- maximum and RMS relative amplitude error across frequency;
- unwrapped phase difference;
- complex normalized residual;
- waveform correlation coefficient;
- peak-arrival time difference;
- waveform L2 misfit over selected arrival windows; and
- results at all eight receiver offsets rather than only S007.

These metrics should exclude frequencies where either source spectrum or
reference GF amplitude is too small for a stable relative comparison.

## Reproducibility files

Retain:

- `compareQseisSpecfemSeis.ipynb` or its converted script;
- the pySeisNN configuration and exact Qseis template selected by it;
- the Qseis executable revision;
- the dedicated generated Qseis input and component outputs;
- Qseis stdout, stderr, and return code;
- SPECFEM `Par_file`, `FORCESOLUTION`, `STATIONS`, mesh, and material files;
- `DB.S007.HXX.semd`, `DB.S007.HXZ.semd`, and
  `plot_source_time_function.txt`;
- all comparison figures and derived metrics; and
- Python environment and software revisions.

## Current limitations

- Paths, station name, target geometry, and plot limits are hard-coded.
- The configured validation directory is reusable, so input and GF headers
  should still be checked before extracting a receiver column.
- The notebook contains an unused `QSEIS_BASE` constant that still points to an
  older validation directory; executed loading uses the corrected `runDir`.
- The active configuration still points to a template under the local Qseis
  checkout rather than the bundled
  `pySeisNN1DFull/qseisTemplates/qs6inpNTempN.dat` path.
- Only one of the eight SPECFEM stations is compared after generation.
- The first frequency-domain deconvolution is unregularized.
- No explicit minimum-source-spectrum threshold is enforced.
- The source-function plot comments mention normalization, but the plotted
  arrays are not normalized in code.
- SPECFEM traces are not tapered while Qseis traces receive a late-time window.
- The common-grid comparison truncates Qseis to the shorter SPECFEM record.
- Complex interpolation is Cartesian rather than phase-aware.
- The notebook does not automatically save comparison arrays, figures, or
  scalar error metrics.
