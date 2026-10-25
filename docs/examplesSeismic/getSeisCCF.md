# Frequency-domain seismic cross-correlation

This chapter documents `getSeisCCF.ipynb`. Unlike the notebooks under the
Newtonian-noise examples, this notebook studies only the stochastic seismic
displacement field. It estimates the normalized frequency-domain
cross-correlation between two receiver points and compares the simulation with
analytic or Bessel-function reference curves.

The same driver supports:

- stochastic plane P and S waves in a homogeneous medium;
- stochastic plane Rayleigh waves in a homogeneous half-space;
- a Qseis force-source field in a homogeneous half-space; and
- a Qseis force-source field in a layered half-space.

For the Qseis cases, sources can be distributed internally within the domain
or on a distant ring. These are synthetic force-driven seismic fields, not
recorded site data.

## Documentation category

This page is stored under `docs/examplesSeismic/` because the calculation does
not evaluate Newtonian noise. The existing `docs/examples/` directory is left
unchanged for now. If it is later renamed to `docs/examplesNN/`, all internal
links should be updated in one operation.

## Quantity calculated

The two receiver points are:

```python
zTar = 0.0
gridX = np.array((0.0, 450.0))
gridY = np.array((0.0, 0.0))
```

Thus both receivers lie on the free surface and are separated by 450 m along
the x axis:

```text
A = (0, 0, 0) m
B = (450, 0, 0) m
d_AB = 450 m
```

For each frequency and component `i`, the notebook accumulates the two
auto-spectra and their complex cross-spectrum over all realizations:

```text
S_AA,i = sum_r |u_A,i^(r)|^2
S_BB,i = sum_r |u_B,i^(r)|^2
S_AB,i = sum_r u_A,i^(r) conjugate(u_B,i^(r))
```

It then plots the real part of the normalized complex correlation:

```text
Re[C_AB,i] = Re[S_AB,i / sqrt(S_AA,i S_BB,i)]
```

The calculation is performed independently for z, x, and y displacement. A
separate division by `n_rea` is unnecessary because the common realization
factor cancels between numerator and denominator.

Only the real part is plotted. The imaginary part, magnitude, and phase of the
complex correlation are not currently retained or displayed.

## Selecting a configuration

The beginning of `main()` contains six alternative configuration paths. One
must be active and the others commented:

| Configuration | Wavefield | Medium | Source positions |
| --- | --- | --- | --- |
| `configParseCCRayHomo.ini` | Analytic Rayleigh | Homogeneous | Random propagation azimuths |
| `configParseCCBodyHomo.ini` | Analytic P and S | Homogeneous | Random wave directions |
| `configParseCCRingHomo.ini` | Qseis full field | Homogeneous | Ring |
| `configParseCCInternalHomo.ini` | Qseis full field | Homogeneous | Internal |
| `configParseCCRingLayer.ini` | Qseis full field | Layered | Ring |
| `configParseCCInternalLayer.ini` | Qseis full field | Layered | Internal |

The active notebook setting is currently the analytic homogeneous Rayleigh
case.

General configuration parameters are documented in
[`configuration_complete.md`](../configuration_complete.md), while the flag
combinations are summarized in
[`simulation_modes.md`](../simulation_modes.md).

## Shared frequency and ensemble settings

All supplied configurations use:

```ini
fmin = 2.0
fmax = 7.9
df = 0.1
t_max = 40
n_samp = 2048
n_rea = 1000
n_src = 100
```

Every realization therefore combines 100 stochastic sources or waves, and the
correlation estimate is accumulated over 1000 realizations. This is a
statistical convergence choice; users should check whether fewer or more
realizations are sufficient for their required accuracy.

Random seeds and individual source metadata are not saved by this notebook, so
the exact ensemble cannot currently be reproduced.

## Homogeneous model

The analytic and homogeneous Qseis configurations use:

```ini
vp = 4000, 4000
vs = 2000, 2000
rho = 2465, 2465
thickness = 10000, 0
```

Identical properties above the zero-thickness bottom half-space make the solid
elastically homogeneous.

## Layered model

The layered Qseis configurations contain six entries:

| Layer thickness | `vp` | `vs` | Density |
| ---: | ---: | ---: | ---: |
| 30 m | 1000 m/s | 500 m/s | 1743 kg/m³ |
| 90 m | 3000 m/s | 1500 m/s | 2294 kg/m³ |
| 180 m | 4500 m/s | 2500 m/s | 2539 kg/m³ |
| 700 m | 5000 m/s | 3000 m/s | 2606.8 kg/m³ |
| 4000 m | 5800 m/s | 3360 m/s | 2720 kg/m³ |
| Half-space | 6000 m/s | 3750 m/s | 2800 kg/m³ |

The depths of the physical interfaces are therefore 30, 120, 300, 1000, and
5000 m.

## Analytic body-wave case

The body-wave configuration selects:

```ini
if_body = True
if_ray = False
if_full_field = False
```

For every realization, `genBodySrc` creates random P- and S-wave directions,
amplitudes, phases, and S-wave polarizations. The notebook requests fractions

```python
eP = 1/3
eS = 2/3
```

and uses `pM = 2` in the theoretical expressions. With receiver separation
along x, the reference correlations are built from spherical Bessel functions
`j0` and `j2` evaluated at the P- and S-wave wavenumbers.

This mode does not execute Qseis. It uses
`computeFullBodyDisp_parallel` directly at the two receivers.

## Analytic Rayleigh-wave case

The Rayleigh configuration selects:

```ini
if_body = False
if_ray = True
if_full_field = False
```

`pysurf96` supplies the fundamental Rayleigh phase velocity at every selected
frequency. Random source azimuths and phases create a stochastic plane-wave
field, which `computeRayleighDisp` evaluates at the two surface receivers.

For separation along x, the theoretical curves are:

```text
C_zz = J0(k_R d)
C_xx = J0(k_R d) - J2(k_R d)
C_yy = J0(k_R d) + J2(k_R d)
```

where `k_R = 2 pi f / v_R(f)` and `d = 450 m`.

This is the simplest case for verifying convergence of the Monte Carlo
correlation toward the expected Bessel-function form.

## Qseis full-field cases

The four force-source configurations select:

```ini
if_body = False
if_ray = False
if_full_field = True
on_fly_nn = False
```

`on_fly_nn` must remain false because this notebook requires displacement at
the two receivers; it is not performing an NN reduction.

For full-field mode, the driver:

1. calls `createFinalInputsTD(config, zList)` for the required receiver depth;
2. runs every Qseis job in parallel;
3. assembles the selected frequency-domain GF components with
   `onlySplitAll=False`;
4. generates a new stochastic force distribution for each realization;
5. evaluates the displacement at A and B with `computeFullDispF`; and
6. cleans the temporary shared-memory files after accumulating the spectra.

Only surface GFs are required here because `zTar = 0`.

The assembled components are:

```text
fh-2.tz  fh-2.tr  fh-2.tt  fz-2.tz  fz-2.tr
```

They contain the vertical, radial, and transverse responses required for
horizontal and vertical surface forces.

## Internal and ring source distributions

For `src_distri = internal`, source x and y coordinates are drawn uniformly
within the configured simulation domain. `r_1` and `r_2` are unused.

For `src_distri = ring`, source radii lie between:

```text
r_1 = 6000 m
r_2 = 6500 m
```

The ring cases consequently require `xmax_gf = 13000 m`, compared with 6000 m
for the homogeneous internal case and 9000 m for the layered internal case.
GF coverage must include the largest separation between any ring source and
either receiver.

Horizontal and vertical force amplitudes and phases are independent because
`decoupled_hv = True`; `scale_vh = 1` applies no vertical/horizontal amplitude
bias.

## Full-field reference curves

For Qseis cases, the notebook calculates fundamental-mode Rayleigh Bessel
functions using the configured model and plots:

```text
C_zz,ref = J0
C_xx,ref = (J0 - J2 + J0) / 2
C_yy,ref = (J0 + J2 + J0) / 2
```

These expressions assume equal vertical and horizontal weighting
(`vR = hR = 1`) and include only the Rayleigh Bessel terms. Love-wave terms are
disabled with `love=False`. The code also calculates `J4`, but it is not used
in the displayed expressions.

For a general Qseis field containing body waves, surface waves, near-field
terms, attenuation, and a finite source distribution, these red curves should
be treated as reference shapes rather than exact predictions.

## Plot produced

`plotCCDeepSurfMulti` produces three panels in this order:

1. z-component correlation;
2. x-component correlation; and
3. y-component correlation.

The simulated real correlation is shown in blue and the theoretical/reference
curve is overlaid in red or magenta. Because the normalized correlation is
dimensionless, a well-converged result should remain between -1 and 1 apart
from small numerical errors.

The notebook currently does not call `savefig` or save the numerical
correlation arrays. To retain a result, save `freqOut`, `ccRealX`, `ccRealY`,
and `ccRealZ` explicitly or export the figure before running another case.

## Shared tutorial output directory

All six configurations, as well as the other stochastic-displacement tutorial
notebooks, use the common root:

```text
/home/koley/CavityResults/AllTutOut/DispTests/
```

The shared subdirectories are:

```text
QseisInpN/
OutDisp/
OutDispRea/
```

These are disposable tutorial outputs. They are intentionally rewritten
rather than separated by notebook or physical case. In particular,
`qseis_out_del = yes` allows input preparation to delete existing Qseis run
directories before creating the database for the newly selected configuration.

Do not expect a homogeneous database to remain after running a layered case,
or a ring-source database to remain after a case with different GF coverage.
Copy any result or database that must be retained to a separate location before
starting another tutorial.

Even analytic body- and Rayleigh-wave branches call `createFinalInputs` to
produce geometry metadata. Because the same `input_folder` is used, users
should treat the entire `DispTests` tree as temporary while switching among
these examples.

## Test-mass argument in full-field mode

The full-field displacement interface shares its signature with routines that
can also reduce a field to NN. The notebook therefore prepares the configured
test-mass coordinate even though this example requests only receiver
displacement:

```python
itmAC = config.itmAC
itmACUse = itmAC.copy()
itmACUse[2] = -np.abs(itmACUse[2])
itmAll = np.vstack((itmACUse,))
```

`itmAll` is passed to `computeFullDispF`. The conversion follows the internal
Cartesian convention in which underground z is negative. With `reduceNN=False`
and `nnMode=None`, no Newtonian-noise calculation is performed.

## Running the example

1. Select exactly one of the six configuration paths.
2. Confirm that the corresponding wavefield flags are correct.
3. Verify the two receiver coordinates and their 450 m separation.
4. Confirm `n_rea` and `n_src`; use smaller values for an initial test.
5. For full-field mode, keep `on_fly_nn = False` and verify the converted
   `itmAll` coordinate.
6. Confirm that overwriting `AllTutOut/DispTests/` is acceptable.
7. For ring sources, verify the larger GF distance coverage.
8. Run the notebook and, for Qseis cases, check every Qseis process status.
9. Inspect convergence of all three correlation components.
10. Save the numerical arrays or figure if the result must be retained.

## Interpretation and convergence checks

Check convergence with respect to:

- number of sources per realization;
- number of realizations;
- source azimuth and spatial distribution;
- frequency sampling;
- Qseis receiver-distance sampling and time window;
- GF horizontal coverage for ring sources; and
- random-seed variability.

For analytic cases, convergence toward the theoretical curves tests the
stochastic source sampling and displacement implementation. For Qseis cases,
departures from the simple Rayleigh reference can be physical and do not by
themselves indicate an error.

## Current limitations

- The notebook displays but does not save correlation arrays or figures.
- Only two surface points separated along x are considered.
- Only same-component correlations (`zz`, `xx`, and `yy`) are calculated;
  cross-component terms are not included.
- Only the real part of normalized complex correlation is plotted.
- Exact stochastic realizations are not reproducible because seeds and source
  metadata are not stored.
- The full-field theoretical overlays use simplified equal H/V Rayleigh
  weighting and omit Love waves.
- Every tutorial configuration writes to the same disposable output tree.
