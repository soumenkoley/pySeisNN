# Coordinates, components, and units

This page defines conventions shared by the configuration files, drivers,
Qseis interface, and saved results.

## Horizontal axes

The simulation uses Cartesian horizontal coordinates:

- x is the first horizontal component and is commonly interpreted as east;
- y is the second horizontal component and is commonly interpreted as north;
- z is vertical.

Arrays containing vector components use the order:

```text
[x, y, z]
```

Some post-processing functions label these components `[E, N, Z]`.

## Two vertical-coordinate conventions

Configuration files use positive depth:

```text
surface: z_config = 0
underground: z_config > 0
above surface: z_config < 0
```

Internal Cartesian test-mass and cavity calculations use z positive upward:

```text
z_internal = -abs(z_config)
```

Thus a configured test mass at `(0, 0, 250)` is normally represented
internally as `(0, 0, -250)`. Qseis receiver-depth and layer calculations,
however, continue to use non-negative depth measured downward. Always check
which convention a function expects before passing coordinates directly.

## Domain definition

The primary configuration defines:

```text
x_min = x_center - x_extent
x_max = x_center + x_extent
y_min = y_center - y_extent
y_max = y_center + y_extent
depth_min = 0
depth_max = z_extent
```

`x_extent` and `y_extent` are half-widths, not full widths. A value of 3000 m
with a centre of zero therefore produces a 6000 m-wide domain.

## Cavity coordinates

`xC`, `yC`, and `zC` specify the cavity centre in the same horizontal system as
the simulation domain. `zC` is entered as positive depth. For a cuboid,
`angleDeg` is an anticlockwise rotation about the z axis.

Refinement factors define a local finely sampled region for spheres. Cuboids
currently use their rotated bounding box plus a fixed 150 m buffer despite the
presence of refinement-factor fields in the file.

## Test-mass coordinates

Each test-mass entry is written as:

```ini
itm_ac = x, y, depth
```

The four standard names refer to the test masses near an Einstein Telescope
vertex. All must be present in the file, although a notebook can select a
subset for calculation.

The test mass should normally lie inside its associated cavity. The code maps
test masses to cavities during geometry preparation; verify this mapping for
multiple or nearby cavities.

## Source coordinates and direction

Stochastic full-field sources are positioned on the surface, so `xSrc` and
`ySrc` locate each force horizontally and the Qseis template supplies the
source depth. `azSrc` gives the horizontal-force azimuth.

Vertical-force validation examples use a downward force in the Qseis
depth-positive convention. This sign convention must be translated carefully
when comparing with solvers that use z positive upward.

## Frequency and Fourier quantities

Frequencies are in hertz, times in seconds, and angular phase values in
radians unless a field explicitly uses degrees. The requested frequency grid
is a subset of the real-FFT bins produced by `t_max` and `n_samp`.

NumPy's FFT is an unnormalized discrete sum. Where a continuous Fourier
amplitude is intended, the notebooks multiply the FFT-domain quantity by
`dt = t_max / n_samp`. Do not compare amplitudes between scripts unless their
FFT and source-wavelet normalization conventions agree.

## Standard SI units

| Quantity | Unit |
| --- | --- |
| Distance, depth, cavity dimensions | m |
| Time | s |
| Frequency | Hz |
| P- and S-wave velocity | m/s |
| Density | kg/m³ |
| Quality factors | dimensionless |
| Displacement | m |
| Force | N |
| Newtonian acceleration | m/s² |
| Acceleration response per unit force | m s⁻² N⁻¹ |

`qp`, `qs`, wavelength fractions, refinement factors, and amplitude ratios are
dimensionless.

## NN sign convention

The saved total is constructed as:

```text
ITot = IBlockTot - IVertFaceTot - IHorFaceTot
```

The surface arrays are computed with the orientation conventions of the
integration geometry and are subtracted from the bulk result. Preserve complex
values until after contributions have been combined. In general:

```text
|bulk - surface| != |bulk| - |surface|
```

## Comparison checklist

When comparing with another solver, match:

- horizontal and vertical axes;
- positive-depth versus positive-up conventions;
- source position, direction, amplitude, and source-time function;
- test-mass or gravity-station position;
- vector-component ordering;
- density-interface normals and signs;
- Fourier-transform normalization; and
- units before any plotting or interpolation.

