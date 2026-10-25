# Configuration reference

The software uses a primary INI file to define the geological model,
frequency sampling, numerical grid, simulation geometry, seismic sources,
filesystem paths, wavefield mode, and parallel-computing settings. A separate
INI file describes one or more cavities.

The primary file is loaded by `modules/configLoader.py` and checked by
`modules/validateInputs.py`. Under Python's default `ConfigParser` behavior,
option names are case-insensitive, but section names must match those expected
by the loader. Unless stated otherwise, distances are in metres and depth is
positive downward.

> Requirements enforced by the code are identified below. Some physically
> necessary conditions are not yet checked automatically.

## `[model]`: one-dimensional geological model

```ini
[model]
thickness = 1000.0, 2000.0, 0.0
vp = 3500.0, 4500.0, 5500.0
vs = 1800.0, 2500.0, 3200.0
rho = 2300.0, 2500.0, 2700.0
qp = 500.0, 700.0, 900.0
qs = 250.0, 350.0, 450.0
rhoAir = 1.2
```

This section describes homogeneous horizontal layers ordered from the surface
downward. Entries at the same array index belong to the same layer.

| Parameter | Type | Unit | Meaning |
| --- | --- | --- | --- |
| `thickness` | Float array | m | Thickness of each layer. The last value must be `0.0`, denoting the bottom half-space. |
| `vp` | Float array | m/s | P-wave velocity in each layer. |
| `vs` | Float array | m/s | S-wave velocity in each layer. |
| `rho` | Float array | kg/m³ | Mass density of each layer. |
| `qp` | Float array | Dimensionless | P-wave attenuation quality factor in each layer. |
| `qs` | Float array | Dimensionless | S-wave attenuation quality factor in each layer. |
| `rhoAir` | Float | kg/m³ | Air density used for the top truncation-surface contribution at the medium–air boundary. |

The final `0.0` thickness does not describe a zero-thickness layer; it marks
the semi-infinite half-space. All material-property arrays must contain one
value per layer, including the half-space.

The validator currently checks that `vp`, `vs`, and `rho` have the same length
as `thickness`, that the final thickness is `0.0`, that `vp > vs` in every
layer, and that density is positive. Positive velocities and quality factors,
and length checks for `qp` and `qs`, are physically required but are not yet
validated explicitly.

## `[frequency]`: analysis band

```ini
[frequency]
fmin = 2.0
fmax = 8.0
df = 0.1
```

Newtonian noise is calculated in the frequency domain.

| Parameter | Type | Unit | Meaning |
| --- | --- | --- | --- |
| `fmin` | Float | Hz | Minimum analysis frequency. |
| `fmax` | Float | Hz | Maximum analysis frequency. |
| `df` | Float | Hz | Requested interval between analysis frequencies. |

The example requests `2.0, 2.1, 2.2, ... , 8.0 Hz`.

### Relation to the FFT grid

Output frequencies must coincide with the native FFT bins defined by `t_max`
and `n_samp`. The native interval is

```text
df_native = 1 / t_max.
```

`simDisp.getFreqGrid` selects native bins in the closed interval from `fmin`
to `fmax`, then subsamples them using

```text
step = max(1, round(df / df_native)).
```

The actual interval is `step * df_native`, so it can differ from `df`. An
endpoint is included only when it occurs on a selected FFT bin. For predictable
sampling, choose `df` as an integer multiple of `df_native`.

The validator requires positive `fmin` and `fmax`, with `fmin < fmax`. It does
not currently check that `df` is positive or compatible with the FFT grid.

## `[grid]`: wavelength-based spatial resolution

```ini
[grid]
lambda_frac = 0.3333
lambda_res = 6
```

| Parameter | Type | Unit | Meaning |
| --- | --- | --- | --- |
| `lambda_frac` | Float | Dimensionless | Fraction of a surface-wave wavelength used to estimate its depth sensitivity. |
| `lambda_res` | Integer | Points per wavelength | Number of spatial points used to resolve the minimum relevant wavelength in a layer. |

If a Rayleigh wave travels at `1000 m/s` at `2 Hz`, its wavelength is `500 m`.
With `lambda_frac = 1/3`, the wave is treated as sensitive to approximately
`500 / 3 = 166.7 m` depth. This wavelength-to-depth mapping determines which
surface-wave wavelengths are relevant to each layer.

The implementation calculates fundamental-mode Rayleigh- and Love-wave phase
velocities with `pysurf96`, uses the smaller surface-wave velocity, and assigns
it to layers according to `wavelength * lambda_frac`. For a homogeneous model,
the Rayleigh velocity is used as the Love-wave fallback.

For each layer, the code compares the assigned surface-wave wavelength with
the S-wave wavelength at `fmax` and calculates

```text
grid_size = minimum_relevant_wavelength / lambda_res.
```

Because the model requires `vp > vs`, the P-wave wavelength cannot reduce the
spacing further. Increasing `lambda_res` produces a finer grid but increases
memory use and computation time. Values from 4 to 6 are typical; the validator
requires `lambda_res >= 4`. `lambda_frac` is intended to lie between zero and
one, but this range is not currently validated.

## `[geometry]`: Green's-function and simulation domains

```ini
[geometry]
xmax_gf = 6000.0
zmax_gf = 5000.0
max_receivers = 100
x_center = 0.0
y_center = 0.0
x_extent = 2000.0
y_extent = 2000.0
z_extent = 4000.0
t_max = 40.96
n_samp = 2048
cavity_type = sphere
```

| Parameter | Type | Unit | Meaning |
| --- | --- | --- | --- |
| `xmax_gf` | Float | m | Maximum horizontal source-receiver distance covered by the Qseis GF database. |
| `zmax_gf` | Float | m | Maximum receiver depth covered by the GF database. At each depth, GFs extend horizontally to `xmax_gf`. |
| `max_receivers` | Integer | Receivers/job | Maximum receiver offsets in one Qseis job; currently at most 100. |
| `x_center` | Float | m | x coordinate of the simulation-domain centre. |
| `y_center` | Float | m | y coordinate of the simulation-domain centre. |
| `x_extent` | Float | m | Positive half-width of the simulation domain in x. |
| `y_extent` | Float | m | Positive half-width of the simulation domain in y. |
| `z_extent` | Float | m | Positive maximum depth measured from the surface. |
| `t_max` | Float | s | Duration of each Qseis GF time series. |
| `n_samp` | Integer | Samples | Number of samples in each Qseis time series. |
| `cavity_type` | String | — | Legacy single-cavity selector: `cuboid` or `sphere`. |

### Simulation bounds

The bounds are

```text
x_center - x_extent <= x <= x_center + x_extent
y_center - y_extent <= y <= y_center + y_extent
0 <= depth <= z_extent.
```

All extents must be positive, although this is not currently checked by the
validator.

### Choosing GF coverage

`xmax_gf` and `zmax_gf` must cover every source-receiver separation required by
the source configuration and simulation domain. A domain spanning `-2 km` to
`+2 km` in both x and y has a corner-to-corner diagonal of approximately
`sqrt(2) * 4 km = 5.66 km`. If this is the largest required separation,
`xmax_gf = 6000 m` provides a small margin. Sources outside the simulation
domain must also be considered.

`zmax_gf` must be at least `z_extent`. For `z_extent = 4000 m`, a value such as
`zmax_gf = 5000 m` provides extra depth coverage.

### Splitting Qseis jobs

If 1,000 receiver offsets are required and `max_receivers = 100`, the software
creates 10 Qseis jobs for that receiver depth. Reducing the value to 50 would
produce 20 jobs. The validator checks only that this value is positive; the
upper limit is imposed by the current Qseis build.

### Time sampling

`t_max` must be long enough for the slowest relevant arrival to cross the
largest source-receiver distance. It therefore depends on the lowest group
velocity in the layered model and the GF extent. About 40 seconds is often safe
for the models studied so far, but the user must verify that the GF is not
truncated.

```text
dt = t_max / n_samp
sampling_frequency = n_samp / t_max
```

For example, `t_max = 40.96 s` and `n_samp = 2048` give `dt = 0.02 s`, a
sampling frequency of `50 Hz`, a Nyquist frequency of `25 Hz`, and an FFT
spacing of approximately `0.0244 Hz`.

Qseis currently supports `n_samp` values 1024, 2048, 4096, and 8192, with 8192
as the present maximum. The allowed values and the Nyquist condition are not
yet checked by the Python validator.

`cavity_type` is retained from an older version. Current multi-cavity drivers
read the separate cavity configuration described later in this guide.

## `[tmPosition]`: Einstein Telescope test masses

```ini
[tmPosition]
itm_ac = x, y, z
itm_ab = x, y, z
etm_ca = x, y, z
etm_ba = x, y, z
```

Each entry is an `(x, y, z)` coordinate in metres. `ITM` means input test mass,
and `ETM` means end test mass. Einstein Telescope contains three
interferometers associated with vertices A, B, and C. The four entries above
identify the test masses near vertex A.

| Parameter | Meaning |
| --- | --- |
| `itm_ac` | Input test mass at A for the arm directed from A toward C. |
| `itm_ab` | Input test mass at A for the arm directed from A toward B. |
| `etm_ca` | End test mass for the interferometer at C on the arm from C toward A. |
| `etm_ba` | End test mass for the interferometer at B on the arm from B toward A. |

The configured z coordinate is positive physical depth. Drivers convert it to
the internal Cartesian convention, in which underground z coordinates are
negative. A negative configured value can represent a point above the surface.

All four entries are mandatory in the current loader. For a single-test-mass
calculation, users may repeat the same position four times or select one test
mass in the driver and ignore the other three.

## `[simSource]`: stochastic seismic sources

```ini
[simSource]
n_rea = 100
r_1 = 2000.0
r_2 = 2500.0
n_src = 100
src_distri = ring
scale_vh = 1.0
```

| Parameter | Type | Unit | Meaning |
| --- | --- | --- | --- |
| `n_rea` | Integer | Realizations | Number of independent stochastic realizations. |
| `r_1` | Float | m | Inner radius used by `ring` and `inDisk`. |
| `r_2` | Float | m | Outer radius used by `ring` and `inDisk`. |
| `n_src` | Integer | Sources/realization | Number of simultaneous sources in each realization. |
| `src_distri` | String | — | Source-position model: `ring`, `internal`, or `inDisk`. |
| `scale_vh` | Float | Amplitude ratio | Multiplier applied to vertical-force amplitudes relative to horizontal-force amplitudes. |

One realization combines many simultaneously acting sources. Source
positions, orientations, amplitudes, and phases are redrawn for every
realization. Typical studies use 100–500 realizations or more.

- `ring` draws source azimuth and radius, with radius uniform between `r_1` and
  `r_2`.
- `internal` draws x and y uniformly within the simulation domain; `r_1` and
  `r_2` are unused.
- `inDisk` distributes sources uniformly by area in the annulus between `r_1`
  and `r_2`.

For Qseis force sources, amplitudes are drawn uniformly between zero and one,
and phases are drawn uniformly between zero and `2*pi`, independently at every
frequency. Analytic body-wave mode draws random P- and S-wave directions,
amplitudes, phases, and S-wave polarizations.

`scale_vh = 1` applies no vertical/horizontal bias. Source amplitudes are not
normalized when `n_src` changes because post-processing scales the ensemble
RMS to a selected surface or underground displacement ASD component.

## `[paths]`: input, output, and reference data

```ini
[paths]
template_file = /path/to/pySeisNN1DFull/qseisTemplates/qs6inpNTempN.dat
input_folder = /path/to/QseisInput/
qseis_exe = /path/to/qseis_binary
out_disp_folder = /path/to/temporary_displacement/
out_disp_rea = /path/to/realization_results/
qseis_out_del = yes
site_asd_path = /path/to/site_asd/
et_d_path = /path/to/et_design/
```

| Parameter | Meaning |
| --- | --- |
| `template_file` | Path to the bundled pySeisNN master template, normally `qseisTemplates/qs6inpNTempN.dat`. The software substitutes run-specific model, receiver, and time-sampling values to create individual Qseis input files. |
| `input_folder` | Read/write directory for generated Qseis `.dat` files, per-run output directories, `splitAll.mat`, and the assembled `splitAllNew.npy` GF database. |
| `qseis_exe` | Path to the compiled Qseis executable. |
| `out_disp_folder` | Working directory for temporary depth-specific displacement metadata and, in legacy paths, displacement arrays. |
| `out_disp_rea` | Persistent output directory for source metadata and realization-level displacement and NN products used by post-processing. |
| `qseis_out_del` | Set exactly to `yes` to delete existing Qseis run subdirectories before regenerating the database. |
| `site_asd_path` | Directory containing candidate-site displacement ASD and attenuation files. |
| `et_d_path` | Directory containing the Einstein Telescope ET-D design-sensitivity file. |

The formatted template is distributed with pySeisNN and does not need to be
copied into the Qseis source tree. Do not modify it directly for individual
runs; the database generator creates new input files from it.

The path validator requires the template and executable to exist, the Qseis
executable to be executable, and all configured directories to exist before a
run begins.

### Qseis cleanup warning

When input generation starts, existing top-level `.dat` files in
`input_folder` are always deleted. When `qseis_out_del` is exactly `yes`, all
existing subdirectories in `input_folder` are also recursively deleted. Any
other value preserves the subdirectories but not the top-level `.dat` files.
Use a directory dedicated to this Qseis database.

### Expected reference files

`site_asd_path` is expected to contain:

- `asdSurf.mat` — surface displacement ASDs;
- `asdBH.mat` — underground/borehole displacement ASDs; and
- `attnModel.mat` — observed surface-to-depth attenuation data.

`et_d_path` is expected to contain `ETD.mat`.

## `[simFlags]`: wavefield and storage modes

```ini
[simFlags]
if_body = False
if_ray = False
if_full_field = True
save_hv = False
decoupled_hv = True
on_fly_nn = True
```

| Parameter | Type | Meaning |
| --- | --- | --- |
| `if_body` | Boolean | Use stochastic plane P and S waves in a homogeneous medium. This mode is not valid for a layered medium. |
| `if_ray` | Boolean | Use the analytic Rayleigh-wave model in a homogeneous medium. This mode is not valid for a layered medium. |
| `if_full_field` | Boolean | Simulate the wavefield from Qseis GFs. This supports homogeneous and layered media. |
| `save_hv` | Boolean | Retain separate horizontal-force, vertical-force, and total displacement arrays. |
| `decoupled_hv` | Boolean | Draw independent horizontal- and vertical-force amplitudes and phases. |
| `on_fly_nn` | Boolean | Reduce displacement chunks directly to NN contributions without storing the full seismic field. |

Exactly one wavefield mode should be selected. `if_full_field` should be true
when both analytic flags are false. The validator prevents either analytic flag
from being combined with another mode, but it does not currently enforce that
at least one flag is true.

`decoupled_hv = True` is the normal stochastic setting. If false, horizontal
and vertical forces share the same amplitudes and phases before `scale_vh` is
applied.

### Displacement storage

With `on_fly_nn = True`, each worker generates a displacement chunk and
immediately reduces it to a partial NN integral. The full field is not placed
in `/dev/shm` or written to disk. This is the recommended production mode
because it reduces memory use and I/O.

With `on_fly_nn = False`, the complete complex displacement field is stored as
a memory-mapped file in `/dev/shm`, consumed by the NN integration, and then
intended to be removed using the returned cleanup paths.

When `save_hv = False`, only the total field is retained. When it is true,
separate `dispH`, `dispV`, and `dispTotal` arrays are created in `/dev/shm` for
diagnostic use. `save_hv = True` is incompatible with on-the-fly NN reduction.
The normal production combination is therefore:

```ini
save_hv = False
on_fly_nn = True
```

## `[compute]`: parallel execution

```ini
[compute]
compute_cores_qseis = 32
compute_cores_disp = 32
compute_cores_nn = 32
compute_strategy = loky_sharedmem
```

| Parameter | Type | Meaning |
| --- | --- | --- |
| `compute_cores_qseis` | Integer | Number of parallel workers used for independent Qseis database jobs. This stage generally scales well. |
| `compute_cores_disp` | Integer | Maximum workers used to partition receiver points during displacement generation. Values of 32 or 64 are commonly used, but scaling is limited. |
| `compute_cores_nn` | Integer | Workers used for receiver-grid NN integration when it is performed separately. Scaling is similar to displacement generation. |
| `compute_strategy` | String | Displacement parallelization strategy. The only currently supported configured value is `loky_sharedmem`. |

The displacement code can reduce the effective worker count when there are too
few receiver points to assign useful chunks. CPU count alone is not sufficient
for sizing a run: memory capacity, `/dev/shm` capacity in stored-field mode,
receiver count, source count, and frequency count must also be considered.

The exact accepted strategy spelling is `loky_sharedmem`, without an underscore
between `shared` and `mem`. With `on_fly_nn = True`, the reduction branch uses
worker-local arrays and returns only partial NN integrals even though the driver
continues to pass this configured strategy value.

## Cavity configuration file

A separate INI file defines one or more cavities. Each section is one cavity;
section names such as `[CavityA]` are arbitrary identifiers.

### Spherical cavity

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

| Parameter | Unit | Meaning |
| --- | --- | --- |
| `shape` | — | Must be `sphere`. |
| `xC`, `yC`, `zC` | m | Cavity-centre coordinates in the simulation coordinate system. `zC` is positive depth. |
| `radius` | m | Sphere radius. |
| `xRefineFactor`, `yRefineFactor`, `zRefineFactor` | Dimensionless | Multipliers applied to the radius to obtain the refinement-box half-width along each axis. |

For example, a `20 m` radius and factors of 6 produce half-widths of `120 m`
and hence a `240 × 240 × 240 m` refinement box centred on the cavity.

### Cuboidal cavity

```ini
[CavityB]
shape = cuboid
xC = 50.0
yC = 0.0
zC = 250.0
length = 160.0
breadth = 20.0
height = 20.0
angleDeg = 30.0
xRefineFactor = 4.0
yRefineFactor = 4.0
zRefineFactor = 4.0
```

| Parameter | Unit | Meaning |
| --- | --- | --- |
| `shape` | — | Must be `cuboid`. |
| `xC`, `yC`, `zC` | m | Cuboid-centre coordinates; `zC` is positive depth. |
| `length` | m | Full dimension along the cuboid's local major horizontal axis. |
| `breadth` | m | Full dimension along its local minor horizontal axis. |
| `height` | m | Full vertical dimension. |
| `angleDeg` | degrees | Anticlockwise rotation about the z axis. |
| `xRefineFactor`, `yRefineFactor`, `zRefineFactor` | Dimensionless | Required legacy refinement fields; currently overridden by the fixed-buffer implementation for cuboids. |

The code first calculates the smallest axis-aligned bounding box containing the
rotated cuboid, then extends that box by a fixed `150 m` in x, y, and z. This
fixed buffer is the current tested behavior. Although the three refinement
factors remain mandatory in the cavity parser, changing them currently does not
change the cuboid refinement extent.

### Multiple and nearby cavities

Every cavity must define its own refinement fields. Nearby or overlapping
refinement regions are merged internally, and integration-point masking avoids
double-counting geological material excluded by multiple cavities. A practical
target is at least roughly `100 m` of finely sampled material around cavity
boundaries; the current cuboid implementation provides `150 m`.

Physically intersecting cavity surfaces should be treated with caution and
validated separately; merging refinement boxes does not by itself establish
that every overlapping-cavity surface integral is physically meaningful.

## Validation checklist

Before a production run, verify that:

- all layer arrays have equal length and end with a half-space entry;
- `vp > vs`, densities and quality factors are positive, and units are SI;
- the requested frequencies lie below the Nyquist frequency;
- `df` is compatible with `1 / t_max`;
- `t_max` contains the latest required GF arrival;
- `xmax_gf` covers the largest source-receiver separation;
- `zmax_gf >= z_extent`;
- the Qseis template, executable, and all directories exist;
- `input_folder` contains no unrelated directories if `qseis_out_del = yes`;
- only one wavefield mode is selected and analytic modes use a homogeneous
  model;
- cavity and test-mass depths use the positive-depth configuration convention;
- the integration grid is converged with respect to `lambda_res` and cavity
  refinement; and
- worker counts fit the available CPUs, RAM, and `/dev/shm` capacity.
