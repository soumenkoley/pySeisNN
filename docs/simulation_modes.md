# Simulation modes and storage flags

The `[simFlags]` section selects how displacement is generated and how it is
passed to the Newtonian-noise integration.

## Wavefield modes

Exactly one of the following should be true:

| `if_body` | `if_ray` | `if_full_field` | Meaning |
| ---: | ---: | ---: | --- |
| True | False | False | Analytic homogeneous P- and S-wave field. |
| False | True | False | Analytic homogeneous Rayleigh-wave field. |
| False | False | True | Qseis GF field for a homogeneous or layered model. |

All other combinations should be considered invalid. The current validator
rejects multiple enabled modes, but it does not reject the case in which all
three are false.

## Analytic body-wave mode

```ini
if_body = True
if_ray = False
if_full_field = False
```

This mode generates stochastic plane P and S waves in a homogeneous medium.
It does not model reflections, conversions, or dispersion caused by geological
layering. Do not use it with a layered model.

Qseis execution is unnecessary, but the driver may still call geometry and
assembly helpers with `onlySplitAll=True` to obtain depth-grid metadata.

## Analytic Rayleigh-wave mode

```ini
if_body = False
if_ray = True
if_full_field = False
```

This mode constructs analytic Rayleigh-wave displacement using the calculated
phase velocity. It is intended for a homogeneous half-space and should not be
used as the wavefield model for layered geology.

As in body-wave mode, no Qseis GF database is required.

## Qseis full-field mode

```ini
if_body = False
if_ray = False
if_full_field = True
```

This mode loads and interpolates assembled Qseis GFs. It includes the response
of the configured one-dimensional model and is the required production mode
for layered media. It is also usable for homogeneous models and point-force
validation examples.

The GF database must match the model, source depth, time grid, receiver
coverage, frequency bins, component selection, and Qseis template.

## `on_fly_nn`

`on_fly_nn = True` reduces each generated displacement chunk directly into a
partial NN integral. It avoids writing and rereading the full wavefield and is
the recommended setting for production runs.

`on_fly_nn = False` requests a stored displacement workflow. The full complex
field can be written to memory-mapped files and then consumed by a separate NN
step. This is useful for diagnostics but requires much more shared memory and
I/O.

## `save_hv`

`save_hv = True` retains separate responses to horizontal and vertical force
components in addition to the total field. It is intended for diagnostics.

`save_hv = False` retains only the total response and minimizes storage.

The normal production combination is:

```ini
save_hv = False
on_fly_nn = True
```

Separate H/V fields are incompatible with the streamlined on-the-fly reduction
path in the current implementation.

## `decoupled_hv`

This flag controls stochastic force metadata rather than the geological
wavefield mode.

- `decoupled_hv = True`: horizontal and vertical force amplitudes and phases
  are drawn independently.
- `decoupled_hv = False`: horizontal and vertical parts share amplitude and
  phase before the vertical/horizontal scaling factor is applied.

The usual stochastic setting is `True`. `scale_vh` then biases the vertical
amplitude relative to the horizontal amplitude; `scale_vh = 1` applies no
systematic bias.

## Database assembly mode

Do not confuse simulation flags with `onlySplitAll` in
`assembleQseisRuns`:

- analytic body/Rayleigh examples use `onlySplitAll=True` because they need
  geometry metadata but not Qseis GFs;
- full-field examples use `onlySplitAll=False` to read, FFT, and save the Qseis
  component files.

## Recommended configurations

### Fast analytic test

```ini
if_body = True       # or use if_ray instead
if_ray = False
if_full_field = False
save_hv = False
on_fly_nn = True
decoupled_hv = True
```

Use only with a homogeneous medium.

### Layered production calculation

```ini
if_body = False
if_ray = False
if_full_field = True
save_hv = False
on_fly_nn = True
decoupled_hv = True
```

### Diagnostic stored wavefield

```ini
if_body = False
if_ray = False
if_full_field = True
save_hv = True       # only if separate force responses are needed
on_fly_nn = False
```

Check RAM, disk, and `/dev/shm` capacity before using this mode.

## Pre-run checks

Before starting a run, verify:

- exactly one wavefield mode is enabled;
- analytic modes are used only with homogeneous media;
- full-field mode has a compatible assembled GF database;
- `save_hv` and `on_fly_nn` are a supported combination;
- stochastic H/V coupling matches the intended physical model; and
- worker count and storage mode fit available memory.

