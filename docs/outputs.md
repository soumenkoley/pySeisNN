# Output files and array contents

The software writes three broad classes of data: Qseis run products, assembled
GF database files, and per-realization displacement/NN results. Exact files can
vary between notebooks and storage modes.

## Output directories

The primary configuration distinguishes:

- `input_folder`: Qseis inputs, run directories, and assembled GF database;
- `out_disp_folder`: temporary displacement work area; and
- `out_disp_rea`: persistent realization-level results used by post-processing.

Do not place unrelated data in `input_folder`: database preparation deletes
top-level `.dat` files and can recursively delete run subdirectories when
`qseis_out_del = yes`.

## Qseis job metadata

### `splitAll.mat`

Written during input generation. It describes the individual depth/distance
jobs before their outputs are joined. Drivers reload its `splitAll` array when
reusing prepared geometry.

### Qseis run directories

Directories named like `Layer<layer_id>Run<run_id>` contain the generated input
and time-domain GF component files for one receiver depth and distance subset.
Typical assembled components are:

```text
fh-2.tz  fh-2.tr  fh-2.tt  fz-2.tz  fz-2.tr
```

These are intermediate database-generation products but should be retained
when exact regeneration or diagnostics are required.

## Assembled GF database

### `splitAllNew.npy`

Contains one row per assembled receiver depth:

```text
[layer_id, dummy_run_id, depth, x_start, x_end, number_of_distances]
```

It maps target depths to the corresponding layer files and supports
interpolation between database depths.

### `Layer<id>_distVec.npy`

One-dimensional array of horizontal source–receiver distances for an assembled
receiver depth.

### `Layer<id>_gf_fft.npy`

Complex frequency-domain Green's functions for the selected frequencies,
distances, and GF components. The current assembly code writes an array ordered
as:

```text
(selected frequency, horizontal distance, component)
```

The component axis follows the `components` list supplied by the driver.

When `onlySplitAll=True`, the consolidated metadata is created but these GF
arrays are not assembled. That mode is used by analytic body- and Rayleigh-wave
examples.

## Per-realization reference displacement

### `surfDeepDispRea<r>.npz`

The common multi-point implementation stores:

| Key | Meaning |
| --- | --- |
| `dispPointFull` | Displacement magnitude at selected test-mass horizontal positions and reference depths. |
| `freqOut` | Selected frequency vector. |

Its usual shape is:

```text
(frequency, test mass, component, reference location)
```

with component order `[x, y, z]`. Reference-location index 0 is normally the
surface and index 1 the selected underground depth. The exact underground
reference is set by the notebook and is not necessarily always the cavity
centre.

Older helper paths may instead store keys named `dispPointAllRea` and
`attnAllRea`. Inspect `data.files` before processing outputs from a different
driver generation.

## Per-realization NN result

### `NNFullRea<r>.npz`

The standard archive contains:

| Key | Meaning |
| --- | --- |
| `IBlockTot` | Complex bulk/volume/dipole NN acceleration. |
| `IVertFaceTot` | Complex vertical outer-boundary contribution. |
| `IHorFaceTot` | Complex horizontal surface/interface contribution. |
| `ITot` | Signed total: `IBlockTot - IVertFaceTot - IHorFaceTot`. |
| `freqOut` | Frequency vector. |

For multiple test masses, contribution arrays normally have shape:

```text
(frequency, component, test mass)
```

For older single-test-mass paths, the final axis may be absent. The NN arrays
already include Newton's gravitational constant and therefore represent
acceleration spectra.

## Realization assembly

`assembleNNAllReaAllTM` and related helpers calculate RMS magnitudes:

```text
sqrt(sum_r |X_r|^2 / n_rea)
```

They return total, volume, and combined surface magnitudes. The combined
surface magnitude is formed from the complex sum
`IVertFaceTot + IHorFaceTot` before taking its absolute value for each
realization.

These assembled arrays no longer contain phase. Retain the original
`NNFullRea*.npz` files if complex response information may be needed later.

## Temporary displacement data

With `on_fly_nn = True`, receiver chunks are immediately reduced to partial NN
integrals, so the complete seismic field is not saved.

With stored-field operation, complex displacement arrays can be placed in
memory-mapped files, commonly under `/dev/shm`. Depending on `save_hv`, these
can include total, horizontal-force, and vertical-force fields. They are
temporary working data and can be extremely large.

## Source metadata

The source generator returns positions, horizontal azimuths, and a dictionary
containing:

```text
ampH, phiH, ampV, phiV
```

The current example drivers do not consistently persist random-source metadata
or random seeds. Therefore, retaining only the NN archives is not sufficient
to reproduce the exact stochastic realization.

## Inspecting an archive

Use NumPy to inspect keys and shapes without loading every array eagerly:

```python
import numpy as np

with np.load("NNFullRea0.npz") as data:
    print(data.files)
    for key in data.files:
        print(key, data[key].shape, data[key].dtype)
```

## Reproducibility set

For a reusable calculation, retain:

- the primary and cavity configurations;
- notebook or driver revision;
- Qseis template and executable version;
- `splitAll.mat` and `splitAllNew.npy`;
- all assembled layer GF and distance files;
- Qseis statuses or logs;
- all `surfDeepDispRea*.npz` and `NNFullRea*.npz` files;
- post-processing choices and ASD reference files; and
- the software revision and environment information.

