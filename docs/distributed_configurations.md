# Distributed tutorial configurations

The `AllConfigFiles/` directory contains the primary and cavity configuration
files used by the tutorial notebooks. These files preserve the numerical and
physical settings of the documented examples and provide starting points for
new calculations.

They are **example configurations, not installation-independent defaults**.
In particular, the `[paths]` entries contain absolute paths from the machine on
which the examples were developed. Edit those entries before running a
tutorial on another system.

For the meaning, units, and constraints of every option, see the
[complete configuration reference](configuration_complete.md). For the depth
and coordinate conventions, see [coordinates and units](coordinates_and_units.md).

## Configuration-file types

Most Newtonian-noise examples use two INI files:

1. A primary file named `configParse*.ini`. It defines the geological model,
   frequency grid, computational domain, test-mass positions, stochastic
   sources, paths, simulation mode, and parallel settings.
2. A cavity file named `SingleSphCavity*.ini` or
   `SingleCubCavity*.ini`. It defines the cavity centre, dimensions, rotation
   where applicable, and the locally refined region.

The displacement-only examples use a primary configuration even when the
cavity parameters do not enter the quantity being plotted. Keep primary and
cavity files from the same tutorial directory together unless you deliberately
intend to construct a new physical case.

## Directory catalogue

| Directory | Intended notebook or task | Field and medium | Realizations |
| --- | --- | --- | --- |
| `SphereBodyTutorial/SingleWave/` | `getSphNNSingleBody.ipynb` | Analytic body wave; homogeneous full space or half-space | One prescribed realization |
| `SphereBodyTutorial/StochWave/` | `getSphNNStochBody.ipynb` | Stochastic analytic body waves; homogeneous full space or half-space | 100 |
| `SphereRayTutorial/SingleWave/` | `getSphNNSingleRay.ipynb` | Analytic Rayleigh wave; homogeneous half-space | One prescribed realization |
| `SphereRayTutorial/StochWave/` | `getSphNNStochRay.ipynb` | Stochastic analytic Rayleigh waves; homogeneous half-space | 100 |
| `CuboidBodyTutorial/SingleWave/` | `getCubNNSingleBody.ipynb` | Analytic body wave; homogeneous full space or half-space | One prescribed realization |
| `CuboidBodyTutorial/StochWave/` | `getCubNNStochBody.ipynb` | Stochastic analytic body waves; homogeneous full space or half-space | 100 |
| `SphereValidTutorial/SingleForce/` | Single-force Qseis and SPECFEM3D validation notebooks | Qseis full field; homogeneous or layered half-space | One source realization |
| `DispTutorialConfig/` | `getSeisDisp2D.ipynb` | Qseis full field; internal or ring sources in a homogeneous half-space | One realization |
| `SphCCTutorialConfig/` | `getSeisCCF.ipynb` | Body wave, Rayleigh wave, or Qseis full field; homogeneous or layered | 1,000 |
| `SphAttnTutorialConfig/` | `getSeisAttn.ipynb` | Body wave, Rayleigh wave, or Qseis full field; homogeneous or layered | 1,000 |

The realization count listed above is the value stored in the corresponding
primary INI file. A notebook may replace it while setting up a small test or a
specific deterministic source.

### Files within the multi-mode directories

`SphCCTutorialConfig/` and `SphAttnTutorialConfig/` use the following naming
convention:

| Name fragment | Meaning |
| --- | --- |
| `BodyHomo` | Analytic stochastic P- and S-wave field in a homogeneous medium |
| `RayHomo` | Analytic stochastic Rayleigh-wave field in a homogeneous half-space |
| `InternalHomo` | Qseis full field from sources distributed inside a homogeneous model |
| `RingHomo` | Qseis full field from sources distributed on a ring around a homogeneous model |
| `InternalLayer` or `InternalLayered` | Qseis full field from internal sources in a layered model |
| `RingLayer` or `RingLayered` | Qseis full field from ring sources in a layered model |

The analytic modes set `if_body = True` or `if_ray = True`. The Qseis modes
set `if_full_field = True`. Analytic homogeneous modes must not be reused for a
layered model: use a full-field configuration and a Qseis Green's-function
database instead.

## Preparing a distributed configuration

It is safest to preserve the distributed file as a reference and make a copy
for each calculation. Then perform the following checks.

### 1. Point the notebook to the distributed files

The existing tutorial notebooks may still show the absolute configuration
paths used during development. Replace them with paths appropriate for the
checkout. When Jupyter is started in the project root, for example:

```python
configFile = "AllConfigFiles/SphereValidTutorial/SingleForce/configParseHalfSpaceFullFieldLayer.ini"
cavityFile = "AllConfigFiles/SphereValidTutorial/SingleForce/SingleSphCavityHalfSpace.ini"
```

If the notebook is launched from another directory, use an absolute path or
construct the path from a known project root. Relative paths are interpreted
from the Python process's current working directory, not necessarily from the
notebook file's directory.

### 2. Replace the `[paths]` entries

Every distributed primary file currently contains machine-specific absolute
paths. Review all entries in `[paths]`:

```ini
[paths]
template_file = /path/to/pySeisNN1DFull/qseisTemplates/qs6inpNTempN.dat
input_folder = /path/to/run/QseisInpN/
qseis_exe = /path/to/fomosto-qseis/src/fomosto_qseis2006b
out_disp_folder = /path/to/run/OutDisp/
out_disp_rea = /path/to/run/OutDispRea/
site_asd_path = /path/to/pySeisNN1DFull/TERZ/
et_d_path = /path/to/pySeisNN1DFull/ETDesign/
```

- `template_file` should point to the unchanged template distributed at
  `qseisTemplates/qs6inpNTempN.dat`. The software creates job-specific Qseis
  inputs from this master file; do not edit the master for individual runs.
- `input_folder` is a writable working directory for Qseis inputs and the
  generated Green's-function database.
- `qseis_exe` is the locally compiled Qseis executable.
- `out_disp_folder` and `out_disp_rea` are writable output locations.
- `site_asd_path` and `et_d_path` point to the supplied site-noise and ET-D
  sensitivity data where a notebook uses them.

Create the required output directories before a long run, and ensure that the
filesystem has sufficient free space.

### 3. Protect existing output

The seismic-field tutorials intentionally share `AllTutOut/DispTests/` in the
development setup. Running another case can therefore replace files created by
an earlier one. Assign separate output directories when results must be
retained or compared.

Also check `qseis_out_del`. When it is `yes`, an existing Green's-function
database in `input_folder` can be deleted and rebuilt. Use a dedicated working
directory and change this setting only after verifying that the database is
compatible with the current model and sampling parameters.

### 4. Check mode and geometry consistency

Before running, verify that:

- exactly the intended field mode is enabled (`if_body`, `if_ray`, or
  `if_full_field`);
- analytic body- and Rayleigh-wave modes are used only with their homogeneous
  models;
- full-space or half-space settings match the branch selected in the notebook;
- the cavity type and cavity file agree;
- all test masses and cavities use the positive-downward depth convention;
- `xmax_gf`, `zmax_gf`, `t_max`, and `n_samp` cover the selected domain,
  source distances, travel times, and frequency band; and
- the requested core counts suit the machine or batch allocation.

## Notebook overrides

The INI file supplies the initial configuration, but the tutorials sometimes
replace values in Python. This is intentional for prescribed single waves,
single-force validation, selected receiver locations, visualization grids, or
quick tests. The effective calculation is therefore determined by both the
INI file and assignments made later in the notebook. Review the notebook cells
after loading the configuration when adapting an example.

## Recommended validation before a production run

1. Load the copied INI files and run the notebook only through configuration
   loading and input validation.
2. Print or inspect the resolved paths and the selected simulation-mode flags.
3. Confirm the model arrays, frequency grid, domain, test-mass positions, and
   cavity geometry.
4. Run a reduced calculation with fewer realizations or sources.
5. Assign production output directories and core counts, then submit the full
   calculation.

