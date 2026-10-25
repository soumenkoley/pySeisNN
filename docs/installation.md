# Environment setup and execution

The repository also distributes the tutorial INI files under
`AllConfigFiles/`. After installing the Python dependencies and Qseis, follow
the [distributed configuration guide](distributed_configurations.md) to select
a tutorial configuration and replace its machine-specific paths.

The software can be run directly from its source directory. Installing the
repository as a Python package is not currently required, either for notebooks
or for long batch calculations.

This guide covers:

- creation of the Conda environment;
- compilation of the external Qseis program;
- configuration of input and output paths;
- running notebooks interactively; and
- converting notebooks to Python scripts for batch execution.

## Supported platform

The documented setup targets 64-bit Linux. The Python calculation itself is
largely portable, but the workflow depends on:

- a compiled Fortran Qseis executable;
- process-based parallelism;
- Unix filesystem paths; and
- optionally, shared-memory files under `/dev/shm`.

Windows and macOS have not been covered by the present tutorial setup.

## Software components

The workflow has two distinct parts:

1. the Python source tree containing the notebooks and `modules/`; and
2. the external `fomosto-qseis` source and compiled executable.

The Conda environment provides the Python libraries and the Fortran build
toolchain used for Qseis.

SPECFEM3D is required only for reproducing the independent validation examples.
It is not needed for standard Qseis-based displacement or NN calculations and
is therefore not part of this installation procedure.

## Create the Conda environment

From the repository root, run:

```bash
conda env create -f environment.yml
conda activate pySeis1DNN
```

The environment contains:

- Python 3.11;
- NumPy;
- SciPy;
- Matplotlib;
- Joblib;
- JupyterLab and ipykernel;
- `nbconvert` for notebook-to-script conversion;
- `pysurf96` for Rayleigh- and Love-wave dispersion;
- a Fortran compiler; and
- Autoconf, Automake, Libtool, and Make for building Qseis.

The scientific Python packages use compatible version ranges, while
`pysurf96` is fixed to version 1.0.1, which is used by the current code.

To update an existing environment after `environment.yml` changes, use:

```bash
conda env update -n pySeis1DNN -f environment.yml --prune
```

`--prune` removes packages that are no longer declared. Omit it if the
environment also contains unrelated tools that must be retained.

## Verify the Python environment

Activate the environment and run this check from the repository root:

```bash
python -c "import numpy, scipy, matplotlib, joblib, pysurf96; print('Python dependencies OK')"
```

Then confirm that the project modules are importable:

```bash
python -c "from modules import configLoader, createDB, gfLoader, simDisp, NNCompute; print('Project imports OK')"
```

The second command must be run from the project root because the repository is
used directly as source and is not yet installed into `site-packages`.

## Register a Jupyter kernel

Register the environment once:

```bash
python -m ipykernel install --user \
  --name pySeis1DNN \
  --display-name "Python (pySeis1DNN)"
```

When opening a notebook, select `Python (pySeis1DNN)` as its kernel. A notebook
opened with another Python environment may fail to import SciPy, Joblib, or
`pysurf96` even if those packages are installed in `pySeis1DNN`.

## Obtain Qseis

The code uses the Qseis 2006b backend distributed as `fomosto-qseis`. The
source used for the current development setup is:

```text
https://git.pyrocko.org/pyrocko/fomosto-qseis.git
```

Clone it outside the Python repository, for example as a sibling directory:

```bash
cd /path/to/workspace
git clone https://git.pyrocko.org/pyrocko/fomosto-qseis.git
cd fomosto-qseis
```

The locally tested checkout was revision:

```text
a8f76d1
```

Recording a Qseis revision is recommended for reproducible calculations. A
newer revision may also work, but it should be validated against the tutorial
examples before production use.

## Compile Qseis

Activate the Conda environment before configuring and compiling:

```bash
conda activate pySeis1DNN
cd /path/to/fomosto-qseis
```

If the repository does not contain a `configure` script, create it with:

```bash
autoreconf -i
```

Then build Qseis:

```bash
./configure
make
```

The executable should be created at:

```text
/path/to/fomosto-qseis/src/fomosto_qseis2006b
```

It is not necessary to run `sudo make install`. The Python configuration can
point directly to this executable.

### Why Qseis should be built inside the environment

The executable can link dynamically to the Fortran runtime supplied by Conda.
Building and running it with `pySeis1DNN` active ensures that the same runtime
libraries are available.

Check the executable and its libraries with:

```bash
test -x /path/to/fomosto-qseis/src/fomosto_qseis2006b
ldd /path/to/fomosto-qseis/src/fomosto_qseis2006b
```

No dependency should be reported as `not found`.

If Qseis was previously configured under a different compiler or Conda
environment, clean or freshly clone the source and rerun `./configure` after
activating `pySeis1DNN`. Generated Makefiles can contain absolute compiler
paths from the environment in which configuration was performed.

## Qseis template

> The required template is bundled with this repository in
> `qseisTemplates/qs6inpNTempN.dat`. It is not supplied by the standard
> `fomosto-qseis` checkout and should not be moved into that checkout.

The pySeisNN distribution includes its Qseis input template at:

```text
/path/to/pySeisNN1DFull/qseisTemplates/qs6inpNTempN.dat
```

This is not an ordinary ready-to-run Qseis input file. It contains Python
format placeholders for values generated by pySeisNN, including receiver
depth, receiver count and distances, time sampling, sampling rate, and the
layered velocity model. `modules/createDB.py` copies the master template and
replaces those placeholders for every Qseis job.

Keep the template in the pySeisNN source tree and point `template_file` to it.
It does not need to be moved or copied into the `fomosto-qseis` checkout. This
also prevents a Qseis update or fresh clone from removing the pySeisNN-specific
asset.

Do not edit the master template separately for individual simulations. Inputs
that vary between runs belong in the primary configuration and are inserted by
the database generator. If the template itself must be changed, retain the
original Qseis attribution header and record the change with the corresponding
pySeisNN revision.

## Configure local paths

The tutorial INI files contain machine-specific absolute paths. At minimum,
update:

```ini
[paths]
template_file = /path/to/pySeisNN1DFull/qseisTemplates/qs6inpNTempN.dat
input_folder = /path/to/results/QseisInpN/
qseis_exe = /path/to/fomosto-qseis/src/fomosto_qseis2006b
out_disp_folder = /path/to/results/OutDisp/
out_disp_rea = /path/to/results/OutDispRea/
site_asd_path = /path/to/project/TERZ/
et_d_path = /path/to/project/ETDesign/
```

The configured directories must already exist because input validation checks
them before starting a run:

```bash
mkdir -p /path/to/results/QseisInpN
mkdir -p /path/to/results/OutDisp
mkdir -p /path/to/results/OutDispRea
```

Use separate production directories for calculations that must be retained.
The tutorial configurations intentionally reuse output paths and can overwrite
previous data.

## Database deletion warning

Input generation always removes top-level `.dat` files from `input_folder`.
When:

```ini
qseis_out_del = yes
```

it also recursively deletes existing Qseis run subdirectories before creating
the new database. Never use a directory containing unrelated or irreplaceable
files.

## Reference data

Some post-processing and model-validation notebooks expect:

```text
TERZ/
  asdSurf.mat
  asdBH.mat
  attnModel.mat
  DCTerziet.mat

ETDesign/
  ETD.mat
```

Not every notebook uses every file. The paths remain required configuration
fields in the current loader, so point them to valid directories even for an
example that does not read the reference data directly.

## Run notebooks interactively

Activate the environment and start Jupyter from the source root:

```bash
cd /path/to/pySeisNN1DFull
conda activate pySeis1DNN
jupyter lab
```

Starting from the root makes `modules` importable and keeps relative paths
predictable. Select the `Python (pySeis1DNN)` kernel.

Before a costly run:

1. inspect the active configuration path in the notebook;
2. check every configured filesystem path;
3. reduce `n_rea`, `n_src`, domain size, or worker counts for a smoke test;
4. verify that database deletion is intentional; and
5. check Qseis process return statuses before using the assembled GFs.

## Convert a notebook to a Python script

Installation of the project is not required for batch execution. Convert the
notebook from the repository root:

```bash
jupyter nbconvert --to script getSeisAttn.ipynb
```

This creates `getSeisAttn.py`. Run it from the same root directory:

```bash
MPLBACKEND=Agg python -u getSeisAttn.py
```

`MPLBACKEND=Agg` selects a non-interactive plotting backend suitable for a
headless compute node. `python -u` makes log output unbuffered.

The current notebooks generally define `main()` and call it under:

```python
if __name__ == "__main__":
    main()
```

so their converted scripts can be executed directly. Inspect the generated
file before a production run, especially if the notebook contains exploratory
cells, stale outputs, shell commands, or code that assumes an interactive
display.

## Preserve generated scripts deliberately

`nbconvert` writes the Python script beside the notebook by default and can
replace a previously generated file with the same name. For a production run,
it is safer to copy the converted script into a run directory together with:

- the primary configuration;
- the cavity configuration, if used;
- the source notebook revision;
- a record of the Conda environment;
- the Qseis revision and template; and
- scheduler and logging scripts.

Record the environment with:

```bash
conda env export -n pySeis1DNN > environment-resolved.yml
```

The repository `environment.yml` is the portable human-maintained
specification. The resolved export records the exact packages used for one
calculation and can include platform-specific build identifiers.

## Run a long calculation from a shell

For a machine where persistent background processes are permitted:

```bash
cd /path/to/pySeisNN1DFull
conda activate pySeis1DNN
MPLBACKEND=Agg python -u runCalculation.py > runCalculation.log 2>&1
```

Use the local scheduler or workload manager instead of an unmanaged background
process on a shared cluster.

The script should explicitly save important arrays and figures. Interactive
`plt.show()` calls do not provide a persistent result on a headless node.

## Example SLURM job

Cluster settings vary, but a minimal CPU job has this structure:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=pyseis-nn
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=pyseis-nn-%j.log

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate pySeis1DNN

cd /path/to/pySeisNN1DFull

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u runCalculation.py
```

Set the configuration worker counts no higher than the CPUs allocated by the
scheduler. `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` help prevent each Joblib
worker from creating an additional pool of numerical-library threads.

Do not copy the memory and time requests blindly. Size them using a smaller
test of the actual grid, source count, frequency count, and storage mode.

## Memory and `/dev/shm`

With `on_fly_nn = True`, displacement chunks are reduced directly to partial
NN integrals and the full seismic field is not retained. This is the preferred
production mode.

With stored-field operation, complex memory-mapped displacement files may be
created under `/dev/shm`. Confirm its available capacity with:

```bash
df -h /dev/shm
```

On some clusters, `/dev/shm` is limited independently of the job's requested
RAM. A job can therefore fail from insufficient shared-memory capacity even
when ordinary memory remains available.

## Parallel-worker guidance

The relevant configuration fields are:

```ini
compute_cores_qseis = 32
compute_cores_disp = 32
compute_cores_nn = 32
```

Qseis generation parallelizes independent jobs and usually scales well.
Displacement and NN stages can become limited by memory bandwidth, data
movement, receiver count, or process overhead. More workers do not guarantee a
faster calculation.

Start with modest worker counts, monitor memory and CPU utilization, and scale
only after measuring the relevant stage.

## Smoke-test checklist

Before a production calculation, verify:

- the `pySeis1DNN` environment is active;
- Python and project imports succeed;
- `pysurf96` returns a dispersion value for a small test model;
- the Qseis executable exists, is executable, and has no missing libraries;
- the Qseis template exists;
- all configured directories exist and are writable;
- reference-data paths are valid;
- a small Qseis job completes successfully;
- a reduced one-realization calculation produces finite displacement and NN
  arrays; and
- the intended outputs are explicitly saved.

## Current source-based execution model

No `setup.py` or `pyproject.toml` is currently required. The software is run
from its checked-out source directory, and notebooks or converted scripts
import `modules` directly.

Packaging the project later could allow commands such as `pip install -e .`
and execution from arbitrary directories. That is a useful future improvement,
but it is independent of running notebooks and batch scripts today.

