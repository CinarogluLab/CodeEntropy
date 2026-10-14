CodeEntropy (CinarogluLab fork)
==============================

<p align="center">
    <img src="docs/_static/logos/biosim-codeentropy_logo_light.png#gh-dark-mode-only" alt="CodeEntropy logo" width="300"/>
    <img src="docs/_static/logos/biosim-codeentropy_logo_dark.png#gh-light-mode-only" alt="CodeEntropy logo" width="300"/>
</p>

This is a fork of [CCPBioSim/CodeEntropy](https://github.com/CCPBioSim/CodeEntropy), maintained by the Çınaroğlu
lab. It is based on CodeEntropy and adds performance improvements, corrections and tools for computing
binding entropies of protein–ligand complexes in explicit water.

CodeEntropy computes the configurational entropy of molecular systems from molecular dynamics (MD) trajectories
using the multiscale cell correlation (MCC) method. Entropy is built up level by level (atoms, residues, whole
molecules) from force and torque covariances and from dihedral-angle statistics. Combined with the waterEntropy
package, it also gives the entropy of the water molecules around a solute.

## Why this fork

We use CodeEntropy for the binding entropy of ligands to proteins, which needs many long trajectories of large
solvated systems (complex, protein alone, ligand alone, pure water) with replicates. With the original code, one
such trajectory could take many hours to days, so a full study was impractical. Working on it
also revealed some errors that change the computed entropies. This fork collects the changes we made to address
both.

## How it differs from the original

### Faster, with identical results

Several parts of the calculation repeated the same expensive work many times per frame: recomputing
whole-molecule axes once per residue, re-reading dihedrals residue by residue, sorting far more neighbour
distances than needed, and similar overheads in the water calculation. These are now done once, with guards
that keep every result exactly the same.

We checked this by comparing complete output files from the original and the fork on a real protein–ligand
trajectory: the numbers are identical, bit for bit.

### Corrections

- **Residue-level conformational entropy.** In upstream versions 2.3.0–2.5.0 this term came out as zero (or,
  in some group layouts, was taken from the wrong molecule, or the run crashed). It is now computed correctly.
- **Dihedral states across 0°/360°.** Dihedral angles were assigned to conformational states without treating
  0° and 360° as the same angle, so states near that boundary were split. This biased conformational entropies
  by a few percent. Angles are now treated as periodic.
- **First-shell water results.** The detailed water entropy results were partly dropped or mislabelled in the
  output. They are now written in full to the output file.

Because of these corrections, results from this fork should not be mixed with results from the original
CodeEntropy, or from earlier versions of this fork, in one analysis.

### New tools and options

- **Binding entropy assembly:** `python -m CodeEntropy.results.binding_entropy config.json` combines runs on the
  complex, the protein, the ligand and pure water into a binding entropy, following Ali et al., *J. Comput.-Aided
  Mol. Des.* 35, 911 (2021). Every scientific choice is stated explicitly in the configuration file (see
  `examples/binding_entropy_config.example.json`).
- **Opt-in settings** (environment variables; the defaults behave like the original):
  - `CODEENTROPY_PREDAG_WATER_GROUPS=1`: with water entropy on, skip the bulk-water MCC calculation that the
    binding workflow does not use. This is much faster for solvated systems; solute results are unchanged.
  - `CODEENTROPY_BULK_WATER_EXPORT=1`: for a pure-water simulation, export the reference water entropy.
  - `CODEENTROPY_FAST_NEIGHBOURS=1`: a faster neighbour search in the water calculation, with identical results.

## Performance

Speed depends on system size and settings. As a guide, for a solvated bromodomain–ligand complex of about 40 000
atoms:

- **Water entropy on, default settings:** roughly an order of magnitude faster than the original.
- **The binding-entropy workflow** (`CODEENTROPY_PREDAG_WATER_GROUPS=1`): a full 1000-frame trajectory takes well
  under an hour on one CPU core, where the original took more than a day.

Independent runs (replicates, different systems) scale almost perfectly when run side by side on a multi-core
machine. We recommend that over the built-in Dask parallelism, which was slower than serial runs in our tests.

## Installation

This fork needs the matching waterEntropy fork. Install that first:

```bash
pip install "waterEntropy @ git+https://github.com/CinarogluLab/waterEntropy"
pip install "CodeEntropy @ git+https://github.com/CinarogluLab/CodeEntropy"
```

Check the versions:

```bash
python -c "import CodeEntropy, waterEntropy; print(CodeEntropy.__version__, waterEntropy.__version__)"
```

## Quick start

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # one thread per run; run replicates in parallel instead
export CODEENTROPY_PREDAG_WATER_GROUPS=1          # recommended for solvated binding calculations
CodeEntropy --top_traj_file md.tpr md.trr --water_entropy True --output_file output.json
```

Good to know:

- The trajectory must contain **forces**, saved in the **same frames** as the coordinates. If they were saved at
  different intervals, use `--step` to pick the frames that have both.
- Do not use `--selection_string "not water"` with water entropy. It silently removes the water term.
- Keep the `output.json` files: they hold the complete results, including the water entropy details.

Everything else works as in the original; see
[CodeEntropy's documentation](https://codeentropy.readthedocs.io/en/latest/).

## Validation

- **Upstream's tests pass**, together with new tests for every change. Tests that encoded the corrected errors were
  updated.
- **The performance changes give identical results**, checked by comparing full output files on a real
  protein–ligand trajectory, and by tests designed to break each shortcut (molecules crossing the periodic
  boundary, changing box sizes, and so on).
- **The corrections were checked against cases with known answers**, and their effect was measured on real data.
- **The code was audited** for scientific and mathematical correctness: equations, units, constants. That audit
  found the dihedral error, now fixed.


## Credit and citation

CodeEntropy is developed by CCPBioSim and its contributors. Please cite the original CodeEntropy (see
`CITATION.cff`) and the MCC method papers when using this fork, and mention that you used the CinarogluLab fork
and its version.

Project based on:

- [arghya90/CodeEntropy](https://github.com/arghya90/CodeEntropy) version 0.3
- [jkalayan/PoseidonBeta](https://github.com/jkalayan/PoseidonBeta)


Distributed under the MIT licence, as the original (see `LICENSE`). Report problems with this fork in its
[issue tracker](https://github.com/CinarogluLab/CodeEntropy/issues). Report problems with CodeEntropy itself
upstream.
