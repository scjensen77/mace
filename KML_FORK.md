# The KML fork of MACE

This branch, `kml`, is ACEsuit/mace at commit `5e524dc` (MACE 0.3.17,
2026-09-09) plus one model: `KMLMACE`, MACE with the KML plane-wave
k-space term and KML's short-range pair block trained as part of the
same model under one loss. The physics lives in the `kml` package
(https://github.com/scjensen77/KML_Model). This fork holds only the
MACE-side adapter and the flags that build it, the way `MACELES` holds
the MACE side of `les`.

Files changed against upstream:

| file | what |
|---|---|
| `mace/modules/kml_kspace.py` | new: `KMLMACE`, `KSpaceLongRangeBlock`, `set_kspace_feature_statistics` |
| `mace/modules/__init__.py` | exports those three names |
| `mace/tools/arg_parser.py` | `--model KMLMACE` and the twelve `--kml_*` flags |
| `mace/tools/model_script_utils.py` | build, feature standardization, warm start, freeze, dump, the `r_max` guard |
| `mace/tools/scripts_utils.py` | the k-space optimizer group and `--kml_lr_factor` |
| `mace/__version__.py` | `0.3.17+kml.1` |

## Install

torch first, so that MACE's unbounded torch requirement is already
satisfied and pip does not pull the default wheel:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0+cpu   # or the CUDA build for your host
pip install "mace-torch @ git+https://github.com/scjensen77/mace.git@v0.3.17-kml.1"
pip install -e /path/to/KML_Model
```

Never `pip install mace-torch` from PyPI for the joint model: it has no
`KMLMACE`, and the run dies on argparse's "invalid choice" without
saying which package answered `import mace`.

## Versioning

`mace.__version__` is upstream's version plus a local label. Every
content change on this branch is a new label and a new tag; a tag never
moves. KML_Model pins the tag and the SHA-256 of the five files above
(`Workflow/mace_fork.py`), so an install whose bytes do not match is
refused before training.

The first TorchScript error on a `KMLMACE` was a module-level
`BOHR_PER_ANG` float closed over in `KSpaceLongRangeBlock.forward`
(`python value of type 'float' cannot be used as a value … closed over
global`). That constant is now a class `Final[float]`.
`Can't redefine method: __n_features_getter` is a retry artefact, not
the cause. Tag this commit `v0.3.17-kml.2` after merge; do not move
`v0.3.17-kml.1`.

When `--model KMLMACE` is built with the fork defaults `kml_nsrbf==1`
or `kml_lr_factor==1.0`, `model_script_utils` logs a WARNING: those
defaults are not the published protocol (nsrbf 5, rcut 2.45 bohr,
lr_factor 0.1).

The diff against upstream is `git diff 5e524dc kml`. The gates, the
drivers and the guide are in KML_Model (`tests/kmlmace_fork_check`,
`Workflow/fit_kml_mace.py`, `docs/source/part2-python/joint-mace.md`).
