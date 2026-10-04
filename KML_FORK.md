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
| `mace/modules/models.py` | `ScaleShiftMACE._scale_shift_forward`: the parent body, so a compiled `KMLMACE` can call it (TorchScript cannot resolve `super().forward`). Numerics of plain `ScaleShiftMACE` are unchanged. |
| `mace/modules/__init__.py` | exports those three names |
| `mace/tools/arg_parser.py` | `--model KMLMACE` and the twelve `--kml_*` flags |
| `mace/tools/model_script_utils.py` | build, feature standardization, warm start, freeze, dump, the `r_max` guard, unpublished-default warnings |
| `mace/tools/scripts_utils.py` | the k-space optimizer group and `--kml_lr_factor` |
| `mace/__version__.py` | `0.3.17+kml.2` |

## Install

torch first, so that MACE's unbounded torch requirement is already
satisfied and pip does not pull the default wheel:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0+cpu   # or the CUDA build for your host
pip install "mace-torch @ git+https://github.com/scjensen77/mace.git@v0.3.17-kml.2"
pip install -e /path/to/KML_Model
```

Never `pip install mace-torch` from PyPI for the joint model: it has no
`KMLMACE`, and the run dies on argparse's "invalid choice" without
saying which package answered `import mace`.

Until `v0.3.17-kml.2` is tagged, install this branch by commit instead
of the tag line above. KML_Model's `FORK_TAG` stays `v0.3.17-kml.1`
until that tag exists.

## Versioning

`mace.__version__` is upstream's version plus a local label. Every
content change on this branch is a new label and a new tag; a tag never
moves. KML_Model pins the tag and the SHA-256 of the fork files
(`Workflow/mace_fork.py`), so an install whose bytes do not match is
refused before training.

This commit is `0.3.17+kml.2`. Tag it `v0.3.17-kml.2` after merge; do
not move `v0.3.17-kml.1`. Squash-merge this PR (or rewrite the branch
to clean commits) so no commit reachable from that tag is the
PLACEHOLDER tree that appeared on the draft branch.

Two TorchScript blockers on `v0.3.17-kml.1` stopped `mace_run_train`
writing `*_compiled.model`:

1. A module-level `BOHR_PER_ANG` float closed over in
   `KSpaceLongRangeBlock.forward` (`python value of type 'float'
   cannot be used as a value … closed over global`). Fixed here as a
   class `Final[float]`.
2. `super().forward` in `KMLMACE.forward` (`'Tensor' object has no
   attribute or method 'forward'`). TorchScript does not resolve
   `super` on a compiled subclass. The parent body is
   `ScaleShiftMACE._scale_shift_forward`; `KMLMACE.forward` calls that.

`Can't redefine method: __n_features_getter` is a retry artefact, not
a third cause. After both fixes, `torch.jit.script(KMLMACE)` and
`e3nn.util.jit.compile` succeed, and `mace_run_train` writes
`*_compiled.model` that `torch.jit.load` runs. Plain `ScaleShiftMACE`
is bit-identical to `v0.3.17-kml.1`.

When `--model KMLMACE` is built with the omitted fork defaults
(`kml_nsrbf` left at 1, or `kml_lr_factor` left at 1.0 without
`--kml_freeze`), `model_script_utils` logs a WARNING: those defaults
are not the published protocol (nsrbf 5, rcut 2.45 bohr, lr_factor
0.1). An explicit `--kml_nsrbf 1` is silent. `--kml_freeze` skips the
lr_factor warning (the factor is unused).

## Tag-bump checklist (KML_Model, same change as the tag)

A new fork tag is four coordinated pins in KML_Model, plus this
version file. Miss one and the gates disagree:

1. `Workflow/mace_fork.py`: `FORK_TAG`, `EXPECTED_MACE_VERSION`,
   `FORK_COMMIT` (the tagged SHA), and `FORK_FILE_SHA256` of every
   file in the table above (including `modules/models.py` from kml.2).
2. `pyproject.toml` extra and `.github/workflows/ci.yml` tag.
3. `requirements-lock/ci.txt` `mace-torch @ git+…@<FORK_COMMIT>`.
4. Tag `scjensen77/mace` `v0.3.17-kml.N`; never move an old tag.

The gates check (1)–(3). They cannot see a GitHub tag that does not
yet exist.

The diff against upstream is `git diff 5e524dc kml`. The gates, the
drivers and the guide are in KML_Model (`tests/kmlmace_fork_check`,
`Workflow/fit_kml_mace.py`, `docs/source/part2-python/joint-mace.md`).
