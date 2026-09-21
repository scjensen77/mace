"""KML's plane-wave k-space term as a MACE long-range block.

Written by Mathieu Istas (``istasm``), in the patched MACE clone that
accompanies his ``kspace-gradient-descent`` work (KML_Model PR #47).
That clone was never under version control, so this file entered the
repository as a whole-file add and ``git blame`` credits the commit
that captured it rather than its author. Of the 407 lines here, 358
are his; what was added on top is the short-range pair block, the
freeze and warm-start plumbing, the ``r_max`` guard, and the single
guarded ``kml`` import.

MACE learns the short range through message passing on a graph cut at
``r_max``; the plane-wave basis of the KML potential is periodic and has
no cutoff, so it carries what the graph cannot see.  Both terms are
differentiable, so both train from one loss and one optimizer step:

    E = E0 + E_MACE(R) + E_kspace(R)        F = -dE/dR

The basis functions live in the ``kml`` package
(``kml.torch_kspace.KSpaceEnergy``) -- one implementation, checked
against the Fortran-referenced numpy one by
``KML_Model/tests/kspace_gradient_check``.  This module is the adapter:
it takes MACE's batched graph, hands the k-space term the positions,
the batch index and the box, and hands back one energy per
configuration.

``kml`` is an optional dependency, as ``les`` is for ``MACELES``.  Make
it importable with ``pip install -e KML_Model`` or on ``PYTHONPATH``.

Limits, both stated where they are enforced:

- The box must be **cubic** (one side length per configuration), which
  is what ``kml`` assumes everywhere: ``sca = 2 pi / ell``.
- Energy and forces only.  A stress or a virial needs the derivative
  with respect to the cell, and a general strain takes a cubic cell out
  of the space this basis is written in.
"""

import logging
from typing import Dict, Optional

import torch

from mace.modules.models import ScaleShiftMACE
from mace.modules.utils import get_outputs, safe_double


# MACE positions are in angstrom; KML's pair functions take bohr.
BOHR_PER_ANG = 1.0 / 0.529177210903   # CODATA 2018, kml/units.py


def _import_kml():
    """``KSpaceEnergy`` and ``generate_kvectors``, or an error saying how to get them.

    Both live in ``kml`` and both are needed to build a block, so both are
    imported here: an unguarded ``import kml`` anywhere else in the module
    would raise a bare ``ModuleNotFoundError`` first and this message would
    never be seen.
    """
    try:
        from kml.kvectors import generate_kvectors
        from kml.torch_kspace import KSpaceEnergy

        return KSpaceEnergy, generate_kvectors
    except ImportError as exc:
        raise ImportError(
            "Cannot import 'kml', which holds the plane-wave k-space basis. "
            "Install the KML_Model repository next to this one "
            "(`pip install -e KML_Model`) or put it on PYTHONPATH."
        ) from exc


def cubic_side_lengths(
    cell: torch.Tensor, num_graphs: int, tolerance: float = 1.0e-6
) -> torch.Tensor:
    """One box side per configuration, from MACE's flat ``[3 n_graphs, 3]`` cell.

    Refuses anything but a cubic cell: the plane-wave basis is written
    in ``sca = 2 pi / ell``, so a non-cubic box would silently be fitted
    as the cube with the same first lattice vector. ``tolerance`` is
    relative to the box side.
    """
    cells = cell.reshape(num_graphs, 3, 3)
    sides = torch.diagonal(cells, dim1=-2, dim2=-1)  # [n_graphs, 3]

    if bool((sides <= 0).any()):
        raise ValueError(
            "the k-space term needs a periodic box on every configuration, "
            "and at least one has a zero or negative lattice vector. A "
            "non-periodic frame has no plane-wave basis; fit those with "
            "MACE alone."
        )

    scale = sides.abs().amax(dim=-1, keepdim=True)
    off_diagonal = cells - torch.diag_embed(sides)
    skew = off_diagonal.abs().amax(dim=-1).amax(dim=-1, keepdim=True) / scale
    anisotropy = (
        sides.amax(dim=-1, keepdim=True) - sides.amin(dim=-1, keepdim=True)
    ) / scale
    worst = float(torch.maximum(skew, anisotropy).max())
    if worst > tolerance:
        # str(), not a format spec: TorchScript compiles this function and
        # rejects formatting inside an f-string.
        raise ValueError(
            "the k-space term requires a cubic cell; the worst "
            "configuration in this batch is off by "
            + str(worst)
            + " relative (tolerance "
            + str(tolerance)
            + "). kml's "
            "plane-wave basis is built from sca = 2 pi / ell with one side "
            "length per frame."
        )

    return sides.mean(dim=-1)


class KSpaceLongRangeBlock(torch.nn.Module):
    """One k-space energy per configuration, from a MACE batch.

    ``nup`` and the cutoffs are ``kml.generate_kvectors``' arguments and
    fix the basis; the coefficients are the trainable parameters. Pass
    ``k2cut`` for the single-tier (Fortran-equivalent) tables, or the
    pair/triplet pair for two-tier.
    """

    def __init__(
        self,
        nup: int,
        k2cut: Optional[int] = None,
        k2cut_pair: Optional[int] = None,
        k2cut_triplet: Optional[int] = None,
        include_constant: bool = False,
        dtype: Optional[torch.dtype] = None,
        nsrbf: int = 1,
        rcut: Optional[float] = None,
    ):
        super().__init__()
        # The block joins a model whose other parameters are already in
        # the run's dtype; a float64 island here would promote every
        # energy it touches and quietly change what the loss is computed in.
        dtype = torch.get_default_dtype() if dtype is None else dtype
        kspace_energy_cls, generate_kvectors = _import_kml()

        self.nup = int(nup)
        self.k2cut = k2cut
        self.k2cut_pair = k2cut_pair
        self.k2cut_triplet = k2cut_triplet

        kvdata = generate_kvectors(
            nup,
            k2cut=k2cut,
            k2cut_pair=k2cut_pair,
            k2cut_triplet=k2cut_triplet,
            verbose=False,
        )
        # KML's short-range pair block (nsrbf > 1), evaluated on MACE's
        # edge list with distances converted to bohr, so the published
        # K coefficients load with the energy unit factor alone.
        short_range = None
        if nsrbf > 1:
            if rcut is None:
                raise ValueError("a short-range block (kml_nsrbf > 1) needs kml_rcut, in bohr")
            from kml.torch_kspace import ShortRangeBasis
            short_range = ShortRangeBasis(nsrbf, rcut, dtype=dtype)
        self.nsrbf = int(nsrbf)
        self.rcut = float(rcut) if rcut is not None else None
        self.energy = kspace_energy_cls(
            kvdata, dtype=dtype, include_constant=include_constant,
            short_range=short_range,
        )
        # Kept so a warm start can compare its bundle's table with this one.
        self.kvdata = kvdata
        self.n_2body = len(kvdata.shells)
        self.n_triplet = len(kvdata.triplet_shells)

        logging.info(
            f"KML k-space block: nup={nup} "
            f"k2cut={k2cut if k2cut is not None else (nup + 1) ** 2}, "
            + (f"{nsrbf - 1} short-range pair functions (rcut {rcut} bohr) + "
               if nsrbf > 1 else "")
            + f"{self.n_2body} 2-body shells + {self.n_triplet} triplet shells "
            f"= {self.energy.n_features} coefficients"
            + (" + constant" if include_constant else "")
        )

    @property
    def n_features(self) -> int:
        return self.energy.n_features

    def features(self, positions, batch, cell, num_graphs,
                 edge_index=None, shifts=None) -> torch.Tensor:
        """Standardized basis features, ``[n_graphs, n_features]``."""
        ell = cubic_side_lengths(cell, num_graphs)
        return self.energy.features(positions, batch, ell, num_graphs,
                                    edge_index, shifts, BOHR_PER_ANG)

    def forward(
        self,
        positions: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        num_graphs: int,
        edge_index: Optional[torch.Tensor] = None,
        shifts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``energy[n_graphs]``, in the units the coefficients carry (eV)."""
        ell = cubic_side_lengths(cell, num_graphs)
        return self.energy(positions, batch, ell, num_graphs,
                           edge_index, shifts, BOHR_PER_ANG)


def set_kspace_feature_statistics(
    block: KSpaceLongRangeBlock,
    data_loader,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> None:
    """Standardize the k-space features on the training set.

    The raw features are sums of ``|rho(k)|^2`` and of triple products
    over shells of different size, so their scales differ by orders of
    magnitude and no single learning rate suits all of them. Measuring
    mean and spread once, here, is what lets the k-space coefficients
    share MACE's optimizer and learning rate; the mapping is a change of
    variables, so the model it can express is unchanged.

    Runs under ``no_grad`` over ``data_loader`` (all of it unless
    ``max_batches`` is given) and writes the two buffers on ``block``.
    MACE builds its training loader with ``drop_last``, so the last
    partial batch (up to batch_size - 1 frames, which ones depends on
    the seed) is left out of the statistics; at 1219 training frames
    and batch 10 that shifts the mean by about 0.04 meV/atom.
    ``device`` defaults to the one the block already sits on, which is
    where its buffers are: at build time that is still the CPU.
    """
    if device is None:
        device = block.energy.feature_scale.device
    sums = torch.zeros(block.n_features, dtype=torch.float64, device=device)
    sums_squared = torch.zeros_like(sums)
    count = 0

    with torch.no_grad():
        for index, batch in enumerate(data_loader):
            if max_batches is not None and index >= max_batches:
                break
            batch = batch.to(device)
            num_graphs = int(batch.ptr.numel() - 1)
            features = block.energy.raw_features(
                batch.positions,
                batch.batch,
                cubic_side_lengths(batch.cell, num_graphs),
                num_graphs,
                batch.edge_index,
                batch.shifts,
                BOHR_PER_ANG,
            ).to(torch.float64)
            sums += features.sum(dim=0)
            sums_squared += (features * features).sum(dim=0)
            count += num_graphs

    if count == 0:
        raise ValueError(
            "no configurations to measure the k-space features on: the "
            "training loader is empty"
        )

    mean = sums / count
    variance = torch.clamp(sums_squared / count - mean * mean, min=0.0)
    spread = torch.sqrt(variance)
    # A feature that does not vary over the training set carries no
    # information. Scaling it by 1 leaves it as it is rather than
    # dividing by zero and publishing a model that predicts NaN.
    scale = torch.where(spread > 0, spread, torch.ones_like(spread))
    n_constant = int((spread <= 0).sum())
    if n_constant:
        logging.warning(
            f"{n_constant} of {block.n_features} k-space features are "
            f"constant over the training set and were left unscaled"
        )

    block.energy.set_feature_normalization(mean, scale)
    logging.info(
        f"KML k-space features standardized on {count} configurations "
        f"(spread {float(spread.min()):.3e} to {float(spread.max()):.3e})"
    )


class KMLMACE(ScaleShiftMACE):
    """ScaleShiftMACE plus the KML plane-wave k-space energy.

    One model, one loss, one optimizer step: the k-space coefficients
    are parameters like any other, and the forces are the derivative of
    the sum, so neither term can absorb the other's error without
    paying for it in the same loss.

    ``kml_kspace_arguments`` is forwarded to
    :class:`KSpaceLongRangeBlock` (``nup``, ``k2cut``, ``k2cut_pair``,
    ``k2cut_triplet``, ``include_constant``).

    Energy, forces and hessian only. Stress, virials, edge forces and
    the LAMMPS path are refused rather than computed without the
    k-space contribution: the k-space energy depends on the cell
    through ``ell``, so a cell derivative that ignored it would be
    wrong, not approximate. ``MACECalculator`` asks for a stress, so it
    cannot drive one of these models either.

    ``forward`` delegates to the parent and adds a term, which
    TorchScript cannot compile (``super().forward`` is unsupported).
    Training does not care -- the run logs that it skipped the optional
    ``*_compiled.model``, an artifact MACE v1.0 removes anyway.
    """

    def __init__(self, kml_kspace_arguments: Optional[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.kml_kspace = KSpaceLongRangeBlock(**dict(kml_kspace_arguments or {}))

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Written out rather than built as a filtered list so the forward
        # stays TorchScript-compatible: a comprehension with an `if` is
        # what stopped MACE writing the compiled model.
        refused = ""
        if compute_virials:
            refused = "compute_virials"
        elif compute_stress:
            refused = "compute_stress"
        elif compute_atomic_stresses:
            refused = "compute_atomic_stresses"
        elif compute_displacement:
            refused = "compute_displacement"
        elif compute_edge_forces:
            refused = "compute_edge_forces"
        elif lammps_mliap:
            refused = "lammps_mliap"
        if refused != "":
            raise NotImplementedError(
                "KMLMACE does not support " + refused + ". Every quantity in "
                "that list is a derivative with respect to the cell or the "
                "edge vectors, and the k-space energy is a function of the "
                "positions and the box directly. Train with an energy-and-"
                "forces loss (--loss weighted, the default); see "
                "mace/modules/kml_kspace.py for what a stress would take."
            )

        # The short-range model, without its force pass: the graph is
        # still live in this scope, so the forces below are taken once,
        # from the sum. Asking twice would give the k-space term no say
        # in them.
        outputs = super().forward(
            data,
            training=training,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            compute_hessian=False,
            compute_edge_forces=False,
            compute_atomic_stresses=False,
            lammps_mliap=False,
        )

        positions = data["positions"]
        num_graphs = int(data["ptr"].numel() - 1)
        kspace_energy = self.kml_kspace(
            positions, data["batch"], data["cell"], num_graphs,
            data["edge_index"], data["shifts"],
        )

        interaction_energy = outputs["interaction_energy"] + kspace_energy
        forces, _, _, hessian, _, _ = get_outputs(
            energy=interaction_energy,
            positions=positions,
            displacement=None,
            vectors=None,
            cell=data["cell"],
            pbc=data.get("pbc"),
            training=training,
            compute_force=compute_force,
            compute_virials=False,
            compute_stress=False,
            compute_hessian=compute_hessian,
            compute_edge_forces=False,
        )

        outputs.update(
            energy=outputs["energy"] + kspace_energy,
            interaction_energy=interaction_energy,
            kspace_energy=safe_double(kspace_energy),
            forces=forces,
            hessian=hessian,
        )
        # node_energy stays the short-range decomposition: the k-space
        # term is a property of the configuration, not a sum over atoms,
        # and splitting it evenly would invent a per-atom value.
        return outputs
