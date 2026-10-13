"""Water entropy aggregation.

This module wraps the waterEntropy routines and maps their
outputs into the project `ResultsReporter` format.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import waterEntropy.recipes.bulk_water as BulkWater
import waterEntropy.recipes.interfacial_solvent as GetSolvent

logger = logging.getLogger(__name__)


def _plain(value: Any) -> Any:
    """Convert NumPy scalars to built-in Python types for JSON export."""
    return value.item() if isinstance(value, np.generic) else value


def _input_composition(universe: Any) -> dict[str, Any]:
    """Describe the Universe actually handed to the water solver.

    Recorded so downstream tools can enforce provenance (e.g. that a bulk reference
    came from a water-only simulation) instead of trusting file names.
    """
    trajectory = universe.trajectory
    non_water = universe.select_atoms("not water").residues
    return {
        "n_atoms": int(len(universe.atoms)),
        "n_water_residues": int(len(universe.select_atoms("water").residues)),
        "n_non_water_residues": int(len(non_water)),
        "non_water_resnames": sorted({str(name) for name in non_water.resnames}),
        "non_water_resname_counts": dict(
            sorted(Counter(str(name) for name in non_water.resnames).items())
        ),
        "topology_file": _plain_str(getattr(universe, "filename", None)),
        "trajectory_file": _plain_str(getattr(trajectory, "filename", None)),
        "trajectory_n_frames": int(len(trajectory)),
    }


def _plain_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _ambiguous_residue_keys(universe: Any) -> list[str]:
    """Return non-water ``RESNAME_RESID`` keys shared by more than one residue.

    waterEntropy identifies solute residues by resname and resid only, so such
    residues (e.g. repeated resids across chains) are merged by the solver and
    cannot be separated afterwards.
    """
    seen: dict[str, int] = {}
    for residue in universe.select_atoms("not water").residues:
        key = f"{residue.resname}_{int(residue.resid)}"
        seen[key] = seen.get(key, 0) + 1
    return sorted(k for k, n in seen.items() if n > 1)


@dataclass(frozen=True)
class WaterEntropyInput:
    """Inputs for water entropy computation.

    Attributes:
        universe: MDAnalysis Universe containing the system.
        start: Start frame index (inclusive).
        end: End frame index (exclusive, or -1 depending on caller convention).
        step: Frame stride.
        temperature: Temperature in Kelvin.
        group_id: Group ID used for logging.
    """

    universe: Any
    start: int
    end: int
    step: int
    temperature: float
    group_id: int | None = None


class WaterEntropy:
    """Compute and log water entropy contributions.

    This class calls the external `waterEntropy` routine to compute:
      - orientational entropy per residue
      - translational vibrational entropy
      - rotational vibrational entropy

    Then it logs residue-level entries and adds a group label.
    """

    def __init__(
        self,
        args: Any,
        reporter: Any,
        solver: Callable[..., tuple[dict, Any, Any, Any, Any]] = (
            GetSolvent.get_interfacial_water_orient_entropy
        ),
        bulk_solver: Callable[..., tuple[dict, Any, Any]] = (
            BulkWater.get_bulk_water_orient_entropy
        ),
    ) -> None:
        """Initialize the water entropy calculator.

        Args:
            args: Argument namespace; must include `temperature`.
            reporter: Logger used to record residue and group results.
            solver: Callable compatible with
                `get_interfacial_water_orient_entropy
                (universe, start, end, step, temperature, parallel=True)`.
                Dependency injection allows unit testing without the external package.
            bulk_solver: Callable compatible with waterEntropy's
                `get_bulk_water_orient_entropy(universe, start, end, step, temperature)`.
        """
        self._args = args
        self._reporter = reporter
        self._bulk_solver = bulk_solver
        self._solver = solver

    def calculate_and_log(
        self,
        universe: Any,
        start: int,
        end: int,
        step: int,
        group_id: int | None = None,
    ) -> None:
        """Compute water entropy and write results to the data logger.

        Args:
            universe: MDAnalysis Universe containing water.
            start: Start frame index.
            end: End frame index.
            step: Frame stride.
            group_id: Group ID to assign all water contributions to.
        """
        inputs = WaterEntropyInput(
            universe=universe,
            start=start,
            end=end,
            step=step,
            temperature=float(self._args.temperature),
            group_id=group_id,
        )
        self._calculate_and_log_from_inputs(inputs)

    def _calculate_and_log_from_inputs(self, inputs: WaterEntropyInput) -> None:
        """Run the solver and log all returned entropy components."""
        Sorient_dict, covariances, vibrations, _unused, n_frames = self._run_solver(
            inputs
        )

        self._record_export(inputs, Sorient_dict, covariances, vibrations, n_frames)
        self._log_orientational_entropy(Sorient_dict, inputs.group_id)
        self._log_translational_entropy(vibrations, covariances, inputs.group_id)
        self._log_rotational_entropy(vibrations, covariances, inputs.group_id)
        self._log_group_label(inputs.universe, Sorient_dict, inputs.group_id)

    def _run_solver(self, inputs: WaterEntropyInput):
        """Call the external solver.

        Args:
            inputs: WaterEntropyInput.

        Returns:
            Tuple of solver outputs.
        """
        logger.info(
            "[WaterEntropy] Computing water entropy (start=%s, end=%s, step=%s)",
            inputs.start,
            inputs.end,
            inputs.step,
        )
        parallel = bool(
            getattr(self._args, "parallel_frames", False)
            or getattr(self._args, "use_dask", False)
            or getattr(self._args, "hpc", False)
        )
        solver_args = (inputs.universe, inputs.start, inputs.end, inputs.step,
                       inputs.temperature)
        if not parallel:
            return self._solver(*solver_args, parallel=False)

        if getattr(self._args, "hpc", False):
            from CodeEntropy.core.dask_clusters import HPCDaskManager
            client = HPCDaskManager(self._args).configure_cluster()
        else:
            from dask.distributed import Client
            client = Client(
                processes=True,
                n_workers=getattr(self._args, "dask_workers", None),
                threads_per_worker=1,
            )
        try:
            return self._solver(*solver_args, parallel=True, client=client)
        finally:
            client.close()

    # Positional meaning of each waterEntropy Orientations summary list.
    ORIENTATIONAL_FIELDS = (
        "S_orient_hb_biased",
        "observations",
        "N_c_mean",
        "N_w_mean",
        "Nc_eff_mean",
        "pbias_mean",
        "S_orient_non_hb_Nc",
        "S_orient_non_hb_Nw",
    )

    # Exact meaning of each orientational field, as computed by waterEntropy.
    ORIENTATIONAL_FIELD_DEFINITIONS = {
        "S_orient_hb_biased": (
            "HB-biased orientational entropy: R ln(Nc_eff^1.5 pi^0.5 pbias/2) per "
            "labelled-shell type, clipped at 0, count-weighted mean over shell types"
        ),
        "observations": "water observations summed over analysed frames",
        "N_c_mean": "count-weighted mean of len(shell labels): all RAD neighbours "
        "of the water, solute contacts included",
        "N_w_mean": "count-weighted mean of N_w: RAD neighbours with the central "
        "water's resname only (per labelled-shell type, first value seen)",
        "Nc_eff_mean": "count-weighted mean effective HB neighbour number",
        "pbias_mean": "count-weighted mean HB donor/acceptor bias",
        "S_orient_non_hb_Nc": (
            "R ln(N^1.5 pi^0.5 0.25/2) with N = len(shell labels) (all neighbours), "
            "p_corr=0.25, sigma=2, clipped at 0, evaluated per labelled-shell type then "
            "count-weighted averaged (not evaluated at the mean N)"
        ),
        "S_orient_non_hb_Nw": (
            "as S_orient_non_hb_Nc but with N = N_w (water neighbours only)"
        ),
    }

    def _build_export(
        self,
        inputs: WaterEntropyInput,
        Sorient_dict: Mapping[Any, Mapping[str, Any]],
        covariances: Any,
        vibrations: Any,
        n_frames: int,
        source: str,
    ) -> dict[str, Any]:
        """Build the complete, explicitly labelled record of one solver result.

        The residue table rows keep their historical (lossy) content; this export
        names every orientational field, keeps per-molecule vibrational entropies
        and raw observation counts, and never invents a missing count.
        ``mean_waters`` is ``observations / n_frames``: the average number of waters
        per analysed frame attributed to that key.

        Args:
            inputs: Water entropy inputs for this group.
            Sorient_dict: Solver orientational summaries, key -> resname -> fields.
            covariances: Solver covariance collection (only ``counts`` is read;
                the matrices have been rescaled in place by the solver).
            vibrations: Solver vibrations with ``translational_S``/``rotational_S``.
            n_frames: Number of frames the solver analysed.
            source: Human-readable description of the solver recipe used.

        Returns:
            JSON-serialisable export dictionary.
        """
        n_frames = int(n_frames)
        orientational = []
        for resid, by_resname in Sorient_dict.items():
            for resname, values in by_resname.items():
                values = list(values)
                entry = {"nearest_resid": _plain(resid), "nearest_resname": str(resname)}
                for name, value in zip(self.ORIENTATIONAL_FIELDS, values, strict=False):
                    entry[name] = _plain(value)
                if len(values) > len(self.ORIENTATIONAL_FIELDS):
                    entry["unlabelled_fields"] = [
                        _plain(v) for v in values[len(self.ORIENTATIONAL_FIELDS) :]
                    ]
                if "observations" in entry and n_frames:
                    entry["mean_waters"] = entry["observations"] / n_frames
                orientational.append(entry)

        counts = getattr(covariances, "counts", {}) or {}
        translational = getattr(vibrations, "translational_S", {}) or {}
        rotational = getattr(vibrations, "rotational_S", {}) or {}
        vibrational = []
        for key in translational:
            solute_key, solvent_name = key
            s_trans = np.atleast_1d(np.asarray(translational[key], dtype=float))
            s_rot = np.atleast_1d(np.asarray(rotational.get(key, []), dtype=float))
            observations = int(counts[key]) if key in counts else None
            resname, _, resid = str(solute_key).rpartition("_")
            vibrational.append(
                {
                    "solute_key": str(solute_key),
                    "nearest_resname": resname or str(solute_key),
                    # Keys without "RESNAME_RESID" form (bulk recipe: "SOL") use the
                    # key itself as resid, matching its orientational entry
                    # Sorient_dict["SOL"]["SOL"].
                    "nearest_resid": resid if resname else str(solute_key),
                    "solvent_resname": str(solvent_name),
                    "S_trans": float(s_trans.sum()),
                    "S_rot": float(s_rot.sum()),
                    "S_trans_modes": s_trans.tolist(),
                    "S_rot_modes": s_rot.tolist(),
                    "observations": observations,
                    "mean_waters": (
                        observations / n_frames
                        if observations is not None and n_frames
                        else None
                    ),
                }
            )

        return {
            "schema": "codeentropy.water_entropy_export/4",
            "source": source,
            "entropy_units": "J/mol/K per water molecule",
            "start": inputs.start,
            "end": inputs.end,
            "step": inputs.step,
            "temperature": inputs.temperature,
            "n_frames": n_frames,
            "field_definitions": dict(self.ORIENTATIONAL_FIELD_DEFINITIONS),
            "ambiguous_residue_keys": _ambiguous_residue_keys(inputs.universe),
            "input_composition": _input_composition(inputs.universe),
            "orientational": orientational,
            "vibrational": vibrational,
        }

    def _record_export(
        self,
        inputs: WaterEntropyInput,
        Sorient_dict: Mapping[Any, Mapping[str, Any]],
        covariances: Any,
        vibrations: Any,
        n_frames: int,
    ) -> None:
        """Store the interfacial (first-shell) export for this water group."""
        export = self._reporter_export("water_entropy_export")
        export[str(inputs.group_id)] = self._build_export(
            inputs,
            Sorient_dict,
            covariances,
            vibrations,
            n_frames,
            source="waterEntropy interfacial (RAD first-shell) solvent",
        )

    def _reporter_export(self, name: str) -> dict[str, Any]:
        export = getattr(self._reporter, name, None)
        if export is None:
            export = {}
            setattr(self._reporter, name, export)
        return export

    def calculate_bulk_export(
        self,
        universe: Any,
        start: int,
        end: int,
        step: int,
        group_id: int | None = None,
    ) -> None:
        """Compute bulk water entropy with the waterEntropy bulk recipe and export it.

        Intended for the water-only simulation: waters whose whole RAD shell is
        water are analysed with the same force/torque, covariance, vibration and
        orientation classes as first-shell waters. Nothing is logged to the result
        tables; the export is the only output.

        Args:
            universe: MDAnalysis Universe of the water-only system.
            start: Start frame index.
            end: Stop frame index (exclusive).
            step: Frame stride.
            group_id: Water group identifier used as the export key.
        """
        inputs = WaterEntropyInput(
            universe=universe,
            start=start,
            end=end,
            step=step,
            temperature=float(self._args.temperature),
            group_id=group_id,
        )
        logger.info(
            "[WaterEntropy] Computing bulk water entropy (start=%s, end=%s, step=%s)",
            start,
            end,
            step,
        )
        Sorient_dict, covariances, vibrations = self._bulk_solver(
            universe, start, end, step, inputs.temperature
        )
        n_frames = len(range(*slice(start, end, step).indices(len(universe.trajectory))))
        export = self._reporter_export("water_entropy_bulk_export")
        export[str(group_id)] = self._build_export(
            inputs,
            Sorient_dict,
            covariances,
            vibrations,
            n_frames,
            source="waterEntropy bulk water (RAD shell of water only)",
        )

    def _log_orientational_entropy(
        self, Sorient_dict: Mapping[Any, Mapping[str, Any]], group_id: int | None
    ) -> None:
        """Log orientational entropy entries.

        Args:
            Sorient_dict: Mapping of residue ids to {resname: [entropy, count]}.
            group_id: Group ID to assign logs to.
        """
        for _resid, resname_dict in Sorient_dict.items():
            for resname, values in resname_dict.items():
                if isinstance(values, list) and len(values) == 2:
                    entropy, count = values
                    self._reporter.add_residue_data(
                        group_id, resname, "Water", "Orientational", count, entropy
                    )

    def _log_translational_entropy(
        self, vibrations: Any, covariances: Any, group_id: int | None
    ) -> None:
        """Log translational vibrational entropy entries.

        Args:
            vibrations: Solver vibrations object with `translational_S`.
            covariances: Solver covariances object with `counts`.
            group_id: Group ID to assign logs to.
        """
        translational = getattr(vibrations, "translational_S", {}) or {}
        counts = getattr(covariances, "counts", {}) or {}

        for (solute_id, _), entropy in translational.items():
            value = (
                float(np.sum(entropy))
                if isinstance(entropy, (list, np.ndarray))
                else float(entropy)
            )
            count = counts.get((solute_id, "WAT"), 1)
            resname = self._solute_id_to_resname(solute_id)
            self._reporter.add_residue_data(
                group_id, resname, "Water", "Transvibrational", count, value
            )

    def _log_rotational_entropy(
        self, vibrations: Any, covariances: Any, group_id: int | None
    ) -> None:
        """Log rotational vibrational entropy entries.

        Args:
            vibrations: Solver vibrations object with `rotational_S`.
            covariances: Solver covariances object with `counts`.
            group_id: Group ID to assign logs to.
        """
        rotational = getattr(vibrations, "rotational_S", {}) or {}
        counts = getattr(covariances, "counts", {}) or {}

        for (solute_id, _), entropy in rotational.items():
            value = (
                float(np.sum(entropy))
                if isinstance(entropy, (list, np.ndarray))
                else float(entropy)
            )
            count = counts.get((solute_id, "WAT"), 1)
            resname = self._solute_id_to_resname(solute_id)
            self._reporter.add_residue_data(
                group_id, resname, "Water", "Rovibrational", count, value
            )

    def _log_group_label(
        self,
        universe: Any,
        Sorient_dict: Mapping[Any, Mapping[str, Any]],
        group_id: int | None,
    ) -> None:
        """Log a group label summarizing the water entries.

        Args:
            universe: MDAnalysis Universe.
            Sorient_dict: Orientational entropy dict used to infer residue names.
            group_id: Group ID.
        """
        water_selection = universe.select_atoms("resname WAT")
        actual_water_residues = len(water_selection.residues)

        water_resnames = set(water_selection.residues.resnames)
        residue_names = {
            resname
            for res_dict in Sorient_dict.values()
            for resname in res_dict.keys()
            if str(resname).upper() in {str(r).upper() for r in water_resnames}
        }

        residue_group = "_".join(sorted(residue_names)) if residue_names else "WAT"
        self._reporter.add_group_label(
            group_id, residue_group, actual_water_residues, len(water_selection.atoms)
        )

    @staticmethod
    def _solute_id_to_resname(solute_id: str) -> str:
        """Convert a solver solute_id to a residue-like name.

        Args:
            solute_id: Identifier returned by the solver.

        Returns:
            Residue name string.
        """
        if "_" in str(solute_id):
            return str(solute_id).rsplit("_", 1)[0]
        return str(solute_id)
