"""Assemble a binding entropy from separate CodeEntropy runs (EE-MCC bookkeeping).

Follows the four-system scheme of Ali et al., J. Comput.-Aided Mol. Des. 35, 911
(2021): complex, receptor and ligand each in water, plus a water-only box.

Only first-shell (RAD) water is counted explicitly. Water released on binding is
costed at the bulk per-molecule entropy s_bulk, with the number released fixed by
stoichiometry (paper Eq. 3):

    N_WB = N_WS,receptor + N_WS,ligand - N_WS,complex

Because N_WB is linear in the shell counts, each system is reduced to one scalar
per replicate and the binding entropy is their difference:

    X_complex  = S_receptor(bound) + S_ligand(bound) + S_or,bound
                 + S_WS,complex - N_WS,complex * s_bulk
    X_receptor = S_receptor + S_WS,receptor - N_WS,receptor * s_bulk
    X_ligand   = S_ligand + S_pos + S_or,ligand + S_WS,ligand - N_WS,ligand * s_bulk
    dS         = X_complex - X_receptor - X_ligand

Receptor molecule-level positional/orientational entropy is the reference frame and
is excluded on both sides, as in the paper.

Uncertainty is first-order propagation conditional on the estimated bulk mean:
per-system SEMs over replicates are combined (in quadrature for independent
replicates, or from per-index differences when paired), plus (N_WB * SEM(s_bulk))^2.
It is not an exact variance for the product of a random shell count and a random
bulk entropy. An undefined SEM (fewer than 2 replicates) propagates as NaN.

Every scientific choice that differs between host-guest and protein-ligand systems,
or that the paper leaves open, must be given explicitly in the configuration;
nothing is defaulted. Run provenance is checked from each export's recorded input
composition: the bulk reference must come from a water-only input; each solvated run
must match its role using the explicit receptor_resnames and ligand_resname; and
receptor and ligand residue counts must agree across the runs that share them.

Usage:
    python -m CodeEntropy.results.binding_entropy config.json [-o result.json]
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GAS_CONSTANT = 8.314  # J/(K mol), same constant as waterEntropy
J_PER_CAL = 4.184

WATER_ORIENTATIONAL_FIELDS = (
    "S_orient_hb_biased",
    "S_orient_non_hb_Nc",
    "S_orient_non_hb_Nw",
)
# waterentropy_bulk_export: same waterEntropy recipe family as the first-shell terms,
# accepted only from inputs recorded as water-only. mcc_water_group: generic MCC
# water group (orientational term lacks p_corr); kept for diagnostics only and
# reported with bulk_source_matched_kernel = False.
BULK_WATER_SOURCES = ("waterentropy_bulk_export", "mcc_water_group")
REQUIRED_CHOICES = (
    "water_orientational_field",
    "bulk_water_source",
    "unbound_ligand_orientational",
    "bound_ligand_orientational_J_per_molK",
    "ligand_positional",
    "shell_scope",
    "replicates",
    "population_tolerance",
)


class BindingEntropyError(ValueError):
    """Raised for invalid configuration or inconsistent run outputs."""


@dataclass(frozen=True)
class Shell:
    """First-shell water totals for one run, per mole of system."""

    n_waters: float
    s_trans: float
    s_rot: float
    s_orient: float

    @property
    def total(self) -> float:
        return self.s_trans + self.s_rot + self.s_orient


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def group_components(run: Mapping[str, Any], group_id: str, path: str) -> dict[str, float]:
    groups = run.get("groups", {})
    if str(group_id) not in groups:
        raise BindingEntropyError(
            f"{path}: group {group_id!r} not found (available: {sorted(groups)})"
        )
    return {k: float(v) for k, v in groups[str(group_id)]["components"].items()}


def split_orientational(components: Mapping[str, float]) -> tuple[float, float]:
    """Return (sum of non-orientational components, sum of orientational components)."""
    other = sum(v for k, v in components.items() if not k.endswith(":Orientational"))
    orient = sum(v for k, v in components.items() if k.endswith(":Orientational"))
    return other, orient


def _entry_key(entry: Mapping[str, Any]) -> tuple[str, str]:
    return str(entry["nearest_resname"]), str(entry["nearest_resid"])


def _in_scope(key: tuple[str, str], scope: Any, ligand_resname: str) -> bool:
    if scope == "all":
        return True
    if key[0] == ligand_resname:
        return True
    return f"{key[0]}_{key[1]}" in scope["include"]


def single_export(run: Mapping[str, Any], path: str, name: str) -> dict[str, Any]:
    exports = run.get(name) or {}
    if len(exports) != 1:
        raise BindingEntropyError(
            f"{path}: expected exactly one '{name}' export, found {len(exports)}"
        )
    (export,) = exports.values()
    if not export.get("n_frames"):
        raise BindingEntropyError(f"{path}: '{name}' export has no analysed frames")
    return export


def input_composition(export: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    composition = export.get("input_composition")
    if not composition:
        raise BindingEntropyError(
            f"{path}: export has no input_composition (schema older than /3); rerun with "
            "a CodeEntropy build that records input provenance"
        )
    return composition


def role_counts(
    export: Mapping[str, Any],
    path: str,
    role: str,
    ligand_resname: str,
    receptor_resnames: frozenset[str],
) -> tuple[int, int]:
    """Check a solvated run's recorded input against its role; return residue counts.

    Only residue names given explicitly in the configuration count: receptor residues
    are those named in ``receptor_resnames`` and ligand residues those named
    ``ligand_resname``. Anything else (ions, co-solvents) is ignored for role purposes.

    Returns:
        (number of receptor residues, number of ligand residues) in the solver input.
    """
    composition = input_composition(export, path)
    counts = composition.get("non_water_resname_counts")
    if counts is None:
        raise BindingEntropyError(
            f"{path}: input_composition has no non_water_resname_counts (schema older "
            "than /4); rerun with a CodeEntropy build that records residue counts"
        )
    n_receptor = sum(int(n) for name, n in counts.items() if name in receptor_resnames)
    n_ligand = int(counts.get(ligand_resname, 0))
    ok, expected = {
        "complex": (n_receptor > 0 and n_ligand > 0, "receptor and ligand residues"),
        "receptor": (n_receptor > 0 and n_ligand == 0, "receptor residues and no ligand"),
        "ligand": (n_ligand > 0 and n_receptor == 0, "ligand residues and no receptor"),
    }[role]
    if not ok:
        raise BindingEntropyError(
            f"{path}: listed as a {role} run, but its input must contain {expected}; "
            f"found {n_receptor} receptor and {n_ligand} ligand residues "
            f"(non-water residue counts: {dict(counts)})"
        )
    return n_receptor, n_ligand


def water_export(run: Mapping[str, Any], path: str) -> dict[str, Any]:
    if not run.get("water_entropy"):
        raise BindingEntropyError(
            f"{path}: no water_entropy export. Solvated runs must use "
            "--water_entropy True with a CodeEntropy build that writes the export."
        )
    return single_export(run, path, "water_entropy")


def require_consistent(label: str, observed: Mapping[str, int]) -> None:
    """Require one residue count across runs that should share the same molecule."""
    if len(set(observed.values())) > 1:
        raise BindingEntropyError(
            f"{label} residue count differs between runs that must share it: {dict(observed)}"
        )


def _index_entries(entries, path: str, kind: str, required: Iterable[str]):
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for entry in entries:
        key = _entry_key(entry)
        missing = [f for f in required if entry.get(f) is None]
        if missing:
            raise BindingEntropyError(f"{path}: {kind} entry {key} lacks {missing}")
        if key in index:
            raise BindingEntropyError(f"{path}: duplicate {kind} entry for {key}")
        index[key] = entry
    return index


def export_totals(
    export: Mapping[str, Any],
    path: str,
    *,
    orient_field: str,
    scope: Any,
    ligand_resname: str,
    tolerance: float,
) -> Shell:
    """Sum per-molecule water entropies weighted by mean water counts per key."""
    solvents = {str(e.get("solvent_resname")) for e in export["vibrational"]}
    if len(solvents) > 1:
        raise BindingEntropyError(
            f"{path}: multiple solvent populations {sorted(solvents)} are not supported"
        )
    vib = _index_entries(
        export["vibrational"], path, "vibrational", ("mean_waters", "S_trans", "S_rot")
    )
    orient = _index_entries(
        export["orientational"], path, "orientational", ("mean_waters", orient_field)
    )
    keys = [k for k in sorted(set(vib) | set(orient)) if _in_scope(k, scope, ligand_resname)]
    ambiguous = set(export.get("ambiguous_residue_keys") or [])
    clashes = sorted(f"{k[0]}_{k[1]}" for k in keys if f"{k[0]}_{k[1]}" in ambiguous)
    if clashes:
        raise BindingEntropyError(
            f"{path}: shell keys {clashes} identify more than one residue (repeated "
            "resname/resid); the solver merged them and they cannot be separated"
        )
    n_vib = s_trans = s_rot = s_orient = 0.0
    for key in keys:
        if key not in vib or key not in orient:
            raise BindingEntropyError(
                f"{path}: shell {key} present in only one of vibrational/orientational"
            )
        nv, no = vib[key]["mean_waters"], orient[key]["mean_waters"]
        if abs(nv - no) > tolerance * max(1.0, abs(nv)):
            raise BindingEntropyError(
                f"{path}: shell {key} population mismatch: vibrational {nv} vs "
                f"orientational {no}"
            )
        n_vib += nv
        s_trans += nv * vib[key]["S_trans"]
        s_rot += nv * vib[key]["S_rot"]
        s_orient += no * orient[key][orient_field]
    return Shell(n_waters=n_vib, s_trans=s_trans, s_rot=s_rot, s_orient=s_orient)


def shell_totals(
    run: Mapping[str, Any],
    path: str,
    role: str,
    receptor_resnames: frozenset[str],
    counts: dict[str, dict[str, int]],
    **kwargs: Any,
) -> Shell:
    """Role-checked first-shell water totals of a solvated run.

    Records this run's receptor and ligand residue counts in ``counts`` for the
    cross-run consistency checks.
    """
    export = water_export(run, path)
    n_receptor, n_ligand = role_counts(
        export, path, role, kwargs["ligand_resname"], receptor_resnames
    )
    if role in ("complex", "receptor"):
        counts["receptor"][path] = n_receptor
    if role in ("complex", "ligand"):
        counts["ligand"][path] = n_ligand
    return export_totals(export, path, **kwargs)


def bulk_per_molecule(run: Mapping[str, Any], path: str, choices, water_gid, tol) -> float:
    """Bulk water entropy per molecule from one water-only run."""
    if choices["bulk_water_source"] == "mcc_water_group":
        return sum(group_components(run, water_gid, path).values())
    export = single_export(run, path, "water_entropy_bulk")
    composition = input_composition(export, path)
    if composition.get("n_non_water_residues") != 0 or not composition.get("n_water_residues"):
        raise BindingEntropyError(
            f"{path}: bulk water must come from a water-only simulation, but the solver input "
            f"had {composition.get('n_non_water_residues')} non-water residues "
            f"({composition.get('non_water_resnames')}) and "
            f"{composition.get('n_water_residues')} water residues"
        )
    totals = export_totals(
        export, path, orient_field=choices["water_orientational_field"], scope="all",
        ligand_resname="", tolerance=tol,
    )
    if totals.n_waters <= 0:
        raise BindingEntropyError(f"{path}: bulk export contains no water observations")
    return totals.total / totals.n_waters


def ligand_orientational(choices: Mapping[str, Any], ligand_components, shell: Shell) -> float:
    mode = choices["unbound_ligand_orientational"]
    if mode == "codeentropy":
        return split_orientational(ligand_components)[1]
    if isinstance(mode, Mapping) and mode.get("method") == "paper_eq7":
        sigma = float(mode["symmetry_number"])
        p_corr = float(mode["p_corr"])
        n_c = shell.n_waters
        return GAS_CONSTANT * math.log(n_c**1.5 * math.pi**0.5 * p_corr / sigma)
    raise BindingEntropyError(
        "unbound_ligand_orientational must be 'codeentropy' or "
        "{'method': 'paper_eq7', 'symmetry_number': ..., 'p_corr': ...}"
    )


def ligand_positional(choices: Mapping[str, Any]) -> float:
    value = choices["ligand_positional"]
    if value == "paper_1M":
        return GAS_CONSTANT * math.log(55.5)
    if isinstance(value, (int, float)):
        return float(value)
    raise BindingEntropyError("ligand_positional must be 'paper_1M' or a number (J/mol/K)")


def run_temperature(run: Mapping[str, Any]) -> float | None:
    temp = run.get("args", {}).get("temperature")
    return None if temp is None else float(temp)


def mean_sem(values: Sequence[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def _runs(config: Mapping[str, Any], system: str) -> list[tuple[str, dict[str, Any]]]:
    paths = config.get(system, {}).get("runs") or []
    if not paths:
        raise BindingEntropyError(f"no runs given for system {system!r}")
    return [(str(p), load_json(p)) for p in paths]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(
        value
    )


def validate_choices(choices: Mapping[str, Any]) -> None:
    """Reject missing or placeholder scientific choices before any run is read."""
    missing = [c for c in REQUIRED_CHOICES if c not in choices]
    if missing:
        raise BindingEntropyError(f"missing required choices: {missing}")
    if choices["water_orientational_field"] not in WATER_ORIENTATIONAL_FIELDS:
        raise BindingEntropyError(
            f"water_orientational_field must be one of {WATER_ORIENTATIONAL_FIELDS}"
        )
    if choices["bulk_water_source"] not in BULK_WATER_SOURCES:
        raise BindingEntropyError(f"bulk_water_source must be one of {BULK_WATER_SOURCES}")
    mode = choices["unbound_ligand_orientational"]
    if mode != "codeentropy" and not (
        isinstance(mode, Mapping)
        and mode.get("method") == "paper_eq7"
        and _is_number(mode.get("symmetry_number"))
        and mode["symmetry_number"] > 0
        and _is_number(mode.get("p_corr"))
        and mode["p_corr"] > 0
    ):
        raise BindingEntropyError(
            "unbound_ligand_orientational must be 'codeentropy' or {'method': "
            "'paper_eq7', 'symmetry_number': >0, 'p_corr': >0}"
        )
    if not _is_number(choices["bound_ligand_orientational_J_per_molK"]):
        raise BindingEntropyError("bound_ligand_orientational_J_per_molK must be a number")
    if choices["ligand_positional"] != "paper_1M" and not _is_number(
        choices["ligand_positional"]
    ):
        raise BindingEntropyError("ligand_positional must be 'paper_1M' or a number (J/mol/K)")
    scope = choices["shell_scope"]
    if scope != "all" and not (
        isinstance(scope, Mapping)
        and isinstance(scope.get("include"), list)
        and all(isinstance(k, str) for k in scope["include"])
    ):
        raise BindingEntropyError("shell_scope must be 'all' or {'include': ['RES_ID', ...]}")
    if choices["replicates"] not in ("independent", "paired"):
        raise BindingEntropyError("replicates must be 'independent' or 'paired'")
    if not _is_number(choices["population_tolerance"]) or choices["population_tolerance"] < 0:
        raise BindingEntropyError("population_tolerance must be a non-negative number")


def assemble(config: Mapping[str, Any]) -> dict[str, Any]:
    """Compute the binding entropy and its decomposition from a configuration."""
    choices = config.get("choices", {})
    validate_choices(choices)
    orient_field = choices["water_orientational_field"]
    scope = choices["shell_scope"]
    if scope != "all":
        scope = {"include": set(scope["include"])}
    ligand_resname = config.get("ligand_resname")
    if not ligand_resname or not isinstance(ligand_resname, str):
        raise BindingEntropyError("ligand_resname is required")
    receptor_names = config.get("receptor_resnames")
    if not (
        isinstance(receptor_names, list)
        and receptor_names
        and all(isinstance(n, str) and n and not n.startswith("CHOOSE") for n in receptor_names)
    ):
        raise BindingEntropyError(
            "receptor_resnames is required: a non-empty list of the receptor's residue names "
            "(e.g. its amino-acid and cap names); unlisted names such as ions are ignored"
        )
    if ligand_resname in receptor_names:
        raise BindingEntropyError("receptor_resnames must not contain ligand_resname")
    receptor_resnames = frozenset(receptor_names)
    residue_counts: dict[str, dict[str, int]] = {"receptor": {}, "ligand": {}}
    tol = float(choices["population_tolerance"])
    shell_kw = dict(
        orient_field=orient_field, scope=scope, ligand_resname=ligand_resname, tolerance=tol
    )

    runs = {s: _runs(config, s) for s in ("complex", "receptor", "ligand", "water")}
    temps = {t for rs in runs.values() for _, r in rs if (t := run_temperature(r)) is not None}
    if len(temps) > 1:
        raise BindingEntropyError(f"runs use different temperatures: {sorted(temps)}")
    temperature = temps.pop() if temps else float(config["temperature"])

    water_gid = config["water"].get("water_group")
    bulk_values = [
        bulk_per_molecule(run, path, choices, water_gid, tol) for path, run in runs["water"]
    ]
    s_bulk, s_bulk_sem = mean_sem(bulk_values)

    s_pos = ligand_positional(choices)
    s_or_bound = float(choices["bound_ligand_orientational_J_per_molK"])

    per_run: dict[str, list[dict[str, float]]] = {"complex": [], "receptor": [], "ligand": []}
    for path, run in runs["complex"]:
        rec = split_orientational(
            group_components(run, config["complex"]["receptor_group"], path)
        )[0]
        lig = split_orientational(
            group_components(run, config["complex"]["ligand_group"], path)
        )[0]
        shell = shell_totals(run, path, "complex", receptor_resnames, residue_counts, **shell_kw)
        per_run["complex"].append(
            dict(solutes=rec + lig + s_or_bound, shell=shell.total, n_ws=shell.n_waters,
                 X=rec + lig + s_or_bound + shell.total - shell.n_waters * s_bulk)
        )
    for path, run in runs["receptor"]:
        rec = split_orientational(
            group_components(run, config["receptor"]["receptor_group"], path)
        )[0]
        shell = shell_totals(run, path, "receptor", receptor_resnames, residue_counts, **shell_kw)
        per_run["receptor"].append(
            dict(solutes=rec, shell=shell.total, n_ws=shell.n_waters,
                 X=rec + shell.total - shell.n_waters * s_bulk)
        )
    for path, run in runs["ligand"]:
        comps = group_components(run, config["ligand"]["ligand_group"], path)
        shell = shell_totals(run, path, "ligand", receptor_resnames, residue_counts, **shell_kw)
        s_or = ligand_orientational(choices, comps, shell)
        solutes = split_orientational(comps)[0] + s_pos + s_or
        per_run["ligand"].append(
            dict(solutes=solutes, shell=shell.total, n_ws=shell.n_waters,
                 X=solutes + shell.total - shell.n_waters * s_bulk)
        )

    require_consistent("receptor", residue_counts["receptor"])
    require_consistent("ligand", residue_counts["ligand"])

    stats = {
        s: {k: mean_sem([r[k] for r in rows]) for k in ("X", "solutes", "shell", "n_ws")}
        for s, rows in per_run.items()
    }
    n_wb = stats["receptor"]["n_ws"][0] + stats["ligand"]["n_ws"][0] - stats["complex"]["n_ws"][0]
    bulk_sem_term = (n_wb * s_bulk_sem) ** 2

    if choices["replicates"] == "paired":
        lengths = {len(per_run[s]) for s in per_run}
        if len(lengths) != 1:
            raise BindingEntropyError(f"paired replicates need equal counts, got {lengths}")
        diffs = [
            c["X"] - r["X"] - lg["X"]
            for c, r, lg in zip(per_run["complex"], per_run["receptor"], per_run["ligand"])
        ]
        d_s, d_s_sem = mean_sem(diffs)
        d_s_sem = math.sqrt(d_s_sem**2 + bulk_sem_term)
    else:
        d_s = stats["complex"]["X"][0] - stats["receptor"]["X"][0] - stats["ligand"]["X"][0]
        d_s_sem = math.sqrt(
            sum(stats[s]["X"][1] ** 2 for s in per_run) + bulk_sem_term
        )

    def t_ds(value: float) -> dict[str, float]:
        kj = temperature * value / 1000.0
        return {"kJ_per_mol": kj, "kcal_per_mol": kj / J_PER_CAL}

    return {
        "schema": "codeentropy.binding_entropy/2",
        "bulk_source_matched_kernel": choices["bulk_water_source"]
        == "waterentropy_bulk_export",
        "bulk_source_note": (
            "True means bulk water uses the same waterEntropy recipe family and orientational "
            "field as the first-shell terms, from an input verified to contain no non-water "
            "residues. It does not by itself demonstrate compatibility with the paper."
        ),
        "uncertainty_method": (
            "first-order propagation conditional on mean bulk entropy; "
            + choices["replicates"]
            + " replicates; NaN when any SEM is undefined"
        ),
        "temperature_K": temperature,
        "choices": {k: (sorted(v) if isinstance(v, set) else v) for k, v in choices.items()},
        "bulk_water_S_per_molecule": {"mean": s_bulk, "sem": s_bulk_sem, "n": len(bulk_values)},
        "ligand_positional_J_per_molK": s_pos,
        "N_water_released": n_wb,
        "systems": {
            s: {
                "n_replicates": len(per_run[s]),
                **{k: {"mean": m, "sem": e} for k, (m, e) in stats[s].items()},
            }
            for s in per_run
        },
        "delta_S_J_per_molK": {"mean": d_s, "sem": d_s_sem},
        "T_delta_S": {"mean": t_ds(d_s), "sem": t_ds(d_s_sem)},
        "per_run": per_run,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="JSON configuration file")
    parser.add_argument("-o", "--output", help="write the full result as JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        result = assemble(load_json(args.config))
    except (BindingEntropyError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    ds, tds = result["delta_S_J_per_molK"], result["T_delta_S"]
    if not result["bulk_source_matched_kernel"]:
        print("WARNING: bulk water from the MCC water group does not use the first-shell "
              "waterEntropy kernel (its orientational term lacks p_corr); diagnostic use only")
    if any(sysres["n_replicates"] < 2 for sysres in result["systems"].values()):
        print("note: a system has fewer than 2 replicates; its SEM is undefined (nan)")
    print(f"N_water_released = {result['N_water_released']:.3f}")
    for name, sysres in result["systems"].items():
        x = sysres["X"]
        print(f"X_{name:<9} = {x['mean']:.3f} +/- {x['sem']:.3f} J/mol/K "
              f"(n={sysres['n_replicates']})")
    print(f"dS   = {ds['mean']:.3f} +/- {ds['sem']:.3f} J/mol/K")
    print(f"TdS  = {tds['mean']['kcal_per_mol']:.3f} +/- {tds['sem']['kcal_per_mol']:.3f} "
          f"kcal/mol at {result['temperature_K']} K")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
