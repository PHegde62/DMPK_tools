"""
qc_engine.py
Orchestrates QC: loops over all rules in qc_rules.yaml, dispatches to
rule_evaluator, collects results, and returns a structured QC summary.
"""

from __future__ import annotations
from typing import Any
from modules.rule_evaluator import evaluate_rule, QCResult


def run_qc(
    tabs: dict[str, Any],
    qc_rules: dict[str, Any],
) -> list[QCResult]:
    """
    Run all QC rules against the extracted tab data.

    Parameters
    ----------
    tabs      : dict returned by tab_merger.merge_tabs()
    qc_rules  : the "qc_rules" config dict

    Returns
    -------
    List of QCResult objects (one per rule, in config order)
    """
    results: list[QCResult] = []
    for rule in qc_rules.get("rules", []):
        result = evaluate_rule(rule, tabs)
        results.append(result)
    return results


def qc_summary(results: list[QCResult]) -> dict[str, Any]:
    """Return counts and a boolean upload_blocked flag."""
    total = len(results)
    passed  = sum(1 for r in results if r.status == "pass")
    failed  = sum(1 for r in results if r.status == "fail")
    warned  = sum(1 for r in results if r.status == "warn")
    manual  = sum(1 for r in results if r.status == "manual")

    return {
        "total":          total,
        "passed":         passed,
        "failed":         failed,
        "warned":         warned,
        "manual":         manual,
        "upload_blocked": failed > 0,
    }
