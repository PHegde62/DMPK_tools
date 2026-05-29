"""
pipeline.py
CLI entry point. Wires all pipeline stages in order:
  1. Load config
  2. Read CRO workbook
  3. Parse & merge tabs
  4. Run QC
  5. Write QC report  (always written, even if upload is blocked)
  6. Standardise & build upload DataFrame
  7. Run upload-level QC rules
  8. Write upload file  (only if no failures)

Usage
-----
python pipeline.py \\
    --assay kinetic_solubility \\
    --input  data/ADME-GES-Solubility-20260528-PC_018.xlsx \\
    --template config/upload_templates.xlsx \\
    --output outputs/

Options
-------
--force-upload   Write upload file even if QC failures exist (use with caution)
--dry-run        Run QC and report only, never write upload file
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from datetime import datetime

# ── Module imports (all relative to project root) ─────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from modules.config_loader   import load_assay_config, get_test_compound_filter, get_control_compounds
from modules.reader          import read_cro_workbook
from modules.tab_merger      import merge_tabs
from modules.qc_engine       import run_qc, qc_summary
from modules.qc_reporter     import write_qc_report
from modules.standardizer    import apply_std_rules, apply_test_compound_filter
from modules.upload_writer   import build_upload_df, write_upload_file, ASSAY_TO_SHEET
from utils.logging_utils     import get_logger


def main() -> int:
    args = _parse_args()

    # ── Timestamp for output filenames ────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log = get_logger(log_file=output_dir / f"run_{ts}.log")
    log.info("=" * 60)
    log.info(f"DMPK Pipeline  |  assay={args.assay}  |  run={ts}")
    log.info(f"Input:    {args.input}")
    log.info(f"Output:   {output_dir}")
    log.info("=" * 60)

    # ── Stage 1: Load config ──────────────────────────────────────────────────
    log.info("Stage 1 — Loading config")
    config = load_assay_config(args.assay)
    column_map = config["column_map"]
    qc_rules   = config["qc_rules"]
    std_rules  = config["std_rules"]
    # Attach BLQ config to std_rules for standardizer convenience
    std_rules["blq_detection"] = qc_rules.get("blq_detection", {})

    # ── Stage 2: Read CRO workbook ────────────────────────────────────────────
    log.info("Stage 2 — Reading CRO workbook")
    tabs = read_cro_workbook(args.input, column_map)
    log.info(f"  Tabs loaded: {list(tabs.keys())}")

    # ── Stage 3: Parse & merge tabs ───────────────────────────────────────────
    log.info("Stage 3 — Parsing compound blocks & merging tabs")
    tabs = merge_tabs(tabs, column_map)
    n_rows = len(tabs["merged"])
    log.info(f"  Merged DataFrame: {n_rows} rows")

    # ── Stage 4: Run QC ───────────────────────────────────────────────────────
    log.info("Stage 4 — Running QC rules")
    results = run_qc(tabs, qc_rules)
    summary = qc_summary(results)
    log.info(
        f"  QC complete: {summary['passed']} pass / "
        f"{summary['warned']} warn / {summary['failed']} fail / "
        f"{summary['manual']} manual"
    )

    # ── Stage 5: Write QC report ──────────────────────────────────────────────
    log.info("Stage 5 — Writing QC report")
    sig = tabs.get("signature", {})
    qc_report_path = output_dir / f"qc_report_{ts}.xlsx"
    write_qc_report(
        results,
        qc_report_path,
        report_metadata={
            "report_number":    sig.get("report_number", ""),
            "cro_name":         sig.get("cro_name", ""),
            "study_start_date": sig.get("study_start_date", ""),
            "assay_type":       args.assay,
        },
    )
    log.info(f"  QC report → {qc_report_path}")

    # ── Stage 6: Standardise ──────────────────────────────────────────────────
    log.info("Stage 6 — Standardising data")
    std_df = apply_std_rules(
        merged_df=tabs["merged"],
        signature=sig,
        std_rules=std_rules,
        da_df=tabs.get("data_analysis_parsed"),
    )

    # Filter to test compounds only
    filter_cfg = get_test_compound_filter(config)
    std_df = apply_test_compound_filter(std_df, filter_cfg)
    log.info(f"  Test compounds for upload: {std_df['compound_id'].tolist()}")

    # Build upload DataFrame
    upload_df = build_upload_df(std_df, std_rules)
    tabs["upload_df"] = upload_df  # make available for upload-level QC rules

    # ── Stage 7: Upload-level QC ──────────────────────────────────────────────
    log.info("Stage 7 — Running upload-level QC")
    upload_qc_rules = {
        "rules": [r for r in qc_rules["rules"] if r.get("section") == "Upload"]
    }
    upload_results = run_qc(tabs, upload_qc_rules)
    results.extend(upload_results)
    summary = qc_summary(results)

    # ── Stage 8: Write upload file ────────────────────────────────────────────
    if summary["upload_blocked"] and not args.force_upload:
        log.warning(
            f"Upload BLOCKED — {summary['failed']} QC failure(s). "
            "Resolve failures and re-run, or use --force-upload to override."
        )
        if args.dry_run:
            log.info("Dry-run mode — no upload file written.")
        return 1

    if args.dry_run:
        log.info("Dry-run mode — skipping upload file write.")
        return 0

    log.info("Stage 8 — Writing upload file")
    template_path = Path(args.template)
    upload_path = output_dir / f"upload_ready_{ts}.xlsx"
    sheet_name = ASSAY_TO_SHEET.get(args.assay)
    write_upload_file(upload_df, template_path, upload_path, assay_sheet_name=sheet_name)
    log.info(f"  Upload file → {upload_path}")

    log.info("Pipeline complete.")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DMPK Assay QC and Upload Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--assay",    required=True,  help="Assay type (matches config folder name)")
    p.add_argument("--input",    required=True,  help="Path to CRO Excel workbook")
    p.add_argument("--template", required=True,  help="Path to upload template Excel")
    p.add_argument("--output",   default="outputs/", help="Output directory (default: outputs/)")
    p.add_argument("--force-upload", action="store_true",
                   help="Write upload file even if QC failures exist")
    p.add_argument("--dry-run",  action="store_true",
                   help="Run QC and generate report only — do not write upload file")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main())
