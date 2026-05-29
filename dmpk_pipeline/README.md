# DMPK QC Pipeline

A lightweight, config-driven QC and upload pipeline for DMPK CRO assay workbooks.
Processes raw CRO Excel files, runs automated QC checks, and generates upload-ready data.

## Live app

**[Launch QC App →](https://PHegde62.github.io/DMPK_tools/)**

No install, no API key, no account needed. Drop a CRO workbook and run.

## Supported assays

| Assay | Status |
|---|---|
| Kinetic Solubility | ✅ Full automation |
| Microsomal Stability | ✅ Full automation |
| Permeability (MDCKII) | ✅ Full automation |
| Plasma Protein Binding | ✅ Full automation |
| LogD | ✅ Full automation |
| Hepatocyte Stability | ✅ Full automation |
| Hepatocyte Binding | ✅ Full automation |

## What it does

- Parses all CRO workbook tabs (Signature, Summary, Materials, Bioanalytical Method, raw data)
- Runs 25–35 automated QC checks per assay (IS %CV, R², S/N, Q1 m/z, control ranges, dilution factors, formula cross-checks)
- Only 2 manual checks remain: MW registry confirmation and final platform upload sign-off
- Generates an upload-ready `upload` sheet appended to the original CRO workbook
- Failures can be overridden with scientist sign-off before download

## Repository structure

```
DMPK-tools/
├── docs/               ← GitHub Pages app (auto-updated from app/)
│   └── index.html
├── app/
│   └── index.html      ← Source for the web app
├── modules/            ← Python pipeline modules (8 modules)
├── config/
│   └── assay_config/   ← Per-assay YAML configs (column maps, QC rules, std rules)
├── tests/              ← Python unit tests (15 passing)
├── pipeline.py         ← CLI entry point
└── requirements.txt
```

## CLI usage

```bash
pip install -r requirements.txt

python pipeline.py \
  --assay kinetic_solubility \
  --input data/your_cro_workbook.xlsx \
  --template config/upload_templates.xlsx \
  --output outputs/
```

## Adding a new assay

1. Create `config/assay_config/<assay_name>/`
2. Add `column_map.yaml`, `qc_rules.yaml`, `std_rules.yaml`
3. Run the pipeline — no Python changes needed
