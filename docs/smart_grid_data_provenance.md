# Smart Grid Data Provenance

*Created: 2026-05-01*
*Owner: Tanisha Rathod (tannedpeach)*
*Issue: [#14](https://github.com/HPML6998-S26-Team13/AssetOpsBench/issues/14)*

## Overview

The Smart Grid 7th-domain MCP servers in this fork operate over **synthetic
data only**. No proprietary or class-licensed data is shipped with the AOB
codebase. Runtime data location is configured via the `SG_DATA_DIR`
environment variable.

## What `SG_DATA_DIR` is

`SG_DATA_DIR` is an environment variable pointing at the directory containing
the synthetic Smart Grid CSV datasets the servers read at runtime.

**Default path:** `./data/sg_processed/` relative to the current working
directory (wherever the server process is launched from).

Resolution order in [`src/servers/smart_grid/base.py`](../src/servers/smart_grid/base.py):

1. `SG_DATA_DIR` environment variable — absolute or cwd-relative path.
2. `./data/sg_processed/` relative to cwd (fallback if the variable is unset).

The path is not required to exist at import time; existence is enforced on the
first data-loading call, which raises a clear `FileNotFoundError` with
remediation instructions if the path is missing.

## What's in `SG_DATA_DIR`

Six synthetic CSV files, one per logical data slice:

| File | Server(s) | Description |
|---|---|---|
| `asset_metadata.csv` | IoT | Static nameplate data per transformer (`transformer_id`, `name`, `manufacturer`, `location`, `voltage_class`, `rating_kva`, `install_date`, `age_years`, `health_status`, `fdd_category`, `rul_days`, `in_service`) |
| `sensor_readings.csv` | IoT, TSFM | Time-series sensor readings (load current, winding temp, oil temp, voltage) |
| `failure_modes.csv` | FMSR | Failure mode catalogue with severity, IEC code, and recommended action |
| `dga_records.csv` | FMSR | Dissolved Gas Analysis (DGA) records per transformer, per sample date |
| `rul_labels.csv` | TSFM | Remaining-useful-life labels and health index per transformer |
| `fault_records.csv` | WO | Historical fault and maintenance event records |

All values are synthetic. Gas concentrations in `dga_records.csv` are
derived from the team's data pipeline; the IEC 60599:2022 Rogers Ratio
fault-table boundaries used for DGA classification are encoded in
`data/knowledge/transformer_standards.json` in the team repo
(see `HPML6998-S26-Team13/hpml-assetopsbench-smart-grid-mcp`).

## No-CSV-port policy

The team repo's processed CSVs (under `data/processed/`) are **not** copied
into this AOB fork or any upstream IBM PR. Reasons:

1. **Licensing** — the five Kaggle source datasets used in the team's data
   pipeline are CC0 individually, but the processed outputs are class-specific
   derivatives. Keeping them out of AOB avoids a licensing audit for upstream
   reviewers.
2. **Reproducibility** — the synthetic data can be regenerated from
   `data/generate_synthetic.py` in the team repo. Any downstream user with the
   generator and the IEC encoding can produce equivalent datasets without
   needing the team's processed CSVs.
3. **AOB cleanliness** — no raw class artifacts in the fork simplifies upstream
   review scope.

**For a reviewer or downstream user needing Smart Grid data:**

```bash
git clone https://github.com/HPML6998-S26-Team13/hpml-assetopsbench-smart-grid-mcp.git
cd hpml-assetopsbench-smart-grid-mcp
pip install -r requirements.txt
python data/generate_synthetic.py          # produces data/processed/*.csv
export SG_DATA_DIR=$(pwd)/data/processed
```

## Source datasets

The team's data pipeline draws from five Kaggle CC0 datasets:

| Dataset | Domain servers |
|---|---|
| Power Transformers FDD & RUL | IoT, TSFM |
| DGA Fault Classification | FMSR |
| Smart Grid Fault Records | WO |
| Transformer Health Index | FMSR (supplemental) |
| Current & Voltage Monitoring | IoT, TSFM (supplemental) |

Dataset licensing details and row counts: `docs/hpml_datasets.pdf` in the team
repo.

## IEC / IEEE standards encoding

DGA-related ground truth (fault codes, condition tiers, gas thresholds) is
encoded in `data/knowledge/transformer_standards.json` in the team repo. That
artifact reflects:

- **IEC 60599:2022** (4th ed., publication 66491) — Rogers Ratio method,
  fault-table boundaries, representative gas profiles
- **IEEE C57.104-2019** — condition framework (C1–C4) and gas threshold values

The FMSR server's `analyze_dga` tool implements the Rogers Ratio method
using the fault-table boundaries from that artifact. Note: the AOB fork
server encodes the table directly in `src/servers/smart_grid/fmsr/main.py`
rather than reading the JSON at runtime. Downstream users regenerating DGA
records should verify that generated gas values round-trip correctly through
`analyze_dga` for their intended fault labels before using them as benchmark
ground truth.

## Citation

SmartGridBench: A Smart Grid transformer maintenance benchmark for MCP-enabled
LLM agents. Team 13, HPML Spring 2026, Columbia University.
*Citation will be updated when the NeurIPS 2026 Datasets & Benchmarks
submission is finalized.*
