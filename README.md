# RoadWatch

RoadWatch ranks Brooklyn's 23,448 street blocks by how likely each is to receive a **311 pothole report in the next 7 days**, and turns that ranking into a weekly, capacity-limited **inspection worklist** with reasons, a map and CSV/GeoJSON exports. It was built for OxHack.

The pipeline:
1. ingests public NYC Open Data and NOAA data into immutable, checksummed snapshots;
2. matches 311 reports to street blocks;
3. builds point-in-time features and trains LightGBM in a monthly walk-forward backtest against simple baselines;
4. serves the results in a Streamlit dashboard.

> **Scope.** RoadWatch predicts public *reports*, not confirmed defects. It does not measure defects found, failures prevented or money saved. All dashboard weeks are retrospective replays of saved forecasts, not live recommendations.

## Requirements

- Windows, macOS or Linux with **Python 3.12** (the commands below use Windows PowerShell paths).
- [uv](https://docs.astral.sh/uv/) is recommended for creating the environment.
- Internet access is needed only to download data (`ingest`) and for the dashboard's map basemap.

## Installation

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements-lock.txt   # exact, tested versions
uv pip install --python .venv\Scripts\python.exe --no-deps -e .            # the roadwatch package
```

**Optional:** a free Socrata app token raises API rate limits. Put it in a `.env` file at the repository root, which is never committed:

```text
SOCRATA_APP_TOKEN=your-token
```

Tokens are sent only as request headers and are never logged.

## Build everything from public data

Run from the repository root. Data is not included in the repository; `ingest` downloads it. The full run takes a while: the 311 download and the walk-forward test are the slowest steps.

```powershell
$py = ".venv\Scripts\python.exe"
& $py -I -m roadwatch ingest all              # download -> validate -> immutable snapshot -> processed tables
& $py -I -m roadwatch stage3                  # street blocks, report-to-block matching, weekly labels
& $py -I -m roadwatch stage4 features         # point-in-time (as-of) features
& $py -I -m roadwatch stage4 select           # model/feature selection on validation weeks only (writes a frozen record)
& $py -I -m roadwatch stage4 test             # monthly walk-forward test predictions
& $py -I -m roadwatch stage4 evaluate         # test metrics and plots
& $py -I -m roadwatch stage4 uncertainty      # paired week-level bootstrap comparisons
& $py -I -m roadwatch stage5 all              # inspection lists, policy comparison, capture definitions, volume forecast
& $py -I -m roadwatch stage6 build            # precompute dashboard inputs
```

Run `stage4 select` once, before any test predictions exist. Re-running it after `stage4 test` would select with test results already on disk.

## Run the dashboard

```powershell
.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8501
```

Open <http://localhost:8501>. Add `--server.address localhost` to keep it private on shared networks. Stop it with **Ctrl+C**.

| View | Contents |
|---|---|
| Inspection worklist & map | Ranked weekly list, map, search and priority filters, per-block reasons, CSV/GeoJSON downloads |
| Decision-policy comparison | Alternative lists under four report-capture definitions, with 95% intervals |
| Weekly volume & scenarios | Weekly report counts and the (context-only) volume forecast |
| Evidence & limitations | Walk-forward test metrics, paired bootstrap, limitations |
| Data provenance | Source snapshots, validation status and checksums |

Sidebar controls:
- **forecast week**;
- **inspection capacity**: 0.5%, 1% or 2% of blocks, or a custom number;
- **inspection list**: the default strict single-block LightGBM top-K list, or two non-adjacent alternatives;
- **map background density**.

Links can preselect a state, for example `?week=2026-03-16&view=policy`.

## Command-line reference

`python -I -m roadwatch <command>`:

| Command | Purpose |
|---|---|
| `ingest <sources\|core\|all> [--new] [--start D] [--end D]` | Download (or resume), validate, snapshot and promote. Exit code 0 = ok, 1 = a core source failed, 2 = only optional sources failed |
| `rebuild <sources\|all>` | Rebuild processed tables from archived snapshots (offline) |
| `status` | List snapshots and their state |
| `report` | Write the data-quality report |
| `register-local` | Checksum locally downloaded original CSV exports, if present |
| `stage3`, `stage4 <step>`, `stage5 <step>`, `stage6 build` | Pipeline stages (see above) |
| `stage5 worklist --week 2026-03-16 --capacity 0.01 --policy primary__LGBM --rule topk` | One week's worklist as CSV and GeoJSON in `outputs/stage5/worklists/`. `--rule topk` reproduces the dashboard's default list |

**Smoke test against the real APIs,** written to an isolated root:

```powershell
& $py -I -m roadwatch --root outputs\smoke ingest sr311 noaa --start 2025-01-01 --end 2025-02-01 --page-size 500
```

## Configuration

`config/stage5.json` holds the decision-engine settings: capacity scenarios, policies, tie-break, the optional ZIP cap, volume-model settings and the dashboard's default inspection list (`dashboard_default_list`).

## Tests

```powershell
.venv\Scripts\python.exe -I -m pytest -q
```

Unit tests use small synthetic fixtures: matching, labels, leakage cut-offs, walk-forward windows, selection rules, capture definitions and exports. Integration tests that need built outputs are skipped automatically until the pipeline has been run.

## Data sources (Brooklyn scope)

| Key | Dataset | Role |
|---|---|---|
| `sr311` | NYC 311 `erm2-nwe9`, Street Condition, Brooklyn, 2020 → today | primary (labels and history features) |
| `sr311_unassigned` | NYC 311 Street Condition with missing borough (citywide) | supplementary |
| `centerline` | NYC Street Centerline (CSCL) `inkn-q76z`, Brooklyn | street blocks and road attributes |
| `noaa` | NOAA GHCN-Daily `USW00094728` (Central Park) via NCEI | weather (tested; not selected) |
| `pavement`, `workorders`, `resurfacing_inhouse`, `resurfacing_schedule` | `6yyb-pb25`, `x9wy-ing4`, `ffaf-8mrv`, `xnfm-u3k5` | optional diagnostics |

## Repository layout

```text
src/roadwatch/   pipeline package: ingestion (sources, socrata, noaa, snapshot, validate, pipeline), matching and
                 labels (names, blocks, matching, labels, stage3), features and models (features, modeling, evaluation,
                 stage4), decisions (inspection, volume, stage5), dashboard logic (dashboard_data, dashboard_map), cli
app.py           Streamlit dashboard
config/          decision-engine configuration
tests/           pytest suite
data/            created by the pipeline (raw snapshots, processed tables); not committed
outputs/         created by the pipeline (evaluation tables, worklists, dashboard inputs); not committed
```

## Limitations

- **Reporting bias.** Labels are 311 reports, so they reflect who reports as well as road condition.
- **Ambiguous locations.** About 40% of reports name an intersection or a vague street stretch, so capture figures depend on how such reports are credited.
- **Probabilities are not reliable in surges.** The weekly volume forecast does not reliably beat simple baselines, so capacity is a manual setting.
- **Static road attributes.** The street-network attributes are a recent snapshot.
- **Further evidence needed before operational use:** inspection outcomes, field confirmation of what an inspection covers, and a prospective trial.
