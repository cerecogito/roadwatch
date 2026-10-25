# RoadWatch: Personal Demo Guide

Written for the presenter. Ships with the finalised prototype branch (new yellow/tomato "Shades" UI).

## 1. Start (10 minutes before)

```powershell
cd C:\Users\drish\Downloads\Roadwatch
.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8501
```

Open `http://localhost:8501/?week=2026-03-16&view=map` in Edge, F11, zoom 100%. Click through every view once to warm it up, then reload this URL.

Notes:
- The selected blocks on the map are now **black** (they were blue). Older docs (`demo_instructions.md`) still say "blue"; the numbers are unchanged.
- Gemini, Tiger Data and Snowflake are optional and off unless set in `.env`. Check the **Assistant & integrations** view before going on stage. If a service says "credentials missing", just don't demo it.
- No internet? The map's basemap tiles won't load, but the block lines and every number still do.

## 2. What to demo (about 4 minutes)

Tell one story: *"Which Brooklyn blocks should inspectors visit first next week, and can we trust that ranking?"*

| # | View | Do | Say |
|---|---|---|---|
| 1 | Inspection worklist & map | Show the banner, then the metrics (235 blocks, 1% capacity, P1/P2/P3 = 47/71/117). | "A replay of one real historical week, 16 March 2026. It forecasts 311 pothole *reports*, not confirmed damage." |
| 2 | Same | Point at the black selected blocks on the map, then the worklist table. | "The map and the worklist are the same selection. Black means chosen for inspection." |
| 3 | Same | Click **2%**, then back to **1%**. | "Inspectors choose capacity and the list updates. This is the lever a planner controls." |
| 4 | Same | Scroll to **Why each block is listed**. | "Every block comes with its reasons. It is not a black box." |
| 5 | Decision-policy comparison | Open it, show the default (strict top-K) against the alternatives. | "We compared several inspection policies on held-out weeks and chose the conservative one." |
| 6 | Weekly volume & scenarios | Show the chart: black line is actual, red is forecast with an 80% band. | "Forecasts of total weekly volume with honest uncertainty, including the March 2026 stress period." |
| 7 | Evidence & limitations | Show the metrics and the limitations. | "We state what this does not prove: no cost or savings claims, historical only." |
| 8 | Data provenance | Brief mention. | "Every source and frozen model hash is traceable." |
| 9 | Assistant & integrations (optional) | Only if verified. Ask Gemini one question about the current worklist. | "The assistant can only read the saved evidence and can't change a number." |

## 3. Likely questions

- **Is it live?** No. It is a historical evaluation and decision-support prototype.
- **Why this model?** Frozen model selection chose LightGBM with the strict top-K policy under conservative expected capture. The alternatives stay one click away.
- **Does it save money?** We make no cost or savings claims, and none are shown.
- **Is the AI making decisions?** No. Rankings come from the saved model. Gemini only explains saved evidence.

## 4. If something breaks

| Problem | Fix |
|---|---|
| Blank page | Reload the URL in step 1. Check `http://localhost:8501/_stcore/health` returns `ok`. |
| Port busy | Use `--server.port 8502`. |
| Integration errors | Set its flag to `0` in `.env`, restart, and skip step 9. |
| Fonts look plain | The Archivo font loads from Google Fonts. Offline, a fallback font is used and everything else is unchanged. |

Fuller scripts: `docs/final/demo_instructions.md`, `combined_run_of_show.md`, `integration_runbook.md`.
