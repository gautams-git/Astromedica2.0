# TRIAD-5 benchmark

The five scoring engines are not modified. LLM weight selection is wired only around Rashi Correlation, through the existing `weights_context` lock. The other four algorithms still run on Track A (original engine) and Track B (agent, default weights, no LLM). Their Track C cell is an explicit gap.

## Twelve patients

The benchmark reads up to twelve patient folders that already contain Prokerala results: `raw_chart.json` and `raw_dasha.json`. It rebuilds the chart locally. It does not call Prokerala, including when a folder is missing.

Default directory: `patients/`. Override with `BENCHMARK_PATIENTS_DIR`.

Those folders are gitignored because they hold birth details. On a machine that already has them, they appear on `GET /patients-list` and open from `GET /patient-data/<id>` with the Rashi three-track panel attached. Opening them reuses the saved chart.

| Split | Patients | Used for |
|---|---|---|
| train | first 6 saved ids | pilot search only |
| validation | next 2 saved ids | held out of the pilot search |
| holdout | the rest, up to 12 | held out of the pilot search |

There are no adjudicated clinical labels. Metrics exist and are tested on hand-written vectors. They are not reported as clinical accuracy.

## Run

From the repository root, with the project virtualenv:

```bash
pytest tests/test_agentic_rashi_correlation.py tests/benchmark -q
python -m benchmark
```

The second command writes:

- `benchmark/reports/executive.md`
- `benchmark/reports/technical.md`
- `benchmark/reports/evidence.json`
- `benchmark/reports/critic.json`
- `benchmark/reports/pilot.json`

## Local server

```bash
python server.py
```

Then open:

- `http://localhost:5000/` — the twelve benchmark patients are on the index
- `http://localhost:5000/results?patient_id=BM01` — Rashi tracks on the results page
- `http://localhost:5000/benchmark` — deviation explorer

`BENCHMARK_RASHI_WEIGHT_MODE` may be `mock` (default when `GEMINI_API_KEY` is unset), `live`, or `off`. Mock is a scripted 20% increase on the nakshatra layer and the P6 anchor. It is not a live model. Live proposals that fail the 50%–150% range check are not executed.

The pilot search does not write production weights.
