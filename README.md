# MLB Pitch Loadout

A Streamlit app that suggests new pitches for MLB pitchers by finding
biomechanically similar comps and surfacing pitches those comps throw that the
target pitcher doesn't.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads its data from the Parquet snapshots in `snapshots/`, which are
committed to the repo — no raw data files are needed to run it.

## Deploying (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io, click **Create app**, and point it at
   this repo with `app.py` as the entrypoint.
3. Done — the app redeploys automatically on every push to the branch you
   selected.

## Refreshing the data

The published app is static: it shows whatever is in `snapshots/` at deploy
time. To refresh:

1. Update the raw Statcast CSVs (paths are set in `STATCAST_PATHS` in
   `data.py`), e.g. via `pybaseball`.
2. Refresh the spin leaderboard files (downloads from Baseball Savant):
   ```bash
   python -c "from data import download_spin_files; download_spin_files()"
   ```
3. Rebuild the snapshots and push:
   ```bash
   python snapshot.py
   git add snapshots && git commit -m "refresh data" && git push
   ```

## Repo layout

- `app.py` — the Streamlit app (reads only `snapshots/*.parquet`)
- `pitch_suggestions.py`, `distances.py` — similarity + suggestion pipeline
- `data.py` — raw-data loading/cleaning pipeline (only needed to rebuild snapshots)
- `snapshot.py` — rebuilds `snapshots/*.parquet` from the raw data
- `arsenal.py`, `biomech.py`, `stability.py`, `validation.py` — analysis modules
  used by the notebooks
- `Spin Files/` — Baseball Savant active-spin leaderboard exports (inputs to
  the snapshot build)
