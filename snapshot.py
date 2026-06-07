"""
Build the small summary tables the app needs and save them as Parquet.

Run this once (and re-run whenever you refresh the underlying data) with the
Python environment that has access to the raw Statcast/spin CSVs:

    python snapshot.py

After this, app.py loads the Parquet snapshot directly and no longer needs the
raw CSVs (or pybaseball) at runtime.
"""
import os
from data import build_all

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snapshots')

# Only the tables app.py actually consumes.
SNAPSHOT_KEYS = ['pitcher_summ_r', 'pitcher_summ_l', 'pitch_type_r', 'pitch_type_l']


def build_snapshot(live=False):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    data = build_all(live=live)
    for key in SNAPSHOT_KEYS:
        path = os.path.join(SNAPSHOT_DIR, f'{key}.parquet')
        data[key].to_parquet(path, index=False)
        size_kb = os.path.getsize(path) / 1024
        print(f'  wrote {key:18s} {len(data[key]):>7,} rows  ({size_kb:,.1f} KB)')
    print(f'Snapshot written to {SNAPSHOT_DIR}')


if __name__ == '__main__':
    build_snapshot(live=False)
