"""Fetch raw daily closes through the user's local Workspace desktop session.

Optional download dependency only: lseg-data==2.1.1. Offline replay uses the
unchanged root requirements and saved CSV/JSON files.
"""

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib, json

ROOT = Path(__file__).resolve().parent


def main():
    import lseg.data as ld
    import importlib.metadata

    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    ld.open_session(name="desktop.workspace")
    try:
        frame = ld.get_history(
            ["TSLA.O", "TSM.N"],
            fields=["TRDPRC_1"],
            interval="1D",
            start="2025-03-07",
            end="2026-03-07",
            adjustments=["unadjusted"],
        )
        if frame.empty:
            raise RuntimeError("No LSEG close observations returned")
        frame.to_csv(data / "lseg_raw_close.csv", index_label="date")
        normalized = frame.copy()
        if hasattr(normalized.columns, "nlevels") and normalized.columns.nlevels > 1:
            normalized.columns = [
                next(v for v in c if v in ["TSLA.O", "TSM.N"])
                for c in normalized.columns
            ]
        normalized = normalized.rename(columns={"TSLA.O": "TSLA", "TSM.N": "TSM"})[
            ["TSLA", "TSM"]
        ].dropna()
        normalized.to_csv(data / "tsla_tsm_close.csv", index_label="date")
        manifest = {
            "provider": "LSEG Workspace local desktop Python API",
            "session": "desktop.workspace",
            "sdk_version": importlib.metadata.version("lseg-data"),
            "field": "TRDPRC_1",
            "adjustments": ["unadjusted"],
            "interval": "1D",
            "rics": ["TSLA.O", "TSM.N"],
            "requested_start": "2025-03-07",
            "requested_end": "2026-03-07",
            "retrieved_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            "observations": len(normalized),
            "first_date": str(normalized.index[0].date()),
            "last_date": str(normalized.index[-1].date()),
            "files": {
                name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                for name in ["lseg_raw_close.csv", "tsla_tsm_close.csv"]
            },
        }
        (data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
    finally:
        ld.close_session()


if __name__ == "__main__":
    main()
