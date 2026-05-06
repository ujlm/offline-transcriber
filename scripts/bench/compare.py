"""Compare benchmark run against baseline and fail on regressions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Thresholds:
    max_wer_delta_pp: float
    max_rtfx_drop_pct: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare bench run against baseline JSON.")
    p.add_argument("results", type=Path)
    p.add_argument("baseline", type=Path)
    return p.parse_args()


def _load_thresholds(payload: dict) -> Thresholds:
    t = payload.get("thresholds") or {}
    return Thresholds(
        max_wer_delta_pp=float(t.get("max_wer_delta_pp", 0.5)),
        max_rtfx_drop_pct=float(t.get("max_rtfx_drop_pct", 10.0)),
    )


def main() -> int:
    args = _parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    thresholds = _load_thresholds(baseline)
    base_items = {
        str(row["item_id"]): row for row in (baseline.get("items") or []) if "item_id" in row
    }
    rows: list[tuple[str, float, float, float, float, str]] = []
    failures: list[str] = []

    for row in results.get("items") or []:
        item_id = str(row.get("item_id"))
        base = base_items.get(item_id)
        if base is None:
            failures.append(f"Missing baseline case for {item_id}")
            continue
        new_wer = float(row.get("wer") or 0.0)
        base_wer = float(base.get("wer") or 0.0)
        new_rtfx = float(row.get("rtfx") or 0.0)
        base_rtfx = float(base.get("rtfx") or 0.0)

        wer_delta_pp = (new_wer - base_wer) * 100.0
        rtfx_drop_pct = 0.0
        if base_rtfx > 0:
            rtfx_drop_pct = ((base_rtfx - new_rtfx) / base_rtfx) * 100.0

        status = "ok"
        if wer_delta_pp > thresholds.max_wer_delta_pp:
            status = "wer_regression"
            failures.append(
                f"{item_id}: WER regression {wer_delta_pp:.3f}pp "
                f"(limit {thresholds.max_wer_delta_pp:.3f}pp)"
            )
        if rtfx_drop_pct > thresholds.max_rtfx_drop_pct:
            status = "rtfx_regression" if status == "ok" else f"{status}+rtfx_regression"
            failures.append(
                f"{item_id}: RTFx drop {rtfx_drop_pct:.3f}% "
                f"(limit {thresholds.max_rtfx_drop_pct:.3f}%)"
            )

        rows.append((item_id, base_wer, new_wer, base_rtfx, new_rtfx, status))

    print("| case | base_wer | new_wer | base_rtfx | new_rtfx | status |")
    print("|---|---:|---:|---:|---:|---|")
    for row in rows:
        print(
            f"| {row[0]} | {row[1]:.4f} | {row[2]:.4f} | {row[3]:.3f} | {row[4]:.3f} | {row[5]} |"
        )

    if failures:
        print("\nRegressions:")
        for msg in failures:
            print(f"- {msg}")
        return 1

    print("\nNo regressions detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
