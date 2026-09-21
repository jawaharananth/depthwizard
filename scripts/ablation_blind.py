"""
Which scale source actually produced 5.21 m / 2.41 m: DEM or the GAMUS
metric checkpoint?

Both `use_dem` and the metric checkpoint were live in the same run that
produced that headline number, and `build_city_image.py` tries DEM before
ever checking the metric backbone (`tier is None and depth_is_metric`), so
DEM wins on every tile where it succeeds and the metric model's contribution
was never isolated. This runs the same 10 tiles through three real
configurations of the same code path -- not a re-derivation from the earlier
numbers -- and reports what each one actually measures.

    python scripts/ablation_blind.py [n_tiles]

No ground truth is touched before scoring in any of the three runs; nothing
here is tuned against the result.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import benchmark_blind as bb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIGS = [
    ("dem_only",    dict(use_dem=True,  use_metric=False)),
    ("metric_only", dict(use_dem=False, use_metric=True)),
    ("dem+metric",  dict(use_dem=True,  use_metric=True)),
]


def _agg(results):
    scored = [r for r in results if r["status"] == "SCORED"]
    if not scored:
        return None
    import numpy as np
    rmse = float(np.sqrt(np.mean([r["metrics"]["overall"]["rmse_m"] ** 2 for r in scored])))
    mae = float(np.mean([r["metrics"]["overall"]["mae_m"] for r in scored]))
    return {"n_scored": len(scored), "n_total": len(results),
            "rmse": round(rmse, 2), "mae": round(mae, 2)}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    runs = {}
    for label, kwargs in CONFIGS:
        print(f"\n{'=' * 70}\nCONFIG: {label}  ({kwargs})\n{'=' * 70}")
        report_path = os.path.join(ROOT, f"_ablation_{label}.md")
        results = bb.main(max_tiles=n, report_path=report_path, **kwargs)
        runs[label] = {"results": results, "agg": _agg(results)}

    tiles = sorted({r["tile"] for r in runs["dem+metric"]["results"]})

    lines = []
    lines.append("# DepthWizard -- Ablation: DEM vs GAMUS metric checkpoint")
    lines.append("")
    lines.append("**Question:** the headline blind number (RMSE 5.21 m / MAE 2.41 m, "
                 "8/10 tiles) was measured with both `--dem` (default True) and the "
                 "GAMUS-finetuned metric checkpoint present at the same time. "
                 "`build_city_image.py` attempts Tier A (DEM) before it ever checks "
                 "the metric backbone, so which one is actually responsible was not "
                 "known until this ablation ran.")
    lines.append("")
    lines.append("**Method:** same 10 DFC2019 tiles, same `benchmark_blind.py` path, "
                 "three configurations of the two flags. No ground truth is used "
                 "before scoring in any run; results are not tuned.")
    lines.append("")
    lines.append("## Aggregate result per configuration")
    lines.append("")
    lines.append("| config | RMSE | MAE | tiles scored |")
    lines.append("|---|---|---|---|")
    for label, _ in CONFIGS:
        a = runs[label]["agg"]
        if a is None:
            lines.append(f"| {label} | -- | -- | 0/{n} (nothing reached a metric tier) |")
        else:
            lines.append(f"| {label} | {a['rmse']} | {a['mae']} | {a['n_scored']}/{a['n_total']} |")
    lines.append("")

    lines.append("## Per-tile scale_source, all three configs")
    lines.append("")
    lines.append("| tile | dem_only | metric_only | dem+metric (winner) |")
    lines.append("|---|---|---|---|")
    by_tile = {label: {r["tile"]: r for r in runs[label]["results"]} for label, _ in CONFIGS}
    for t in tiles:
        row = [t]
        for label, _ in CONFIGS:
            r = by_tile[label].get(t)
            if r is None:
                row.append("--")
            elif r["status"] != "SCORED":
                row.append(r["status"].lower())
            else:
                row.append(f"{r['scale_source']} (RMSE {r['metrics']['overall']['rmse_m']:.2f})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Reproducing")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/ablation_blind.py 10")
    lines.append("```")
    lines.append("")

    out_path = os.path.join(ROOT, "BENCHMARK_ABLATION.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {out_path}")

    for label, _ in CONFIGS:
        a = runs[label]["agg"]
        print(f"  {label:12s} {a}")


if __name__ == "__main__":
    main()
