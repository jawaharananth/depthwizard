"""Charts for the SIH idea deck, drawn from the measured result files.

    python tools/ppt_charts.py            # writes docs/deck/*.png

Every number is read from final_out/*.json or the checkpoint's own files, so
the deck cannot drift from what was measured.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "deck")
NAVY, GREEN, GREY, RED, INK = "#1F4E79", "#2E7D4F", "#A7B1BC", "#B8322A", "#1B2733"
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 13, "axes.edgecolor": "#C9D1D9",
                     "axes.spines.top": False, "axes.spines.right": False})


def absolute_vs_lidar():
    """Absolute DSM RMSE against airborne LiDAR, ours vs Copernicus alone."""
    ours = json.load(open(os.path.join(ROOT, "final_out", "compare_run03.json")))["height_run03"]["swiss"]
    glo = {}
    rows = json.load(open(os.path.join(ROOT, "data", "swiss_lidar", "results.json")))
    for r in rows:
        glo[r["landscape"]] = r["absolute"]["glo30"]["dem_only"]["rmse"]
    order = ["hilly", "sparse", "forested", "urban"]
    lab = {"hilly": "HILLY\nZermatt", "sparse": "SPARSE\nfarmland", "forested": "FOREST\nSihlwald", "urban": "URBAN\nZurich"}
    fig, ax = plt.subplots(figsize=(8.4, 3.75), dpi=200)
    x = range(len(order))
    a = [glo[k] for k in order]
    b = [ours[k]["absolute"]["rmse"] for k in order]
    ax.bar([i - 0.2 for i in x], a, 0.38, color=GREY, label="Copernicus GLO-30 alone")
    ax.bar([i + 0.2 for i in x], b, 0.38, color=GREEN, label="DepthWizard absolute DSM")
    for i, (u, v) in enumerate(zip(a, b)):
        ax.text(i - 0.2, u + 0.25, f"{u:.1f}", ha="center", color="#56616C", fontsize=12)
        ax.text(i + 0.2, v + 0.25, f"{v:.1f} m", ha="center", color=GREEN, fontsize=13, fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels([lab[k] for k in order], fontsize=12, fontweight="bold", color=INK)
    ax.set_ylabel("RMSE vs airborne LiDAR (m)", color="#56616C")
    ax.set_ylim(0, max(a) * 1.25)
    ax.legend(frameon=False, loc="upper right", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "absolute_vs_lidar.png"), transparent=True)


def flat_plane_fix():
    """The measured bug: the old surface was a flat plane under the city."""
    rows = json.load(open(os.path.join(ROOT, "data", "swiss_lidar", "results.json")))
    ours = json.load(open(os.path.join(ROOT, "final_out", "compare_run03.json")))["height_run03"]["swiss"]
    order = ["hilly", "forested", "urban", "sparse"]
    old = {r["landscape"]: r["absolute"]["glo30"]["flat_plane_old"]["rmse"] for r in rows}
    fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=200)
    x = range(len(order))
    a = [old[k] for k in order]; b = [ours[k]["absolute"]["rmse"] for k in order]
    ax.bar([i - 0.2 for i in x], a, 0.38, color=RED, label="before: model ground + one offset")
    ax.bar([i + 0.2 for i in x], b, 0.38, color=GREEN, label="after: DEM terrain + model height")
    for i, (u, v) in enumerate(zip(a, b)):
        ax.text(i - 0.2, u + 2, f"{u:.0f} m", ha="center", color=RED, fontsize=12, fontweight="bold")
        ax.text(i + 0.2, v + 2, f"{v:.1f}", ha="center", color=GREEN, fontsize=12, fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels([k.upper() for k in order], fontsize=12, fontweight="bold", color=INK)
    ax.set_ylabel("absolute RMSE vs LiDAR (m)", color="#56616C")
    ax.set_ylim(0, max(a) * 1.18)
    ax.legend(frameon=False, loc="upper right", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "flat_plane_fix.png"), transparent=True)


def resolution():
    """Per-building error vs input resolution, ISRO's 0.6 m marked."""
    r = json.load(open(os.path.join(ROOT, "final_out", "resolution_test.json")))
    pts = [(0.25, r["0.25 m native"]), (0.5, r["0.50 m, upsampled"]), (0.6, r["0.60 m, upsampled"]),
           (0.75, r["0.75 m, upsampled"]), (1.0, r["1.00 m, upsampled"])]
    base = pts[0][1]["bldg_rmse"]
    fig, ax = plt.subplots(figsize=(6.4, 3.3), dpi=200)
    xs = [p[0] for p in pts]; ys = [p[1]["bldg_rmse"] for p in pts]
    ax.plot(xs, ys, "-o", color=NAVY, lw=2.5, ms=7)
    ax.axvline(0.6, color="#E0A21B", lw=2, ls="--")
    ax.text(0.61, max(ys) * 0.99, "ISRO evaluates\nat 0.6 m", color="#B07A0A", fontsize=11, va="top")
    for x, y in zip(xs, ys):
        below = abs(x - 0.6) < 1e-6                           # keep clear of the ISRO line and 0.5 m
        ax.text(x + (0.012 if below else 0), y - 0.075 if below else y + 0.04, f"{y:.2f}",
                ha="left" if below else "center", fontsize=11, color=INK)
    ax.set_xlabel("input ground sample (m)"); ax.set_ylabel("per-building RMSE (m)")
    ax.set_ylim(min(ys) - 0.2, max(ys) + 0.25)
    y06 = pts[2][1]["bldg_rmse"]
    ax.set_title(f"0.6 m costs +{100 * (y06 / base - 1):.1f} % per building", fontsize=13, color=GREEN, fontweight="bold", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "resolution.png"), transparent=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    absolute_vs_lidar(); flat_plane_fix(); resolution()
    print("charts ->", OUT, sorted(os.listdir(OUT)))
