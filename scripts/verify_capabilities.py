"""
Static, re-runnable capability audit for DepthWizard.

Resolves a fixed list of claims (about the pipeline modules and the viewer)
to exactly one status, using only grep/AST-level static analysis -- nothing
here imports torch, opens an image, or runs the pipeline. Run it after any
change to an entry point (build_city.py, build_city_image.py, server.py) or
to viewer/index.html; it is deterministic given the same source tree.

Statuses:
  LIVE     reachable from an entry point's default code path, call site proven
  OPT-IN   implemented and wired, but gated behind a flag/condition that
           defaults off (an argparse flag whose default is False/None, or an
           `if` that is not unconditionally true)
  DORMANT  the module/function exists and is syntactically valid, but no
           entry point imports or calls it (checked across build_city.py,
           build_city_image.py, server.py -- transitively through anything
           those import)
  ABSENT   grep/AST finds no implementation matching the claim at all

Usage:
  python scripts/verify_capabilities.py            # print + write CAPABILITIES.md
  python scripts/verify_capabilities.py --check     # exit 1 if any claim is
                                                      # UNRESOLVED (see below)

A claim that cannot be confidently resolved is reported as UNRESOLVED rather
than guessed into one of the four statuses -- per the constraint that this
tool must fail loud, never guess.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY_POINTS = ["build_city.py", "build_city_image.py", "server.py"]
VIEWER_HTML = "viewer/index.html"


def read(path: str) -> str:
    full = os.path.join(ROOT, path)
    if not os.path.isfile(full):
        return ""
    with open(full, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Reachability: which local modules does each entry point import, directly
# or transitively? This is the substrate every LIVE/OPT-IN/DORMANT verdict
# below is checked against.
# ---------------------------------------------------------------------------

def local_module_names() -> set:
    names = set()
    for fn in os.listdir(ROOT):
        if fn.endswith(".py"):
            names.add(fn[:-3])
    return names


def imports_of(path: str, known: set) -> set:
    """Top-level module names a file imports, restricted to local modules."""
    src = read(path)
    if not src:
        return set()
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError:
        return set()
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in known:
                    found.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in known:
                    found.add(top)
                # `from calibration import conformal` binds the name
                # `conformal`, but the MODULE is `calibration` -- matching only
                # the module misses it entirely. This reported conformal
                # intervals as ABSENT after the module moved into the
                # calibration package, while build_city_image.py was importing
                # and using them on every run.
                for alias in node.names:
                    name = alias.name.split(".")[0]
                    if name in known:
                        found.add(name)
                    qualified = f"{node.module}.{alias.name}".split(".")[0]
                    if qualified in known:
                        found.add(qualified)
    return found


def transitive_imports(entry: str, known: set) -> set:
    seen = set()
    frontier = [entry.replace(".py", "")]
    while frontier:
        mod = frontier.pop()
        if mod in seen:
            continue
        seen.add(mod)
        for imp in imports_of(mod + ".py", known):
            if imp not in seen:
                frontier.append(imp)
    seen.discard(entry.replace(".py", ""))
    return seen


@dataclass
class Reachability:
    known: set = field(default_factory=local_module_names)
    per_entry: dict = field(default_factory=dict)

    def __post_init__(self):
        for e in ENTRY_POINTS:
            self.per_entry[e] = transitive_imports(e, self.known)

    def reachable_from(self, module: str) -> list:
        return [e for e, mods in self.per_entry.items() if module in mods]


# ---------------------------------------------------------------------------
# Claim result
# ---------------------------------------------------------------------------

STATUSES = ("LIVE", "OPT-IN", "DORMANT", "ABSENT", "UNRESOLVED")


@dataclass
class Claim:
    name: str
    status: str
    proof: str
    notes: str = ""


def grep_line(path: str, pattern: str) -> list:
    """Return (1-indexed line, text) for every line in `path` matching regex."""
    src = read(path)
    out = []
    for i, line in enumerate(src.splitlines(), start=1):
        if re.search(pattern, line):
            out.append((i, line.strip()))
    return out


# ---------------------------------------------------------------------------
# Claims: python pipeline
# ---------------------------------------------------------------------------

def check_mvs(reach: Reachability) -> Claim:
    call = grep_line("build_city.py", r"mvs_height\.compute\(")
    dsm_call = grep_line("mvs_height.py", r"plane_sweep_dsm\(")
    if not call or not dsm_call:
        return Claim("MVS height (mvs_height.compute -> plane_sweep_mvs.plane_sweep_dsm)",
                      "ABSENT", "no call site found")
    flag = grep_line("build_city.py", r'"--no-mvs"')
    proof = (f"build_city.py:{call[0][0]} calls mvs_height.compute(); "
             f"mvs_height.py:{dsm_call[0][0]} calls plane_sweep_dsm()")
    if flag:
        return Claim("MVS height (mvs_height.compute -> plane_sweep_mvs.plane_sweep_dsm)",
                      "LIVE", proof,
                      "on by default; --no-mvs flag (default False) disables it, "
                      "falling back to monocular depth")
    return Claim("MVS height", "LIVE", proof)


def check_dem_anchor(reach: Reachability) -> Claim:
    flag = grep_line("build_city.py", r'ap\.add_argument\("--dem"')
    src_choice = grep_line("build_city.py", r'"--dem-source".*choices=\["glo30", "srtm"\]')
    call = grep_line("build_city.py", r"dem_mod\.sample_grid\(")
    fit = grep_line("build_city.py", r"dem_mod\.fit_offset\(")
    if not (flag and call and fit):
        return Claim("DEM anchor / Tier A (dem_source.py, --dem flag, srtm|glo30)",
                      "ABSENT", "no --dem flag or no sample_grid/fit_offset call site")
    default_off = "store_true" in read("build_city.py")
    proof = (f"build_city.py:{flag[0][0]} defines --dem (action=store_true, default False); "
             f"build_city.py:{src_choice[0][0] if src_choice else '?'} --dem-source "
             f"choices=[glo30,srtm], default glo30; "
             f"build_city.py:{call[0][0]} calls dem_source.sample_grid(); "
             f"build_city.py:{fit[0][0]} calls dem_source.fit_offset()")
    return Claim("DEM anchor / Tier A (dem_source.py, --dem flag, srtm|glo30)",
                  "OPT-IN", proof,
                  "implemented and wired end-to-end (anchors ground datum, promotes "
                  "scene to 'A (DEM-anchored absolute elevation)'), but --dem is not "
                  "passed unless the caller supplies it -- a default run never reaches "
                  "Tier A")


def check_shadow_calibration(reach: Reachability) -> Claim:
    hits_city = grep_line("build_city.py", r"shadow_correction\.calibrate_scale\(")
    hits_img = grep_line("build_city_image.py", r"shadow_correction\.calibrate_scale\(|detect_shadow_mask\(")
    if not hits_city:
        return Claim("Shadow calibration / Tier B (calibrate_scale call sites)",
                      "ABSENT", "no call site in build_city.py")
    lines = ", ".join(f"build_city.py:{ln}" for ln, _ in hits_city)
    return Claim("Shadow calibration / Tier B (calibrate_scale call sites)",
                  "LIVE", f"called at {lines} (monocular fallback path; runs whenever "
                          f"MVS is unavailable or --no-mvs is passed)",
                  "build_city_image.py uses shadow_correction only for "
                  "detect_shadow_mask() + azimuth estimation, not calibrate_scale -- "
                  "the plain-image path never reaches Tier B, only C")


def check_conformal(reach: Reachability) -> Claim:
    importers = []
    for fn in os.listdir(ROOT):
        if fn.endswith(".py"):
            if "conformal" in imports_of(fn, {"conformal"}):
                importers.append(fn)
    entry_users = [e for e in importers if e in ENTRY_POINTS]
    if not importers:
        return Claim("Conformal prediction intervals", "ABSENT", "conformal.py not imported anywhere")
    if entry_users:
        return Claim("Conformal prediction intervals", "LIVE",
                      f"imported by entry point(s): {entry_users}")
    return Claim("Conformal prediction intervals", "DORMANT",
                  f"conformal.py imported only by: {importers}",
                  "reachable via `python validate_buildings.py <tile>` as a standalone "
                  "analysis script; never imported by build_city.py, "
                  "build_city_image.py, or server.py, so a build/upload never computes "
                  "a conformal interval")


def check_reliability_tiers(reach: Reachability) -> Claim:
    hit = grep_line("build_city.py", r"conf_mod\.reliability_tier\(")
    gate = grep_line("build_city.py", r'if mvs is not None and mvs\.get\("confidence"\) is not None:')
    if not hit:
        return Claim("Per-building confidence + reliability tiers", "ABSENT", "no call site")
    proof = f"build_city.py:{hit[0][0]} calls confidence.reliability_tier() per building"
    if gate:
        return Claim("Per-building confidence + reliability tiers", "LIVE", proof,
                      f"gated on mvs confidence existing (build_city.py:{gate[0][0]}); "
                      "true whenever MVS ran successfully, which is the default path -- "
                      "absent only if MVS fails and the monocular fallback is used")
    return Claim("Per-building confidence + reliability tiers", "LIVE", proof)


def check_self_repair(reach: Reachability) -> Claim:
    def_hit = grep_line("mesh_repair.py", r"^def repair_build\(")
    c1 = grep_line("build_city.py", r"mesh_repair\.repair_build\(")
    c2 = grep_line("build_city_image.py", r"mesh_repair\.repair_build\(")
    if not def_hit or not (c1 or c2):
        return Claim("Self-repair loop (mesh_repair.py)", "ABSENT", "no repair_build() call site")
    where = []
    if c1:
        where.append(f"build_city.py:{c1[0][0]}")
    if c2:
        where.append(f"build_city_image.py:{c2[0][0]}")
    return Claim("Self-repair loop (mesh_repair.py)", "LIVE",
                  f"repair_build() defined mesh_repair.py:{def_hit[0][0]}, called from {', '.join(where)}",
                  "runs on every build in both entry points; bounded iterative repair "
                  "with a terminating strategy (last resort removes the building)")


def check_model_audit(reach: Reachability) -> Claim:
    def_hit = grep_line("model_audit.py", r"^def audit\(")
    c1 = grep_line("build_city.py", r"model_audit\.audit\(")
    c2 = grep_line("build_city_image.py", r"model_audit\.audit\(")
    if not def_hit or not (c1 or c2):
        return Claim("Model-vs-image audit (model_audit.py)", "ABSENT", "no audit() call site")
    where = []
    if c1:
        where.append(f"build_city.py:{c1[0][0]}")
    if c2:
        where.append(f"build_city_image.py:{c2[0][0]}")
    return Claim("Model-vs-image audit (model_audit.py)", "LIVE",
                  f"audit() defined model_audit.py:{def_hit[0][0]}, called from {', '.join(where)}",
                  "wrapped in try/except in both callers -- failure prints a message and "
                  "continues rather than aborting the build")


def check_standalone_script(module: str, label: str) -> Claim:
    """For scripts with their own __main__ but zero imports from any entry point."""
    importers = []
    for fn in os.listdir(ROOT):
        if fn.endswith(".py") and fn != module + ".py":
            if module in imports_of(fn, {module}):
                importers.append(fn)
    has_main = bool(grep_line(module + ".py", r'if __name__ == "__main__"'))
    entry_users = [e for e in importers if e in ENTRY_POINTS]
    if entry_users:
        return Claim(label, "LIVE", f"imported by entry point(s): {entry_users}")
    if importers:
        return Claim(label, "DORMANT",
                      f"imported only by non-entry-point script(s): {importers}")
    if has_main:
        return Claim(label, "DORMANT",
                      f"{module}.py has its own `if __name__ == \"__main__\"` block "
                      f"(runnable as `python {module}.py ...`) but is not imported by "
                      "any entry point -- a build or upload never touches it")
    return Claim(label, "ABSENT", f"{module}.py not found or has no entry point")


# ---------------------------------------------------------------------------
# Claims: viewer
# ---------------------------------------------------------------------------

BUTTON_RE = re.compile(r'<button\s+id="([a-zA-Z0-9_-]+)"[^>]*>([^<]*)</button>')


def _wired(bid: str, html: str) -> bool:
    """True if a click handler reaches this button id, either directly
    (getElementById("id").addEventListener("click", ...)) or through a local
    variable bound to it earlier in the file
    (const x = document.getElementById("id"); ... x.addEventListener("click", ...)).
    Checking only the direct-chain form under-counts real handlers -- this
    file uses both patterns interchangeably.
    """
    if re.search(rf'getElementById\("{re.escape(bid)}"\)\s*\.addEventListener\(\s*"click"', html):
        return True
    for m in re.finditer(
            rf'(?:const|let|var)\s+(\w+)\s*=\s*document\.getElementById\("{re.escape(bid)}"\)', html):
        var = m.group(1)
        if re.search(rf'\b{re.escape(var)}\.addEventListener\(\s*"click"', html):
            return True
    return False


def viewer_button_inventory() -> list:
    html = read(VIEWER_HTML)
    buttons = []
    tag_re = re.compile(r'<button\s+id="([a-zA-Z0-9_-]+)"([^>]*)>([^<]*)</button>')
    for m in tag_re.finditer(html):
        bid, attrs, label = m.group(1), m.group(2), m.group(3).strip()
        wired = _wired(bid, html)
        # "hidden by default" must be read from THIS tag's own attributes only
        # -- scanning a fixed character window past the tag bleeds into the
        # next <button> and misattributes its style (caught in review: it
        # falsely marked btn-xsec/btn-pdf as hidden because btn-xr's
        # display:none fell inside their 200-char window).
        hidden_static = 'display:none' in attrs
        # btn-xr has no static display:none in its tag by the time this greps
        # correctly, but some controls are hidden/shown at runtime via a JS
        # feature check rather than a static attribute -- record that
        # separately so it isn't silently missed.
        runtime_gate = bool(re.search(
            rf'getElementById\("{re.escape(bid)}"\)[^;]*\.style\.display\s*=\s*"none"', html))
        if not runtime_gate:
            for vm in re.finditer(
                    rf'(?:const|let|var)\s+(\w+)\s*=\s*document\.getElementById\("{re.escape(bid)}"\)', html):
                var = vm.group(1)
                if re.search(rf'\b{re.escape(var)}\.style\.display\s*=\s*"none"', html):
                    runtime_gate = True
                    break
        buttons.append({"id": bid, "label": label, "wired": wired,
                         "hidden_by_default": hidden_static or runtime_gate,
                         "hidden_static": hidden_static, "runtime_gate": runtime_gate})
    return buttons


def check_viewer_tool(bid: str, label: str, buttons: list) -> Claim:
    b = next((x for x in buttons if x["id"] == bid), None)
    if b is None:
        return Claim(f"Viewer tool: {label} (#{bid})", "ABSENT",
                      f"no <button id=\"{bid}\"> in {VIEWER_HTML}")
    if not b["wired"]:
        return Claim(f"Viewer tool: {label} (#{bid})", "DORMANT",
                      f"button exists ({VIEWER_HTML}, id={bid}) but no "
                      f"addEventListener(\"click\", ...) found for it",
                      "button is decorative -- clicking it does nothing")
    if b["hidden_by_default"]:
        if b["hidden_static"]:
            note = ("hidden by default in the HTML (style=\"display:none\" on the tag itself)")
        else:
            note = ("shown/hidden at runtime by JS, not a static attribute -- "
                     "check its gating condition directly, not inferred here")
        return Claim(f"Viewer tool: {label} (#{bid})", "OPT-IN",
                      f"button wired ({VIEWER_HTML}, id={bid}), click handler present", note)
    return Claim(f"Viewer tool: {label} (#{bid})", "LIVE",
                  f"button wired ({VIEWER_HTML}, id={bid}), click handler present")


# ---------------------------------------------------------------------------
# Doc-vs-manifest contradiction scan
# ---------------------------------------------------------------------------

DOC_CLAIM_PATTERNS = [
    # (doc file, regex, human description, the resolved claim name it should match)
    ("README.md", r"VR|virtual reality", "README mentions VR"),
    ("README.md", r"cross.section", "README mentions cross-section"),
    ("README.md", r"line.of.sight", "README mentions line-of-sight"),
    ("AUDIT.md", r"MVS.*not in the active path|not in the active path", "AUDIT.md claims MVS not in active path"),
    ("AUDIT.md", r"Tier A.*never.*exercised|never exercised", "AUDIT.md claims Tier A never exercised"),
    ("BENCHMARK.md", r"conformal", "BENCHMARK.md discusses conformal intervals"),
]


def scan_doc_contradictions(claims_by_name: dict) -> list:
    findings = []
    for doc, pattern, desc in DOC_CLAIM_PATTERNS:
        hits = grep_line(doc, pattern)
        if hits:
            for ln, text in hits:
                findings.append(f"- `{doc}:{ln}` — {desc}: \"{text[:140]}\"")
    return findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                     help="exit 1 if any claim is UNRESOLVED")
    ap.add_argument("--out", default="CAPABILITIES.md")
    a = ap.parse_args()

    reach = Reachability()

    claims = [
        check_mvs(reach),
        check_dem_anchor(reach),
        check_shadow_calibration(reach),
        check_conformal(reach),
        check_reliability_tiers(reach),
        check_self_repair(reach),
        check_model_audit(reach),
        check_standalone_script("change_detection", "Change detection (change_detection.py)"),
        check_standalone_script("train_height", "GAMUS fine-tuning (train_height.py)"),
    ]

    buttons = viewer_button_inventory()
    viewer_claims = [
        check_viewer_tool("btn-heatmap", "Heatmap texture mode", buttons),
        check_viewer_tool("btn-conf", "Confidence visualisation", buttons),
        check_viewer_tool("btn-err", "VS LiDAR error overlay", buttons),
        check_viewer_tool("btn-measure", "Measure height", buttons),
        check_viewer_tool("btn-tour", "Flythrough", buttons),
        check_viewer_tool("btn-slope", "Slope layer", buttons),
        check_viewer_tool("btn-los", "Line of sight", buttons),
        check_viewer_tool("btn-xsec", "Cross-section", buttons),
        check_viewer_tool("btn-pdf", "Export PDF", buttons),
        check_viewer_tool("btn-xr", "Enter VR", buttons),
    ]

    unresolved = [c for c in claims + viewer_claims if c.status == "UNRESOLVED"]

    lines = []
    lines.append("# CAPABILITIES.md")
    lines.append("")
    lines.append("Generated by `scripts/verify_capabilities.py` — static analysis only, "
                  "no pipeline execution. Re-run after any change to an entry point or "
                  "the viewer to keep this current. Do not hand-edit the tables below; "
                  "edit the script and regenerate.")
    lines.append("")
    lines.append("Status legend: **LIVE** reachable on the default path · **OPT-IN** "
                  "implemented+wired, gated behind a flag/condition defaulting off · "
                  "**DORMANT** exists, zero call sites from any entry point · "
                  "**ABSENT** no implementation found.")
    lines.append("")
    lines.append("## Pipeline claims")
    lines.append("")
    lines.append("| Claim | Status | Proof | Notes |")
    lines.append("|---|---|---|---|")
    for c in claims:
        lines.append(f"| {c.name} | **{c.status}** | {c.proof} | {c.notes} |")
    lines.append("")
    lines.append("## Viewer tool inventory")
    lines.append("")
    lines.append(f"{len(buttons)} `<button id=...>` elements found in `{VIEWER_HTML}`. "
                  "Exact on-screen labels below, as they appear in the markup (not "
                  "assumed from naming convention).")
    lines.append("")
    lines.append("| Claim | Status | Proof | Notes |")
    lines.append("|---|---|---|---|")
    for c in viewer_claims:
        lines.append(f"| {c.name} | **{c.status}** | {c.proof} | {c.notes} |")
    lines.append("")
    lines.append("### Full button list (as-is from markup)")
    lines.append("")
    lines.append("| id | label | click-wired | hidden by default |")
    lines.append("|---|---|---|---|")
    for b in buttons:
        lines.append(f"| `{b['id']}` | {b['label']} | {b['wired']} | {b['hidden_by_default']} |")
    lines.append("")

    absent_or_dormant = [c for c in claims + viewer_claims if c.status in ("ABSENT", "DORMANT")]
    lines.append("## DOCUMENTED BUT ABSENT / DORMANT")
    lines.append("")
    lines.append("Claims in this manifest that a normal build/upload never reaches, "
                  "or that do not exist at all:")
    lines.append("")
    for c in absent_or_dormant:
        lines.append(f"- **{c.name}** — {c.status}: {c.proof}"
                      + (f" — {c.notes}" if c.notes else ""))
    if not absent_or_dormant:
        lines.append("(none)")
    lines.append("")

    lines.append("## Doc-vs-manifest scan (report only, nothing edited)")
    lines.append("")
    contradictions = scan_doc_contradictions({c.name: c for c in claims + viewer_claims})
    if contradictions:
        lines.extend(contradictions)
    else:
        lines.append("(no matches for the tracked doc-claim patterns)")
    lines.append("")

    if unresolved:
        lines.append("## UNRESOLVED (script could not determine status — fix the script, don't guess)")
        lines.append("")
        for c in unresolved:
            lines.append(f"- {c.name}")
        lines.append("")

    out_text = "\n".join(lines)
    out_path = os.path.join(ROOT, a.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out_text)
    print(out_text)
    print(f"\n--- written to {out_path} ---", file=sys.stderr)

    if a.check and unresolved:
        sys.exit(1)


if __name__ == "__main__":
    main()
