# Viewer roadmap — vs zaidnansari2011's live UI (2026-09-23)

Source: 4 real screenshots of `project5.zaidansari.tech` shared by the user,
compared against this project's own viewer. Ranked by real cost, not wishlist
order.

## Cheap, real, buildable now (no new ML)

1. **Scene metadata panel** — area covered, pixel size (GSD), grid size,
   height range. Pure computation from data already in hand per build
   (`gsd`, `ndsm.shape`, height min/max). No new inference needed.
2. **Height-boost slider** — vertical exaggeration for readability, explicitly
   decoupled from the real numbers ("heights are stretched to make the shape
   readable, the numbers stay true"). Pure viewer/Three.js feature: scale the
   mesh's Y axis for display, report real (unscaled) values in every readout.
3. **Sun-direction slider** — live relighting, viewer-only (rotate the
   directional light, no data dependency).
4. **"WHERE THIS IS" georeferencing panel** — lat/lon, UTM zone, datum when
   the scene is DEM-anchored (already has real geo data on that path, just
   not surfaced this way), or an honest "ships without georeferencing" state
   on Tier C scenes with none. Moderate effort (wiring, not new data).
5. **Honest small-N caveats in copy** — e.g. "across N buildings, too few to
   be a stable figure." Cheap: pipe the per-building audit's own counts
   (`model_audit.py`, built earlier this session) into UI text.

## Hard, real, structurally gated — DO NOT treat as quick

6. **Per-pixel organic roof bumps** (visible in their Omaha screenshot).
   Requires a genuinely dense per-pixel height field surviving into the mesh,
   not a flat height-per-building.

   CORRECTED 2026-09-23 (earlier note here was wrong in scope): the flat
   0.02-0.17 m std measured on Karnataka was on the STOCK fallback backbone
   (the metric checkpoint was REFUSED there: scene's 99th percentile came out
   2.12 m, implausible). Re-checked directly on a scene where the real
   GAMUS-tuned metric model IS accepted, using real DFC2019 LiDAR
   (JAX_004) as ground truth: per-pixel std within real building footprints
   is 0.6-2.1 m -- genuinely NOT flat.

   BUT pixel-wise correlation between the model's own variation and the real
   LiDAR roof shape, same footprints: +0.573, +0.167, +0.020, +0.009, -0.090.
   Mostly weak-to-zero, one moderate outlier. So the honest finding is
   neither "flat noise floor" (Karnataka) nor "real roof-pitch signal"
   (would need consistently high correlation) -- it's closer to plausible-
   looking texture that mostly does not track the real building. Rendering
   it as organic bumps would visually resemble zaidnansari's screenshots but
   would be fabricating detail more often than reconstructing it -- the
   exact risk this project's whole session-long discipline exists to avoid
   (same failure mode flagged for VIGA).
   Not a Colab/GPU question either way -- Colab buys compute, not signal
   quality. This is a data problem, un-fixed by more compute.
   Also gated on: even where the signal DID correlate, `mesh_repair.
   repair_build` currently collapses every building to one constant height
   on purpose (for defect-repair robustness) -- rendering organic roofs
   would need a second, unrepaired height-field mesh per building alongside
   the flat prism, a real architecture change.

   Honest verdict: not doing this. The upside (visual similarity to a
   competitor's screenshot) is outweighed by the downside (rendering
   texture as if it were measured roof structure, on a signal proven weakly
   correlated with the real thing).
