# Data in this repository — sources and licences

Binary data is committed here so the demo runs without a download step. Each
source has a different licence and they are not interchangeable.

## `gamus_data/`, `gamus_probe/` — GAMUS

**Licence: CC-BY-4.0.** Redistributable *with attribution*, which is what this
file provides.

> Xiong, Z., Chen, S., Wang, Y., Mou, L., Zhu, X.X.
> *GAMUS: A Geometry-aware Multi-modal Semantic Segmentation Benchmark for
> Remote Sensing Data.* arXiv:2305.14914 (2023).
> Dataset: https://huggingface.co/datasets/earthflow/GAMUS
> Code: https://github.com/EarthNets/RSI-MMSegmentation

What is here is a small subset of a 80 GB dataset — the tiles used to measure
the domain gap recorded in `BENCHMARK.md`, plus a training slice. The full set,
or any other subset, is reproducible:

```bash
python gamus.py download --split train --limit 400
python gamus.py download --split val --limit 40
```

Tiles are `*_RGB.h5` (imagery), `*_AGL.h5` (height above ground, in metres,
LiDAR-derived) and `*_CLS.h5` (six semantic classes).

## `scenes/` — pre-built reconstructions

Output of this pipeline. `scenes/jax165` is derived from DFC2019 imagery (see
below); `scenes/varanasi` is derived from OpenAerialMap imagery.

## `uploads/` — test input

⚠️ **Check this before publishing.** The image here was uploaded through the
dashboard for testing and appears to be derived from DFC2019 imagery.

**DFC2019 is licensed by IEEE GRSS and is not redistributable.** This
repository's `.gitignore` excludes `dfc2019_data/` for exactly that reason, so
committing a derivative of it is inconsistent with that position. It is kept
here because it was explicitly requested; if this repository is submitted or
made public, remove `uploads/` and the `scenes/jax165` textures, or replace them
with imagery you hold redistribution rights to.

Obtain DFC2019 directly from
[IEEE DataPort](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019).

## `stage_timings.json` — ETA calibration

Measured stage durations from **one specific machine** (Intel Core 5 120U, 15 W,
no discrete GPU). They are stored per megapixel, so they scale with image size,
but not across hardware: a machine with a CUDA GPU is roughly an order of
magnitude faster on the depth stage, which is ~94% of a build.

Delete the file to recalibrate. `progress.py` shows no ETA until a stage has
been timed rather than displaying a fabricated one, so an empty calibration is
handled correctly and is safer than an inherited wrong one.

## Not committed

`checkpoints/` — fine-tuned model weights. The only checkpoint produced so far
is from the smoke test (1 epoch, 2 tiles, CPU) and scored **worse than the
pretrained baseline**: MAE 7.16 m against 6.68 m, correlation −0.121 against
0.222. `train_height.py` refuses to recommend it. At 372 MB in a single file it
also exceeds GitHub's 100 MB limit and would require Git LFS.

`dfc2019_data/` — 16 GB, IEEE GRSS licensed.
