---
title: DepthWizard
sdk: docker
app_port: 8080
pinned: true
---

# DepthWizard — one satellite image to a measured 3D city

Smart India Hackathon 2026 · SIH26175 (ISRO SAC) · Team Vantage Point

- **Explore** seven Indian places already built: HITEC City, Lower Parel (Mumbai), Darjeeling, Gangtok,
  Periyar, Kanha and a Jaisalmer village. Hover any roof for its height and error bar; try the flood,
  slope and measuring tools.
- **Upload** your own image (GeoTIFF, PNG or JPG, up to 95 MB). A GeoTIFF returns an absolute DSM on its
  own grid. On this free server a 1 km scene takes about 6–10 minutes; progress is shown live.

Accuracy, measured on data the model never saw: 1.87 m per building (458 buildings, airborne LiDAR);
0.13–1.50 m against the reference DEM on ISRO's metric at 8 Indian places.
