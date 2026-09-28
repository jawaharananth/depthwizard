#!/bin/bash
# Builds the showcase places one at a time (they share viewer/output for
# staging, so never two at once): waits for any showcase build already
# running, skips places already saved with their cursor rasters, and lets each
# build fetch whatever of its place is not cached yet. Python's stdin is
# /dev/null: otherwise the child reads the rest of the place list and the
# loop stops after one place.
cd "$(dirname "$0")/.."
busy() {
  powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'build_showcase\.py' }) { exit 0 } else { exit 1 }"
}
while IFS='|' read -r id lat lon kind name region; do
  while busy; do sleep 60; done
  # Built means metric: a scene that fell back to assumed heights is rebuilt.
  if [ -f "scenes/$id/hover_agl.u16" ] && grep -q '"height_is_metric": true' "scenes/$id/scene.json"; then
    echo "[queue] $id already built"; continue
  fi
  ./venv/Scripts/python.exe build_showcase.py "$id" --lat "$lat" --lon "$lon" --kind "$kind" \
      --name "$name" --region "$region" < /dev/null
done < scripts/showcase_places.txt
