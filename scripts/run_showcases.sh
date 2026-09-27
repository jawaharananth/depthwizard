#!/bin/bash
# Builds the showcase places one at a time, after any running place build.
cd "$(dirname "$0")/.."
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'build_places|build_showcase.py .*--fetch-only' }) { exit 0 } else { exit 1 }"; do sleep 60; done
while IFS='|' read id lat lon kind name region; do
  ./venv/Scripts/python.exe build_showcase.py "$id" --lat $lat --lon $lon --kind $kind --name "$name" --region "$region"
done < scripts/showcase_list.txt
