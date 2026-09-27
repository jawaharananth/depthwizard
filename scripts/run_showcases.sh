#!/bin/bash
# Builds the showcase places one at a time, after any running place build.
# Python's stdin is /dev/null: otherwise the child reads the rest of the place
# list and the loop stops after one place.
cd "$(dirname "$0")/.."
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'build_places|build_showcase.py .*--fetch-only' }) { exit 0 } else { exit 1 }"; do sleep 60; done
while IFS='|' read -r id lat lon kind name region; do
  ./venv/Scripts/python.exe build_showcase.py "$id" --lat "$lat" --lon "$lon" --kind "$kind" \
      --name "$name" --region "$region" < /dev/null
done < scripts/showcase_places.txt
