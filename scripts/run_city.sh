#!/usr/bin/env bash
# Full city run, unattended: download -> detect -> evaluate. Each step resumes.
cd "$(dirname "$0")/.."
P="/c/Program Files/ArcGIS/Pro/bin/Python/Scripts/propy.bat"
log() { echo "$(date '+%m-%d %H:%M') $*" >> outputs/log_run_city.txt; }
log "start"
until grep -qE "^done" outputs/log_download.txt 2>/dev/null; do
  if ! tasklist | grep -qi python; then        # download not running: (re)start it
    log "download (re)started"; "$P" scripts/download_naip.py > outputs/log_download.txt 2>&1
  fi
  sleep 30
done
log "download done"
"$P" scripts/detect_buildings.py > outputs/log_detect.txt 2>&1; log "detect exit $?"
"$P" scripts/evaluate.py > outputs/log_evaluate.txt 2>&1; log "evaluate exit $?"
log "finished"
