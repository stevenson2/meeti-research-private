#!/bin/bash
# R14 Tier-1 generic AF-database downloader (idempotent resume).
# Usage: bash r14_more_afdb_download.sh <afpdb|aftdb>
# Handles nested record paths (aftdb: learning-set/n01) by mkdir per record.
set -u
DB="${1:?usage: $0 <afpdb|aftdb>}"
DST="/home/devcontainers/ecg_data/$DB"
BASE="https://archive.physionet.org/physiobank/database/$DB"
mkdir -p "$DST"
curl -sf "$BASE/RECORDS" -o "$DST/RECORDS" || { echo "FETCH-RECORDS-FAIL $DB"; exit 1; }
n_total=0; nodat=""
for r in $(cat "$DST/RECORDS"); do
  n_total=$((n_total+1))
  mkdir -p "$DST/$(dirname "$r")"
  [ -s "$DST/$r.hea" ] || curl -sf "$BASE/$r.hea" -o "$DST/$r.hea"
  # parse .dat file name(s) from header signal lines (field 1, non-comment);
  # hea dat names are RELATIVE to the record's own directory (aftdb nests
  # records under learning-set/...), so keep the record's dirname in the URL
  if [ -s "$DST/$r.hea" ]; then
    rdir=$(dirname "$r")
    for dat in $(grep -v '^#' "$DST/$r.hea" | awk 'NF>=5 {print $1}' | grep '\.dat$'); do
      if [ ! -s "$DST/$rdir/$dat" ]; then
        curl -sf "$BASE/$rdir/$dat" -o "$DST/$rdir/$dat" || nodat="$nodat $rdir/$dat"
      fi
    done
  fi
  for ext in atr qrs qrsc; do
    [ -s "$DST/$r.$ext" ] || curl -sf "$BASE/$r.$ext" -o "$DST/$r.$ext" || true
  done
done
n_dat=$(find "$DST" -name '*.dat' -size +1k 2>/dev/null | wc -l)
echo "DONE $DB: records=$n_total dat_files=$n_dat"
echo "no-signal:$nodat"
du -sh "$DST"
