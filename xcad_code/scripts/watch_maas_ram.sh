#!/bin/bash
# watch_maas_ram.sh -- RSS/GPU sampler for the maas test-phase OOM hunt.
# Runs the given command, samples the child's VmRSS + GPU memory every 0.5 s,
# prints a trace so we can see WHICH phase eats the 123 GB.
set -u
OUT=/tmp/maas_ram_trace.txt
: > "$OUT"
"$@" &
pid=$!
peak=0
while kill -0 "$pid" 2>/dev/null; do
  rss=$(awk '/VmRSS/{print $2}' /proc/$pid/status 2>/dev/null)
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "${rss:-}" ]; then
    [ "$rss" -gt "$peak" ] && peak=$rss
    echo "$(date -u +%H:%M:%S) rss_MB=$((rss/1024)) gpu_MB=${gpu}" >> "$OUT"
  fi
  sleep 0.5
done
wait "$pid"; rc=$?
echo "exit=$rc peak_rss_MB=$((peak/1024))" | tee -a "$OUT"
tail -40 "$OUT"
