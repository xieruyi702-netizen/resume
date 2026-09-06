#!/usr/bin/env bash
# 扫参：mode × shards × rate，JMeter 找最大端到端 QPS
# 用法: bash bench/run-rate-sweep.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/bench"
CAP="${CAP:-200}"
OUT="sweep-results.csv"
echo "mode,shards,rate,throughput,samples,passed,rejected,mean_ms,p99_ms" > "$OUT"

set_both() {
  local mode="$1" shards="$2" rate="$3"
  for app in seckill-app1 seckill-app2; do
    docker exec "$app" wget -qO- --post-data='' \
      "http://127.0.0.1:8080/admin/rate?mode=${mode}&capacity=${CAP}&ratePerSec=${rate}&shards=${shards}" \
      >/dev/null || true
  done
}

reset_env() {
  for i in 1 2 3; do
    curl -s -X POST "http://localhost:8080/admin/reset?stock=100000" >/dev/null || true
  done
  sleep 0.5
}

run_one() {
  local mode="$1" shards="$2" rate="$3"
  local tag="${mode}_s${shards}_r${rate}"
  rm -rf "sweep-${tag}.jtl" "sweep-${tag}-report"
  reset_env
  set_both "$mode" "$shards" "$rate"
  jmeter -n -t rate-limit-compare.jmx -JHOST=localhost -JPORT=8080 \
    -l "sweep-${tag}.jtl" -e -o "sweep-${tag}-report" >/dev/null
  local stats passed rejected
  stats=$(python3 - <<PY
import json
d=json.load(open("sweep-${tag}-report/statistics.json"))
t=d.get("Total") or d.get("POST seckill")
print(f"{t['throughput']:.1f},{t['sampleCount']},{t['meanResTime']:.1f},{t.get('pct2ResTime', t.get('pct3ResTime', 0)):.1f}")
PY
)
  passed=$(docker exec seckill-app1 wget -qO- http://127.0.0.1:8080/admin/rate | python3 -c "import sys,json; print(json.load(sys.stdin)['passed'])")
  rejected=$(docker exec seckill-app1 wget -qO- http://127.0.0.1:8080/admin/rate | python3 -c "import sys,json; print(json.load(sys.stdin)['rejected'])")
  # throughput,samples,mean,p99
  IFS=',' read -r thr samples mean p99 <<< "$stats"
  echo "${mode},${shards},${rate},${thr},${samples},${passed},${rejected},${mean},${p99}" | tee -a "$OUT"
}

echo "=== rate sweep CAP=$CAP (reduced grid) ==="
for mode in token leaky; do
  for shards in 1 8 32; do
    for rate in 100 2000 50000; do
      echo "-- $mode shards=$shards rate=$rate --"
      run_one "$mode" "$shards" "$rate"
    done
  done
done

echo
echo "=== top 10 by throughput ==="
python3 - <<'PY'
import csv
rows=list(csv.DictReader(open("sweep-results.csv")))
rows.sort(key=lambda r: float(r["throughput"]), reverse=True)
for r in rows[:10]:
    print(f"{r['throughput']:>8} req/s  mode={r['mode']:5} shards={r['shards']:>2} rate={r['rate']:>5}  "
          f"pass={r['passed']} rej={r['rejected']} mean={r['mean_ms']}ms")
print("best:", rows[0])
PY
