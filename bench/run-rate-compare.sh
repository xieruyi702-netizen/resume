#!/usr/bin/env bash
# JMeter 端到端对比令牌桶 / 漏桶（Nginx → 双实例）
# 用法: bash bench/run-rate-compare.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/bench"

CAP="${CAP:-200}"
RATE="${RATE:-2000}"
THREADS="${THREADS:-64}"
DURATION="${DURATION:-15}"

set_rate() {
  local mode="$1"
  for app in seckill-app1 seckill-app2; do
    docker exec "$app" wget -qO- \
      "http://127.0.0.1:8080/admin/rate?mode=${mode}&capacity=${CAP}&ratePerSec=${RATE}" \
      --post-data='' >/dev/null
  done
  echo "  both apps -> mode=$mode capacity=$CAP rate=$RATE/s"
}

rate_stats() {
  echo "  app1: $(docker exec seckill-app1 wget -qO- http://127.0.0.1:8080/admin/rate)"
  echo "  app2: $(docker exec seckill-app2 wget -qO- http://127.0.0.1:8080/admin/rate)"
}

reset_env() {
  # 经 Nginx 一次即可（Redis/DB 共享）；多打几次确保至少一实例执行完
  for i in 1 2 3; do
    curl -s -X POST "http://localhost:8080/admin/reset?stock=100000" >/dev/null || true
  done
  sleep 1
}

run_one() {
  local mode="$1"
  local out="results-rate-${mode}"
  rm -rf "${out}.jtl" "${out}-report"
  echo "==== mode=$mode ===="
  reset_env
  set_rate "$mode"
  rate_stats

  jmeter -n -t rate-limit-compare.jmx \
    -JHOST=localhost -JPORT=8080 \
    -l "${out}.jtl" -e -o "${out}-report" \
    >/dev/null

  echo "  jmeter summary:"
  # 汇总行：label,samples,...; 取 TOTAL 行
  if [[ -f "${out}-report/statistics.json" ]]; then
    python3 - <<PY
import json
d=json.load(open("${out}-report/statistics.json"))
t=d.get("Total") or d.get("POST seckill")
print(f"  samples={t['sampleCount']}  errorPct={t['errorPct']:.2f}%  "
      f"throughput={t['throughput']:.1f} req/s  "
      f"mean={t['meanResTime']:.1f}ms  p99={t.get('pct2ResTime', t.get('pct3ResTime', 0)):.1f}ms")
PY
  else
    tail -5 "${out}.jtl" || true
  fi
  echo "  after:"
  rate_stats
}

echo "JMeter rate-limit compare: threads(plan)=64 duration=15s cap=$CAP rate=$RATE/s × 2 instances"
run_one token
run_one leaky
echo "done. reports: bench/results-rate-token-report/  bench/results-rate-leaky-report/"
