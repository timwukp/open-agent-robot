#!/usr/bin/env bash
# open-agent-robot — node runner for the EC2 side.
#   ./run_ec2.sh start [node…]   (default: gateway body)
#   ./run_ec2.sh start-all       (every node in the table — models download on first run)
#   ./run_ec2.sh stop | status | logs <node>
# Pidfile-based (never pkill by pattern — it can match the invoking shell).
#
# OAR_DRY=1 prints what would be started instead of starting it, and skips the
# ROS/venv preconditions, so infra/check_deploy.sh can verify the real inventory
# on a laptop instead of parsing this file and hoping the parse matches. That is
# also why the node table is a here-doc and not `declare -A`: macOS ships bash
# 3.2, which has no associative arrays, so the guard could never run the script.
set -euo pipefail
cd "$(dirname "$0")/.."
DRY=${OAR_DRY:-}
RUN=${OAR_RUN:-/tmp/oar}
PY="$HOME/venv/bin/python"
if [ -z "$DRY" ]; then
  mkdir -p "$RUN"
  set +u; source /opt/ros/jazzy/setup.bash; set -u  # ROS setup.bash trips set -u
fi

# --- the node inventory: name|command, one per line, single source of truth ---
# ALL is DERIVED from column 1 below — never a second hand-maintained list. It
# used to be one, and `yue` was missing from it: the Cantonese engine was never
# started by start-all and never reported by status, so Cantonese text hit a
# closed socket in speech/tts_node.py:_sentences_yue (which has no fallback) on
# every clean boot. Deriving makes that defect unrepresentable, not just tested.
NODES=$(cat <<EOF
gateway|$PY gateway/ws_gateway.py
body|env MUJOCO_GL=egl $PY body/body_node.py --xml body/assets/unitree_g1/scene_with_hands.xml
asr|$PY speech/asr_node.py
tts|$PY speech/tts_node.py
hand|$PY perception/hand_node.py
detect|$PY perception/detect_node.py
yue|$HOME/venv-yue/bin/python speech/yue_server.py
agent|$PY agent/agent_node.py
game|$PY game/game_node.py
EOF
)
cmd_for() { printf '%s\n' "$NODES" | awk -F'|' -v n="$1" '$1==n{print $2; f=1} END{exit !f}'; }
ALL=$(printf '%s\n' "$NODES" | awk -F'|' 'NF{printf "%s ", $1}')
CORE="gateway body"

start_one() {
  local n=$1 cmd
  if ! cmd=$(cmd_for "$n"); then
    echo "$n: unknown node (known: $ALL)" >&2; return 1
  fi
  if [ -n "$DRY" ]; then
    echo "$n: would run $cmd"; return
  fi
  if [ -f "$RUN/$n.pid" ] && kill -0 "$(cat "$RUN/$n.pid")" 2>/dev/null; then
    echo "$n: already running"; return
  fi
  nohup $cmd > "$RUN/$n.log" 2>&1 &
  echo $! > "$RUN/$n.pid"
  echo "$n: started (pid $!)"
}

case "${1:-status}" in
  start)      shift; for n in ${*:-$CORE}; do start_one "$n"; done ;;
  start-all)  for n in $ALL; do start_one "$n"; done ;;
  stop)
    for f in "$RUN"/*.pid; do
      [ -f "$f" ] || continue
      pid=$(cat "$f")
      kill "$pid" 2>/dev/null || true
      for _ in 1 2 3; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -9 "$pid" 2>/dev/null && echo "SIGKILLed $(basename "$f" .pid)" \
        || echo "stopped $(basename "$f" .pid)"
      rm -f "$f"
    done ;;
  status)
    for n in $ALL; do
      if [ -f "$RUN/$n.pid" ] && kill -0 "$(cat "$RUN/$n.pid")" 2>/dev/null; then
        echo "$n: UP (pid $(cat "$RUN/$n.pid"))"
      else
        echo "$n: down"
      fi
    done
    [ -n "$DRY" ] || nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true ;;
  logs) tail -n 40 "$RUN/${2:?node name}.log" ;;
  *) echo "usage: $0 {start [nodes]|start-all|stop|status|logs <node>}"; exit 1 ;;
esac
