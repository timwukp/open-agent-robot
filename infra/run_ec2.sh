#!/usr/bin/env bash
# open-agent-robot — node runner for the EC2 side.
#   ./run_ec2.sh start [node…]   (default: gateway body)
#   ./run_ec2.sh start-all       (adds asr tts hand agent game — model downloads on first run)
#   ./run_ec2.sh stop | status | logs <node>
# Pidfile-based (never pkill by pattern — it can match the invoking shell).
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=/tmp/oar; mkdir -p "$RUN"
PY="$HOME/venv/bin/python"
set +u; source /opt/ros/jazzy/setup.bash; set -u  # ROS setup.bash trips set -u

declare -A CMD=(
  [gateway]="$PY gateway/ws_gateway.py"
  [body]="env MUJOCO_GL=egl $PY body/body_node.py --xml body/assets/unitree_g1/scene_with_hands.xml"
  [asr]="$PY speech/asr_node.py"
  [tts]="$PY speech/tts_node.py"
  [hand]="$PY perception/hand_node.py"
  [detect]="$PY perception/detect_node.py"
  [yue]="$HOME/venv-yue/bin/python speech/yue_server.py"
  [agent]="$PY agent/agent_node.py"
  [game]="$PY game/game_node.py"
)
CORE="gateway body"
ALL="gateway body asr tts hand detect agent game"

start_one() {
  local n=$1
  if [ -f "$RUN/$n.pid" ] && kill -0 "$(cat "$RUN/$n.pid")" 2>/dev/null; then
    echo "$n: already running"; return
  fi
  nohup ${CMD[$n]} > "$RUN/$n.log" 2>&1 &
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
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true ;;
  logs) tail -n 40 "$RUN/${2:?node name}.log" ;;
  *) echo "usage: $0 {start [nodes]|start-all|stop|status|logs <node>}"; exit 1 ;;
esac
