# Demo Runbook — 石頭剪刀布 with the virtual robot

Topology: EC2 g4dn.xlarge (ap-southeast-1) runs the robot's brain+body
(ROS 2 nodes + MuJoCo); the MacBook is its eye/ear/mouth (C daemon) and the
display (browser). All traffic rides one SSH tunnel; the only open port on
EC2 is 22, restricted to your current IP.

## Step 0 — EC2 up (skip if already running)

Your instance id / region / public IP live in `infra/local.env`, which
`infra/provision.sh` generates and git ignores (they are account-specific and
the public IP rotates on every stop/start).

```bash
source infra/local.env
aws ec2 start-instances --instance-ids "$OAR_INSTANCE_ID" --region "$OAR_REGION"
# wait ~40s, then refresh ssh_config + local.env with the new IP:
infra/provision.sh          # idempotent: also re-points the SG at your current IP
```
- First time on a fresh account: run `infra/provision.sh` with no local.env — it
  creates the key pair, security group, IAM role and instance, then writes both
  `infra/ssh_config` and `infra/local.env`.
- SSH timeouts later usually mean *your* IP rotated → re-run `provision.sh`.

## Step 0.5 — one-time install on the instance (skip on an instance you already set up)

Run **on the instance**, and note it is three stages, not two — the third is what
makes Cantonese work at all:

```bash
ssh -F infra/ssh_config robot
cd open-agent-robot
./infra/setup-ec2.sh driver     # NVIDIA driver; then `sudo reboot` and reconnect
./infra/setup-ec2.sh stack      # ROS 2 Jazzy + ~/venv + the G1 body model
                                #   (ends by running body/fetch_assets.sh — without it
                                #    body_node dies: the G1 XML is third-party and is
                                #    deliberately not committed)
./infra/setup-ec2.sh yue        # ~/venv-yue for Cantonese TTS; skip only if you
                                #   never speak Cantonese to it. The first Cantonese
                                #   sentence then downloads ~12 GB of weights.
```

Verify before starting anything: `./infra/check_deploy.sh` must print `0 findings`.
It fails if the runner would launch a path no install stage produces.

## Step 1 — robot nodes on EC2 (skip if `status` shows 9× UP)
```bash
ssh -F infra/ssh_config -o ClearAllForwardings=yes robot \
    'open-agent-robot/infra/run_ec2.sh start-all'
ssh -F infra/ssh_config -o ClearAllForwardings=yes robot \
    'open-agent-robot/infra/run_ec2.sh status'
# expect all 9 UP: gateway body asr tts hand detect yue agent game
```
`-o ClearAllForwardings=yes` is needed on every *command* invocation once the Step 2
tunnel exists: `infra/ssh_config` carries the 8080/8765 forwards on the `robot` host,
so a second connection tries to bind them again and dies with
`bind [127.0.0.1]:8080: Address already in use … Could not request local forwarding`
— which looks like a broken instance and is only a port collision with yourself.

## Step 2 — tunnel (MacBook, project root: ~/Downloads/open-agent-robot)
```bash
cd ~/Downloads/open-agent-robot
ssh -F infra/ssh_config -N -f robot        # forwards 8080 (MJPEG) + 8765 (WS)
```
Check it's alive: `curl -s --max-time 3 http://localhost:8080/stream | head -c 20`
should print `--frame…`. (One tunnel only — a second one fails to bind and exits.)

## Step 3 — the daemon (eye/ear/mouth)
```bash
cd ~/Downloads/open-agent-robot
daemon/oar-daemon                 # full experience (mic + speaker + camera)
# daemon/oar-daemon --no-audio    # silent mode: camera only
# daemon/oar-daemon --check-perms # 2s self-test: expect mic=… camera=OK
```
Run it in the foreground — game events print there; Ctrl-C to stop.
First run from a new terminal app triggers macOS camera/mic permission
dialogs → Allow.

## Step 4 — the robot's face (browser)
```bash
open web/robot-view/index.html
```
Expect: green dot ● top-left, the Unitree G1 standing on a checkerboard,
HUD "就緒 — 說「石頭剪刀布」開始", then "daemon 已連線(眼耳嘴就緒)".

## Step 5 — play
Say to the laptop (normal pace): **「我們玩石頭剪刀布」** or
**"let's play rock paper scissors"**.
1. Robot replies with a voice invitation (~5 s).
2. Robot chants 「石頭、剪刀、布!」— **throw your hand on 「布」** and
   hold the shape ~1 s, palm facing the camera, filling ~¼ of the frame.
3. The robot's hand forms its own choice on the same beat (watch the browser).
4. It announces the result (「我贏了!」/「你贏了!」/「平手!」), and a few
   seconds later Claude Haiku adds a one-line banter.
5. Say the trigger again for the next round. Running score is in the HUD.

Silent mode (--no-audio): ask Claude to trigger a round remotely
(`ros2 topic pub --once /game/cmd std_msgs/String "data: start"` on EC2).

## Step 6 — skills (optional)
Say **「幫我紅隊一下這個計劃…」**, **「這個說法可靠嗎」**, or
**"is there a logical fallacy in this"**. A deterministic phrase table in
`skills/loader.py` recognises the utterance and attaches the critical-thinking
skill to that turn only. `run_ec2.sh logs agent` shows
`skills active: ['critical-thinking']` and a `read_skill_file …` line for each
deeper file the model pulls.

Nothing happens for a phrase that is not in the table — that is the design, not a
bug. `skills/critical-thinking/triggers.tsv` is the full list, and
[`skills/README.md`](../skills/README.md) explains how to add to it.

## Troubleshooting (fast checks)
| Symptom | Check / fix |
|---|---|
| SSH timeout | Your IP rotated → re-run `infra/provision.sh` |
| `bind [127.0.0.1]:8080: Address already in use` / `Could not request local forwarding` | Not a broken instance — your own Step 2 tunnel already holds those ports. Add `-o ClearAllForwardings=yes` to the command invocation |
| Cantonese reply is silent or the tts log shows `ConnectionRefusedError` | The `yue` node is down. `tts_node._sentences_yue` posts to `127.0.0.1:8901` with **no fallback**, so Cantonese fails hard rather than degrading to Mandarin. `run_ec2.sh start yue`; if `~/venv-yue` is missing, `setup-ec2.sh yue` was never run |
| `body` node dies at startup on `--xml … scene_with_hands.xml` | The G1 model was never fetched → `body/fetch_assets.sh` (run by `setup-ec2.sh stack`) |
| Browser dot red / no robot | Tunnel dead → re-run Step 2 |
| Robot doesn't hear you | Daemon terminal shows `[audio] … up`? ASR heard-log: `ssh -F infra/ssh_config robot 'grep heard: /tmp/oar/asr.log | tail -3'` |
| 「看到什麼」answers empty | detect node down? `run_ec2.sh status`, then `run_ec2.sh logs detect`. To confirm the topic is live, subscribe from a short python script inside `~/venv` rather than with `ros2 topic echo` — the CLI can false-negative from a stale discovery cache, so an empty echo is not evidence of an empty topic |
| Game won't stop | say 停/stop — hard reflex in game_node, 15 s cooldown; if stuck: `ros2 topic pub --once /game/cmd std_msgs/String "data: stop"` |
| 「沒看到手」 every round | Camera fps: daemon terminal `[cam] read pkt N` should tick ~every 2 s (60 pkts); hand live view: `ros2 topic echo /hand_pose` on EC2 |
| Crackling speech | You're on a pre-10:13 binary → `cd daemon && make`, restart daemon |
| Game restarts itself | Pre-fix asr/game nodes → `run_ec2.sh stop && start-all` |

## Shutdown (stop paying $0.736/hr)
```bash
# Mac: Ctrl-C the daemon, and Ctrl-C the tunnel in its own terminal.
#   Do NOT reach for `pkill -f "ssh -F infra/ssh_config"`: a pattern kill can match
#   any shell whose command line contains the string, including the one running it.
#   Same rule as run_ec2.sh — kill a pid you own, never a pattern. If you need the
#   tunnel scriptable, start it with a control socket and close that:
#     ssh -F infra/ssh_config -M -S /tmp/oar-tunnel.sock -N -f robot
#     ssh -S /tmp/oar-tunnel.sock -O exit robot
# EC2 (state survives a stop; nodes need start-all after next boot):
source infra/local.env
aws ec2 stop-instances --instance-ids "$OAR_INSTANCE_ID" --region "$OAR_REGION"
```
