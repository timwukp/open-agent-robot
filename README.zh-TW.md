# open-agent-robot

一個開放、可改造的多模態人形機器人 agent。機器人的**大腦與身體**以 ROS 2 節點跑在雲端一台 GPU 機器上;你的**筆電是它的眼、耳、嘴**;你的**瀏覽器是它的臉**。所有流量走同一條 SSH 通道。參考用例是**語音觸發的石頭剪刀布**(出拳與節拍同步)—— 它把整個迴路都逼出來:聽、看、想、說、動,而且全部要卡在一個硬實時的節拍上。

程式碼是為了將來搬上真機而寫的,而第六輪研究釐清了那到底需要什麼。虛擬與實體之間的接縫是一份 **DDS topic 契約,不是換一個節點**:`unitree_mujoco` 發 `rt/lowstate`、收 `rt/lowcmd`,與真的 Unitree G1 **位元組完全一致**,所以同一個 `hardware_interface::SystemInterface` 服務兩邊,切換收斂成兩個 xacro 參數。身體層目前還**不在**那個接縫上 —— 它在 render tick 上發 `/joint_states`,而這既是錯的**方向**(`/joint_states` 按 ROS 慣例是驅動器的回饋 topic,上真機會與 G1 driver 撞 topic),也慢了 4–43 倍。見 [`docs/research-round6-simreal-joint-mapping-2026-08-22.md`](docs/research-round6-simreal-joint-mapping-2026-08-22.md)。

> 現狀:端到端可用的 demo(普通話 / 英文 / 粵語語音、開放詞彙場景描述、猜拳遊戲、講故事、可打斷的說話)。研究級,不是產品。設計脈絡見 `docs/`。

## 為什麼長這樣

三個約束決定了架構,值得先講,因為它們解釋了那些乍看奇怪的選擇。

**1. 遊戲節拍不能等大模型。** 實測從機器到 Bedrock 的首 token 延遲 p50 約 0.9 秒 —— 用來閒聊很好,用來喊「石頭、剪刀、布 —— 出!」是致命的。所以系統明確切開:**反射層(System 1)**掌管節拍、停止詞、搶話,**絕不**呼叫 LLM;**思考層(System 2)**負責對話、場景理解、手勢選擇。使用者喊「停」時,是 `game_node` 裡的確定性關鍵詞比對讓遊戲停下 —— 不是一個被禮貌請求的模型。

**2. 筆電硬體才是誠實的介面。** 一個聽不見真實房間的機器人不值得做。一支小小的 C daemon 直接掌管麥克風、喇叭、相機(音訊用 miniaudio,相機用 libavdevice),因為延遲與雜音真正住在音訊路徑裡。它把 20 ms 的 PCM 塊與 MJPEG 幀透過一條 WebSocket 送出,幀頭 9 bytes(`[1B 型別][8B 擷取時間戳 µs]`)。

**3. 說話必須可以被打斷。** TTS **逐句**合成、逐句下傳,所以就算是長故事,第一句也在約 0.6 秒內出聲;使用者開口壓過機器人時會觸發清環,聲音當場斷在句中。這個「殺聲開關」也正是將來要做的**非同步輸出安全檢查**能不加延遲的原因(見 `docs/design-critique-memory-safety-*.md`)。

## 架構

```
MacBook                                  │  EC2 g4dn.xlarge (T4) — 單一 SSH 通道
─────────────────────────────────────────┼──────────────────────────────────────────
daemon/  (C)                             │  gateway/  ws_gateway.py   (WS ⇄ ROS 2)
  麥克風 16 kHz ─┐                        │      │
  相機 ──────────┼─ WebSocket /stream ────┼──────┤  speech/   asr_node   (FunASR 流式)
  喇叭 ◀─────────┘                        │      ├─          tts_node   (Kokoro 中/英 +
                                         │      │                       CosyVoice 粵語)
web/robot-view/  (瀏覽器)                 │      ├─ perception/ hand_node   (MediaPipe)
  MJPEG 身體視圖 + 事件 HUD ◀─────────────┼──────┤             detect_node (YOLO26 TRT)
                                         │      ├─ agent/    agent_node (Bedrock Converse)
                                         │      ├─ game/     game_node  (反射層)
                                         │      └─ body/     body_node  (MuJoCo + MJPEG)
```

| 層 | 元件 | 是什麼 |
|---|---|---|
| 聽 | `speech/asr_node.py` | FunASR `paraformer-zh-streaming`,600 ms 塊,整句累積;半雙工閘讓機器人不會把自己的聲音聽寫進去 |
| 說 | `speech/tts_node.py` | Kokoro-82M(普通話+英文)逐句流式;偵測到粵語特徵字則路由到 CosyVoice 微服務(`speech/yue_server.py`) |
| 看 | `perception/detect_node.py` | YOLO26s 編成 TensorRT(T4 實測 161 FPS),限速到 5 fps,以 1 秒滾動窗聚合 |
| 手 | `perception/hand_node.py` | MediaPipe Tasks HandLandmarker;猜拳分類器**刻意忽略拇指**(每個人擺法不同) |
| 想 | `agent/agent_node.py` | Bedrock Converse、Claude Sonnet 5 global 推論設定檔;每輪附一張縮圖,支援開放詞彙的場景問答 |
| 反射 | `game/game_node.py` | 機器人主導口令,判定窗 `[T−0.1 s, T+0.6 s]`;停止詞在**任何**狀態都生效;每局冷卻 |
| 身體 | `body/body_node.py` | MuJoCo 運動學木偶(Dex3-1 三指手:拇指 3 自由度、食指 2、中指 2,單手 7 個),以 MJPEG 串到瀏覽器 |
| 傳輸 | `daemon/main.c`、`gateway/ws_gateway.py` | 單一 WebSocket、二進位分幀;音訊優先於視訊,兩者都有流控 |

## 快速開始

前置:一個開通 Bedrock 模型權限的 AWS 帳號、登入好的 AWS CLI、macOS + Homebrew,以及 `brew install ffmpeg libwebsockets`。

```bash
# 1. 開一台上鎖的 GPU 機器(只開 SSH、只放你的 IP、強制 IMDSv2、
#    加密根碟、實例角色只給 Bedrock invoke)。可重複執行。
infra/provision.sh            # 產生 infra/ssh_config 與 infra/local.env(皆已 git-ignore)

# 2. 在機器上安裝整套環境(驅動、ROS 2 Jazzy、模型)
infra/setup-ec2.sh            # 兩個階段,見腳本檔頭

# 3. 啟動機器人的大腦與身體
ssh -F infra/ssh_config robot 'open-agent-robot/infra/run_ec2.sh start-all'

# 4. 開通道、編譯並啟動 daemon、打開它的臉
ssh -F infra/ssh_config -N -f robot
cd daemon && make && ./oar-daemon
open web/robot-view/index.html
```

然後對筆電說「我們玩石頭剪刀布」或 "let's play rock paper scissors"。
完整流程與排錯:[`docs/DEMO.md`](docs/DEMO.md)。

## 目錄地圖

| 路徑 | 內容 |
|---|---|
| `daemon/` | macOS C daemon(音訊、相機、WebSocket 客戶端);`make` 會自動抓 `miniaudio.h` |
| `speech/`、`perception/`、`agent/`、`game/`、`body/`、`gateway/` | ROS 2 節點,一個節點一個關注點 |
| `web/robot-view/` | 單頁「臉」:MJPEG 身體視圖 + 事件 HUD |
| `infra/` | 開機供裝、環境安裝、行程監管(`run_ec2.sh` 用 pid 檔,**絕不**用 `pkill` 樣式) |
| `docs/` | 設計記錄:六輪研究、day-0 延遲量測、demo 手冊、記憶與安全的設計批判 |

## 設計記錄

`docs/` 是這個 repo 最有價值的部分。它保留的是研究與論證,不只是結論 —— 包括那些**反直覺**的發現:

- [`docs/day0-measurements.md`](docs/day0-measurements.md) —— 架構所依據的延遲數字
- [`docs/research-multimodal-robot-agent-stack-2026-08-20.md`](docs/research-multimodal-robot-agent-stack-2026-08-20.md) —— 技術棧普查(ASR/TTS/視覺/仿真),附授權
- [`docs/research-round3-feasibility-2026-08-20.md`](docs/research-round3-feasibility-2026-08-20.md) —— 可行性與硬體預算
- [`docs/research-round4-agent-brain-2026-08-21.md`](docs/research-round4-agent-brain-2026-08-21.md) —— 記憶 / 技能 / MCP-OAuth / 批判思維。重點:**沒有外部反饋的自我批判會實測降低準確率**,所以批判思維技能做成觸發式,而且只餵真實的外部證據。
- [`docs/design-critique-memory-safety-2026-08-22.md`](docs/design-critique-memory-safety-2026-08-22.md) —— 對我們自己記憶設計的紅隊審視、情節記憶 schema、遺忘政策,以及分層的道德邊界安全架構。核心設計律:**LLM 記憶系統的安全性來自限制 schema,不是來自過濾文字。**
- [`docs/research-round5-memory-safety-2026-08-22.md`](docs/research-round5-memory-safety-2026-08-22.md) —— 可行性一輪:用已驗證的數字回答上面那份批判留下的開放問題(鞏固成本、檢索延遲預算、「硬邊界必須在 LLM 之外」的已發表實證)。兩個決策被證據翻轉:本機 guard 模型是被**語言覆蓋**否決而不是顯存,salience 公式移除語音韻律項,因為從生物特徵推論情緒在 EU AI Act 下屬高風險。
- [`docs/research-round6-simreal-joint-mapping-2026-08-22.md`](docs/research-round6-simreal-joint-mapping-2026-08-22.md) —— MuJoCo 木偶到底能不能一一對應地驅動實體人形?7 個主張家族、88 條對抗式驗證判決。重點:**虛擬與實體的接縫是一份 DDS topic 契約,不是一個插件**;肌肉/腱驅動這條路是關的(零個可購買且有開放介面的機型,而唯一嘗試商業化的團隊最後出貨的是齒輪驅動的關節空間致動器);以及**規劃能力與安全完全脫鉤** —— 最佳規劃模型只有 0.4% 產不出計劃,卻在 28.3% 的任務上產出**危險**計劃,所以普通的成功率監控對主要失效模式完全瞎眼。

## 安全姿態

- 機器上只開 22 埠,且只放行操作者當前 IP,其餘全走通道;強制 IMDSv2、根碟加密、實例角色只有 `bedrock:InvokeModel`。
- 帳號相關的值(實例 id、公網 IP、金鑰路徑)由 `provision.sh` 產生到 git-ignore 的檔案裡;版本庫中不追蹤任何帳號專屬資訊。
- 原始麥克風音訊與相機幀默認**永不落盤**。每輪對話會送一張縮圖到 Bedrock —— 這是第三方傳輸,任何部署都必須向使用者明示。

## 授權

我們的程式碼是 MIT(見 [`LICENSE`](LICENSE))。

**但有一個依賴不是**:YOLO26 檢測器透過 Ultralytics 使用,而 Ultralytics 是 **AGPL-3.0**。本 repo 是研究 demo,且檢測節點是可選的(`infra/run_ec2.sh` 單獨啟動它)。若你要做任何商業用途,請取得 Ultralytics 授權,或把 `perception/detect_node.py` 換成寬鬆授權的檢測器 —— 該節點對外只發一個 `/detections` JSON 主題,可以孤立替換。

模型權重各有自己的授權(Kokoro、CosyVoice、FunASR、MediaPipe),再散布前請逐一確認。

English: [README.md](README.md)
