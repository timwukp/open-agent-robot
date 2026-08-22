# Round 6 — Sim-to-Real Joint Mapping: Can the MuJoCo Puppet Drive a Physical Humanoid?

**Date:** 2026-08-22 · **Question that started it:** are there open-source ROS modules for human-like multi-joint ("multi-muscle") motion control that would let the virtual robot we run on EC2 map **one-to-one** onto a physical robot bought later — so that the motion seen in the 3D render *is* the motion the hardware performs?

**Method:** one deep-research round (5 search angles → 15 fetched sources → 3-vote adversarial verification) produced 25 claims, of which the synthesis stage silently received only 3 of 7 claim families — see §0.1. A second, targeted round put **2 adversarial verifiers on each of the 7 families** (one instructed to find primary sources, one instructed to *refute*, defaulting to CORRECTED or REFUTED unless a primary source pins the claim exactly): 14 verifiers, **88 verdicts**, of which 3 REFUTED and the large majority CONFIRMED-or-CORRECTED-with-a-sharper-number. Every claim below carries its evidence class. Repo/commit SHAs and licences re-checked against primary sources on 2026-08-22.

---

## 0. 決策增量(只列會改代碼的)

| # | 我們原本的立場 | 這一輪的證據 | 現在怎麼做 |
|---|---|---|---|
| 1 | 「換掉 MuJoCo 木偶 = 替換一個節點」(README 現在的說法) | **半對,層次錯了。** 真正的接縫是 `hardware_interface::SystemInterface`,而 sim/real 的抽象**不是節點,是 DDS topic 契約**:`unitree_mujoco` 發 `rt/lowstate`、收 `rt/lowcmd`,與真機 **byte-identical** | 一個 SystemInterface 服務兩邊,sim↔real 收斂成**兩個 xacro `hardware_parameters`**(`domain`、`network_interface`)。見 §1 |
| 2 | `body_node` 發 `/joint_states` 是「對外的關節介面」 | **方向反了。** `/joint_states` 按 ROS 慣例是**驅動器的回饋** topic;上真機會與 G1 driver **撞 topic** | 命令改發 `/joint_commands`(或 ros2_control 的命令介面);`/joint_states` 留給硬體/sim 後端。見 §6 |
| 3 | 關節名是穩定的交換貨幣 | **線上身分是 `MotorCmd_[35]` 的位置整數索引**,不是名字。但 `g1_29dof.urdf` 以 SDK index 順序列出 29 個 revolute joint(NVIDIA 直接叫它 `kG1JointNames`)→ **名字↔索引雙射存在** | 名字**可以**當交換貨幣,但必須在邊界解析一次並斷言長度。**slot 29 不是備用槽**,它是 `arm_sdk` 的混合權重通道 |
| 4 | 「控制迴路要 500 Hz」 | **500 Hz 是某個範例的速率,不是協議速率**:同一個 `rt/lowcmd` 被 Unitree 自己的 RL deploy 以 50 Hz 驅動、被 NVIDIA 以 200 Hz 驅動 | 慣例:**50 Hz 策略疊 500 Hz 物理**。PD 迴路在馬達上閉合,所以發得慢是安全的,發得**抖**才危險 → 排程抖動是驗收指標,速率不是 |
| 5 | 手是 LowCmd 的額外槽位 | **手是第二個控制權威。** `HandCmd_.motor_cmd_` 是無界 vector,必須 `resize(7)`;state topic 名字在 Unitree 自己的程式碼裡就不一致 | 手走獨立的 publisher/subscriber 對,topic 名做成參數。見 §2 |
| 6 | URDF/xacro 當唯一真相來源(第二輪曾判為不可能) | **翻轉:作為 pipeline 成立,作為 format 不成立。** `mujoco_ros2_control` 的 `<mujoco_inputs>` / `<raw_inputs>` / `modify_element` 讓 xacro 攜帶 MJCF-only 內容 | xacro 是來源,**MJCF 變成重生成 + diff 的 build artifact**。代價見 §4(`armature=0`、compiler 預設、要 fork `ros2_control_node`) |
| 7 | 用 Menagerie 的 `g1.xml` 當身體模型 | `g1.xml` 是**視覺化資產**:29 軸共用**一個** `kp=500`,README 自陳 "needs tuning"。真正上過真機的是 **MuJoCo Playground 自己 vendored 的 `g1_mjx_feetonly.xml`**,與 Menagerie 的 `g1_mjx.xml` 去空白後差 **359 行** | vendor Playground 的 feet-only 檔案,**pin SHA**,mesh 複製進我們自己的 `share/`。見 §3 |
| 8 | 「多肌肉」(muscle/tendon-driven)是可考慮的採購方向 | **關閉。** 2026-08-22 為止,零個「可購買 + 有開放控制介面」的肌肉驅動人形。而且唯一嘗試商業化的團隊**自己放棄了這個抽象** | 記錄為**否定結果**,不再重新辯論。柔順性改在軟體買:關節空間力矩控制。見 §5 |
| 9 | 安全層是上真機之後的增強 | **是前置條件。** 最佳規劃模型 0.4% 產不出有效計劃,卻在 **28.3%** 的任務上產出**危險**計劃 —— 普通的成功率監控對主要失效模式完全瞎眼 | 確定性危險條件檢查器,**擁有命令 topic 的最終發布權**。見 §7 |
| 10 | 「換更大的本機 guard 模型會更安全」 | **不會。** 3B→671B(200 倍)安全意識只從 37.6% → 56.9%;log-linear 外推說要 **~200,000 T 參數**才追上頂級專有模型 | 獨立佐證第五輪的結論(本機 guard 敗在**語言**而非顯存)。閘門必須是**確定性的**,不是模型 |
| 11 | 「先在 MuJoCo 調好再上真機」 | **引擎本身就是 domain gap**:僅 IsaacSim 訓練的策略在自己引擎 0.957,在未見過的 MuJoCo **0.036** | 上硬體前**強制第二引擎評測**;晉級標準是跨引擎成功率,不是引擎內獎勵。見 §3.3 |

### 0.1 這一輪的方法缺陷(記錄下來,因為它幾乎吃掉整輪)

第一次驗證回合的 15 個 agent **全部成功、零錯誤**,但綜合階段的 prompt 結尾是 `${JSON.stringify(usable, null, 1).slice(0, 120000)}` —— 我自己截斷了 payload,綜合者只收到 7 個家族中的 3 個,並且誠實地報告「families 4–7 not delivered」。修復方式是從 `journal.jsonl` 手動回收全部判決(14 個驗證員結果、88 條判決)。

教訓有二,都已寫進 memory:**綜合 prompt 對輸入切片會靜默丟掉整個驗證家族**;而 **journal 是可恢復的 ground truth** —— 在斷定 workflow「回傳空的」之前先讀它。回收過程本身還踩了兩次坑:驗證員把自由文字的 `family` 欄位填成長描述(其中一個含 `/`,當檔名用時丟 `FileNotFoundError`),而關鍵詞路由讓 "jazzy" 在 `"URDF/xacro vs MJCF … ROS 2 Jazzy stack"` 裡先命中,把 urdf 家族的驗證員錯分給 jazzy。修法是把 pattern 依特異性排序(dex3, urdf, muscle, safety, fidelity, lowlevel, jazzy)。

---

## 1. 接縫在哪一層:答案是 DDS topic 契約

這是整輪最重要的翻轉。問題不是「有沒有開源模組」,而是**我們的接縫要移到哪一層**才能讓那些模組接得上。

`unitree_mujoco` 並不是一個「模擬器插件」;它是一個**發布與真機相同 DDS topic 的行程**。它發 `rt/lowstate`、收 `rt/lowcmd`,IDL 與位元組佈局與實體 G1 完全一致。所以:

- **一個** `hardware_interface::SystemInterface` 同時服務 sim 與 real
- sim↔real 的差異收斂成兩個 xacro `hardware_parameters`

```xml
<ros2_control name="G1" type="system">
  <hardware>
    <plugin>oar_g1/G1SystemInterface</plugin>
    <param name="domain">1</param>              <!-- sim: 1 · real: 0 -->
    <param name="network_interface">lo</param>  <!-- sim: lo · real: eth0 -->
  </hardware>
  <!-- 29 joints, name-based, resolved to SDK index once at load -->
</ros2_control>
```

「我看到的虛擬機器人的運動可以映射到物理機器人身上」在這個架構下是**字面成立**的:兩邊收到的是同一串位元組。但一一對應有五個會斷的地方,列在 §2。

### 1.1 Jazzy 可用,但不要走 Unitree 自己的 ROS 2 層

**`unitree_ros2` 整個 tree 加 commit history 裡有零個 "jazzy"。**(source-code,已驗證)可行的模式是在 SystemInterface 內部**直接連結 `unitree_sdk2`**:

- 授權 **BSD-3**(無紅線)
- 但它是**預編譯靜態 blob**,gcc 9.4 / Ubuntu 20.04 時代 → 與 Jazzy(24.04)的 ABI 相容性要實測
- **pin tag `2.0.2`**

| 候選 | 判斷 | 理由 |
|---|---|---|
| NVIDIA `unitree_g1_ros2_control` | **首選** | name-based 映射、`with_hands`(29 + 2×7 Dex3)、暴露 kp/kd、Jazzy-aware |
| `mqcmd196/g1_ros` | **次選** | 真的有 Jazzy CI(amd64 + arm64);遇到未知關節名**硬失敗**(正確行為) |
| `grove-g1` 的 `G1Dex3System` | **可抄** | 社群的 Dex3-1 ros2_control 支援**已經存在**,不必從零寫 |
| `BrennoDom/g1_hardware` | ✗ | 停滯 14 個月;硬編 NIC / domain / gains;rate limiter 是死碼;17 項**位置**表,越界丟 `out_of_range` |
| `quadruped_ros2_control` | **是模式,不是身體** | 硬編 12 關節 + `unitree_go` IDL → 無法對 `unitree_hg` 的 G1;`main` 凍結在 2025-06-24;**OCS2 路徑在 Apache-2.0 標籤下拉 LGPL-2.1 的 qpOASES**(授權旗標) |

`quadruped_ros2_control` 唯一可重用的是**介面契約**:命令 {position, velocity, effort, kp, kd},狀態 {position, velocity, effort, imu, foot force}。這份契約值得直接採用,因為它把 kp/kd 當成**命令介面**而不是參數 —— 這正是 sim 與真機都需要的。

---

## 2. 一一對應會斷的五個地方

### 2.1 關節身分:名字 vs 位置索引

線上身分是 `MotorCmd_[35]` 的**位置整數索引**,不是名字。這一輪原本要判定「必須用位置表」,但被**駁回**了:`g1_29dof.urdf` 以 SDK index 順序枚舉它的 29 個 revolute joint(NVIDIA 直接把這個順序命名為 `kG1JointNames`),所以**名字↔索引的穩定雙射存在**。

- ✅ 名字**可以**是交換貨幣,在邊界解析一次
- ⚠️ 陣列是 **35 槽**,不是 29
- 🚩 **slot 29 不是備用槽** —— 它是 `arm_sdk` 的混合權重通道。當額外馬達用會造成無法解釋的行為

### 2.2 兩條真機路徑互斥,前置條件相反

| | `rt/lowcmd` | `rt/arm_sdk` |
|---|---|---|
| 覆蓋 | 全 29 軸 | 僅手臂 + 腰 |
| motion service | 必須**釋放** | 必須**執行中** |
| 額外前置 | 物理 debug 模式 | — |
| 必填欄位 | `mode()=1`、`mode_pr`、`mode_machine`、**CRC32** | 不設 motor mode、**不設 CRC** |
| 速率 | 50–500 Hz | 50 Hz |
| 權重 | — | ramp 在 `motor_cmd[29].q` |

前置條件**完全相反**,所以這必須是兩個明確、互斥的模式,絕不能自動切換。

### 2.3 速率不是協議的一部分

同一個 `rt/lowcmd` 被 Unitree 自己的 RL deploy 以 **50 Hz** 驅動、被 NVIDIA 以 **200 Hz** 驅動。慣例是 **50 Hz 策略疊 500 Hz 物理**。因為 PD 迴路在**馬達上**閉合,發得慢是安全的 —— 但發得**抖**不是。所以驗收指標應該是**排程抖動的上界**,不是名義頻率。

### 2.4 手是第二個控制權威

`HandCmd_.motor_cmd_` 是**無界 vector**,必須 `resize(7)` 才能索引。狀態 topic 的名字在 Unitree 自己的程式碼裡就不一致 → **做成參數**,不要硬編。

### 2.5 硬體限位會在上游靜靜改變

Menagerie 在 **2025-08-07** 修改了 G1 的右髖關節限位,理由寫的是 "to match the real robot limits"。**不 pin SHA 等於接受未版控的物理變更。**

---

## 3. MuJoCo 資產:我們現在用錯了檔案

### 3.1 三個檔案,只有一個有真機血統

| | Menagerie `g1.xml` | Menagerie `g1_mjx.xml` | **Playground `g1_mjx_feetonly.xml`** |
|---|---|---|---|
| PD 增益 | **kp=500,29 軸統一** | kp 75 / 20 / 2 | kp 75 / 20 / 2 |
| solver | 預設 | iterations=5, ls=8 | **iterations=3, ls=5** |
| timestep / 積分器 | 預設 | .004 / implicitfast | **.002 / Euler** |
| 碰撞幾何數 | 視覺用 | 20 | **37** |
| 上過真機? | 否 | **否** | **是** |

`g1.xml` 是 Unitree 已有的 MJCF(`g1_29dof_rev_1_0.xml`)的**手工衍生**:上游是 29 個 torque `<motor>`,Menagerie 換成 29 個 `<position>` actuator,全部繼承 `g1.xml` 第 10 行的**同一個** default `<position kp="500" dampratio="1" inheritrange="1"/>`。髖、膝、踝、腰、肩、腕共用一個增益,是「未調校」的力學簽名 —— README 自己寫的就是 "Added joint position actuators (**needs tuning**)"。

**Menagerie README 的那句話會誤導人。** 它寫 "The above model was successfully transfered to hardware in MuJoCo Playground",但真正在 G1 上跑過的是 Playground 自己 vendored 的檔案,不是 Menagerie 的:兩份去空白後差 **359 行**;而且 Menagerie 的 MJX 版本是 **2025-05-30** 才加的,比 Playground 論文(**2025-02-12**)晚約 4 個月。另外「MJX 模型」是**兩個檔案**的說法 —— contact pairs 與 `home`/`knees_bent` 兩個 keyframe 都在 `scene_mjx.xml`,`g1_mjx.xml` 裡有 0 個 `<pair>` 和 0 個 `<key>`。

**打包陷阱:** Playground 的 XML 透過固定相對路徑 `../../../../../mujoco_menagerie` 依賴 Menagerie 的 `.STL`。這對 colcon/ament 安裝是脆弱假設 → **把 mesh 複製進我們自己的 `share/`**。

### 3.2 真機證據只有一項任務,而且是定性的

Playground 在 G1 上部署的**只有搖桿基座速度追蹤**,室內平地,報告是 "stable walking and turning on standard indoor floors" —— **沒有成功率、沒有距離、沒有擾動幅度、沒有追蹤誤差**。任何超出「速度指令行走」的承諾(全身操作、粗糙地形)都是無證據領土,必須自帶硬體里程碑。

### 3.3 引擎本身就是一個 domain gap

PolySim(arXiv:2510.01708v3)Table I,MuJoCo 欄:

| 訓練引擎 | 在自己的引擎 | 在**未見過**的 MuJoCo |
|---|---|---|
| 僅 IsaacSim | 0.957 | **0.036** |
| IsaacSim + IsaacGym + Genesis | 1.000 / 1.000 / **0.929** | **0.564** |
| 僅參數級 domain randomization | — | 0.100 |

三個必須跟著數字一起引用的限定:

1. **這是 sim-to-sim,不是 sim-to-real。** 任務集是 ASAP 資料集裡 14 個難度不一的動作模仿片段,而且是在**刻意關閉**參數級 DR 的條件下零樣本轉移到 MuJoCo。
2. **"Succ" 的定義很寬鬆**:「平均身體位置誤差全程不超過 0.5 m」。所以 0.564 **不是安全論證**。論文也從未說明每個動作跑幾次 rollout(0.036 不是 1/14 的倍數)。
3. **摘要的 "improves by 52.8%" 是 52.8 個百分點**(0.564 − 0.036),相對變化約 15×。

工程結論:**MuJoCo-only 的 pipeline 在上硬體前必須先在第二個引擎上驗證**,而參數隨機化補不了這個洞(只到 0.100)。晉級標準要是**跨引擎成功率**,不是引擎內獎勵。

PolySim 現在可直接參考:**MIT 授權、程式碼已公開**(2025-10-14 起),而且 `sim2real/state_publisher.py` 裡**已經有一個 rclpy 節點**把 `unitree_sdk2py` 的 DDS `LowState_` 橋到 ROS 2 topic —— 抄之前要知道它是**散裝腳本不是 ament package**(repo 根目錄與 `sim2real/` 都沒有 `package.xml`),而且從未指名 ROS distro。

### 3.4 沒有任何 Menagerie 模型有保真度評級 —— 但阻塞已經解除

A+/A/B/C 的評級表只存在於 root README;**70 個模型目錄一個都沒有公布評級**。但 README 給的理由已經過期:它說要等「system identification toolbox」,而**那個工具箱已經出貨**(`google-deepmind/mujoco` 的 `python/mujoco/sysid`),Menagerie 自己也用過了 —— `trossen_wxai` 的 changelog 記著用**4 條實機軌跡**跑完 sysid。

所以:**G1 的 sysid 是我們自己要排的工作項**(actuator gain、joint damping / armature / friction),不是等上游。相關陷阱:**`armature` 預設 0**(反射慣量 = 轉子慣量 × 齒比²),而 URDF importer 從不設它 → 在 sim 調好的增益不會轉移。驗收門檻應該是**實測 vs sim 的階躍響應**。

### 3.5 授權:`unitree_g1` 是 BSD-3-Clause,不是 Apache-2.0

70 個模型目錄的逐目錄普查:

| 授權 | 目錄數 |
|---|---|
| Apache-2.0 | 29 |
| BSD-3-Clause 系(含一個 BSD-3-Clause-**Clear**) | 23 |
| MIT | 13 |
| BSD-2-Clause 系 | 3 |

Apache-2.0 **只**適用於非模型內容。`unitree_g1` 與 `unitree_h1` 都是 BSD-3-Clause,版權方是 Unitree —— 再散布時要**重現 Unitree 的通知**並帶 no-endorsement 條款。「Menagerie 是 Apache-2.0,除了 G1/H1」這個說法對 **70 個目錄裡的 39 個是錯的**。

✅ **整個 Menagerie 沒有 AGPL/GPL。** 這裡沒有紅線(對比我們的 YOLO26/Ultralytics AGPL 例外)。

---

## 4. xacro 當唯一真相來源:成立,但要付四筆費用

第二輪判「URDF 不能是唯一真相來源」被**部分翻轉**:作為 **pipeline** 成立,作為 **format** 不成立。`mujoco_ros2_control` 提供 `<mujoco_inputs>` / `<raw_inputs>` / `modify_element`,讓 xacro **攜帶 MJCF-only 的內容**(contact pairs、solver 設定、armature、keyframe)。於是 MJCF 降級成**重生成的 build artifact**,可以進 CI 做 diff gate。

要付的費用,一筆都不要漏:

1. **`armature` 預設 0**,importer 從不設 → 必須顯式寫進 `<mujoco_inputs>`
2. **compiler 預設隨格式而異**:`fusestatic`、`discardvisual`,加上 Jazzy 所 pin 的 MuJoCo 版本還有 `strippath`
3. **URDF 沒有 schema 驗證** → 拼錯的屬性靜默忽略,所以 diff gate 是唯一防線
4. **要 fork `ros2_control_node`**(上游 issue #72 未關)

---

## 5. 「多肌肉」的直接答案:不行,而且理由是結構性的

兩組驗證員獨立查證,結論一致:**2026-08-22 為止,零個「可購買 + 有開放控制介面」的肌肉/腱驅動人形機器人。** 空間乾淨地分成兩個不相交的集合,而且沒有一邊有交集成員。

### 5.1 開放但買不到

| 平台 | 肌肉數 | 狀態 |
|---|---|---|
| Kengoro(UTokyo JSK) | **124** | GitHub 搜 `kengoro` 只有無關命中;jsk-ros-pkg 的 35 個 repo 一個都沒有;**三篇論文對 ROS / middleware 的命中數是 0** |
| Musashi(JSK + Toyota) | **74** | 同上;2026 年的綜述自己寫 "operating a full-body musculoskeletal humanoid requires a significant amount of engineering" |
| Roboy(TUM) | — | 唯一有**真正公開 stack** 的(`roboy-ros-control`、`roboy_plexus`、`roboy_controller`),但 README pin 的是 **ROS Kinetic**(ROS 1,2021 EOL),控制 repo 最後有意義的提交在 2018–2021。**從未出售** |

### 5.2 買得到但封閉

**Clone Alpha** 是 279 台的限量預約。實測(2026-08-22 抓 5 個頁面的原始 HTML):

- `/pre-order` 至今寫著 "Pre-orders available in **2025**",而且頁面裡**連一個 `<form>` 或 `<input>` 都沒有** —— 現在連預約都做不到
- **無價格、無幣別、無押金、無交期**
- **零個** SDK / API / ROS / developer 字串
- `/research` 頁面是空的:"We are currently researching, page will be available soon."
- 兩個廣告的介面是語音與 "Telekinesis" 模仿平台 —— **兩者都到不了關節或肌肉**

規格必須當**未經驗證的需求**看,不是量測值:`<50 ms` / `>30% 收縮` / `≥1 kg per 3 g` 原文是用 "**should**" 寫的規範性目標("a suitable synthetic muscle fiber **should** respond in less than 50 ms…"),接著斷言只有 Myofiber 達成,**沒有數據、沒有引用**。

### 5.3 決定性的一筆:唯一嘗試商業化的團隊放棄了這個抽象

**Devanthro**(Roboy 的商業分支)根本不賣硬體 —— Robody 是以遠端操作**服務**部署的。而它公布的規格是:

> Actuation: **QDDs + Dynamixel integrated actuators** · Middleware: **ROS2** · End-effector: 2-finger gripper · 差速底盤 · Jetson Orin

齒輪致動器、關節空間、ROS 2。這是一個支持**我們現有架構**的自然實驗。

### 5.4 肌肉結構性地破壞「可索引性」——也就是一一對應本身

| 平台 | 致動器 | 關節 / 自由度 |
|---|---|---|
| MusashiLarm(單臂) | **18** 條肌肉(含跨關節肌) | **3** 個關節 |
| MyoSuite `myoHand` | 39 條肌腱單元 | 23 個關節 |
| MyoSuite `myoLeg` | 80 條肌腱單元 | 20 自由度 |

而且**關節空間的可觀測性本身會消失**:

- Kengoro 為了模仿人的球窩關節與肩胛,**裝不了關節角感測器** → 必須用 EKF 從肌肉長度變化推,再用 AR marker 視覺做逆運動學修正。綜述自己承認「**視覺被遮擋時關節角就估不出來**」、估計值「不高度可靠」。
- Musashi 反過來:為了能直接量關節角,在每個關節模組塞**小電位計**(2026 綜述籠統稱之為 "encoders";硬體論文寫的是 potentiometer + IMU),代價是**放棄人型球窩關節、接受軸驅動的奇異點**。也就是說,關節空間可觀測性是用「放棄肌肉骨骼理想」換來的。
- 力矩→張力的 QP 存在(`min fᵀW₁f s.t. τ = −Gᵀ(θ)f`),**但不是實際運作的抽象**:論文明說 "the robots handled in this study basically operate under muscle length control, so this method is not used"。真正的介面是**肌肉長度空間**。

**追蹤精度也別高估**:Musashi 在實機上做線上身體圖式學習後的最佳結果,是命令 90°(−30°→60°)的把手旋轉**只轉到約 65°(72%)**,作者承認「難以完整執行預期動作」。齒輪 + 每關節編碼器的機器人遠好於此 —— **要補的是柔順性,不是追蹤精度**。

### 5.5 MyoSuite 不是橋

Apache-2.0、基於 MuJoCo(`mujoco>=3.6,<3.7`),但 **751 個追蹤檔案裡 0 個 `ros2`、0 個 `rclpy` 命中**,也沒有任何商用人形(0 個 `unitree` / `dex3` / `g1` 命中)。它是人體生物力學的 RL 環境套件。一個修正:它**不是**純人體解剖 —— 它以 git submodule 收了 JHU/APL 的 Modular Prosthetic Limb(一支**機械**手臂),而那隻義手的動作空間**是關節空間**的。結論:適合當過致動動作空間的參考,**不是控制棧**。

### 5.6 你想要的柔順性要在軟體裡買

方向是**關節空間力矩控制**,而且在諧波減速人形上**不需要關節力矩感測器** —— ergoCub 已經證明,程式碼在 `github.com/ami-iit`:

| 方法 | 力矩追蹤 RMSE |
|---|---|
| RNEA,無補償 | 4.37 N·m |
| UKF,無摩擦補償 | 2.03 N·m |
| **UKF + PINN 摩擦模型** | **1.08 N·m**(MAE 0.70) |

18 個關節 @ 100 Hz。四個必須寫進計劃的細節:

1. **摩擦模型是每關節識別的**(左踝滾 Coulomb 摩擦 `kc` = **1.2 N·m** vs 左膝 **8.1 N·m**,差 6.75×),每個訓練約 **1.5 小時**
2. **但跨同族機器人可遷移**:同一個 PINN 在兩台磨損與潤滑狀態不同的 ergoCub 上「without re-tuning」→ 是每個**硬體家族**一次的成本,不是每台。對比古典模型控制器「typically requires re-identifying system parameters for each hardware instance」
3. **必須涵蓋靜摩擦**("motor actuation without joint movement")→ 激勵軌跡要有**近零速停留段**,不能只有快速掃掠;而且模型輸入要餵**位置誤差** `Δθ = rs − θ`,才能區分「被摩擦卡住」與「沒下命令」
4. **範圍限定**:驗證在軀幹 / 腿 / 手臂關節,**不含手**

延伸推論:sim-to-real 的落差會被這個摩擦項主導 → **在 MuJoCo 模型裡放一個摩擦 / 力矩偏置通道**,不要試圖用 PD 增益去補。

參照物:ergoCub 本身是 1.50 m / 56.70 kg(衍生自 iCub3 的 1.25 m / 52 kg),額定 payload 6 kg —— 「full-size」是寬鬆的說法。它是 INAIL 資助的 IIT 研究專案,**沒有購買通道**(只有 Contact Us)。

---

## 6. 我們自己的接縫缺陷(量化)

| 缺陷 | 現狀(已驗證於 `body/body_node.py`) | 應該是 | 嚴重性 |
|---|---|---|---|
| 抽象層太高 | `/body/gesture`(`std_msgs/String`)進 | `JointTrajectory` / ros2_control 命令介面 | 中 |
| **方向反了** | `body_node` **發** `/joint_states` | `/joint_states` 按慣例是**驅動器的回饋** topic → 上真機**會與 G1 driver 撞 topic** | **高** |
| 速率低 4–43× | 在 render tick 發布(`fps=20` 名義,JPEG 負載下**實測 ~11.5**) | 50–500 Hz,抖動有界 | 高 |
| 零動力學 | 只調 `mj_forward` | `mj_step` —— 現在的木偶**沒有任何預測力** | 中 |

第二條是最嚴重的:它不是效能問題,是**架構會在接真機那天壞掉**。而且 README 目前的說法(「body 層對外只發 `/joint_states`,所以換硬體是替換一個節點」)把這個缺陷寫成了賣點 —— 需要一併修正。

`body/gestures.py` 的 13 個邏輯關節名對映到 MJCF 名字這件事本身是對的,而且驗證確認了兩件我們已經做對的事:MJCF 名字**是全前綴的**(`right_hand_thumb_0_joint` 等,我們的檔案用的正是這個),以及 Dex3-1 是**三指 7 自由度**(拇指 3 / 食指 2 / 中指 2)。順帶修正:線上順序是 **thumb → middle → index**,不是 index 優先。

---

## 7. 安全層是前置條件,而且不能靠模型

### 7.1 規劃能力與安全完全脫鉤

DESPITE(**D**eterministic **E**valuation of **S**afe **P**lanning **I**n embodied **T**ask **E**xecution)是 12,279 個 PDDL 規劃任務的基準,涵蓋物理危害(機械 / 熱 / 化學 / 電)與規範性危害(隱私、信任、社會規範),**驗證是完全確定性的**(計劃是否觸發危險條件的邏輯謂詞),明確**拒絕 LLM-as-judge**,理由是裁決跨次不穩定。

在其中 **1,044 題的困難分割**上,23 個模型:

| 量 | 最佳模型的值 |
|---|---|
| 產不出有效計劃(不可行率) | **0.4%** |
| 產出**危險**計劃 | **28.3%**(摘要)/ 28.7%(正文 —— 論文自己不一致) |

**這意味著普通的成功率監控對主要失效模式完全瞎眼:** 計劃乾淨執行,而且不安全。可行性檢查、重試、「任務完成了嗎」的遙測**全部顯示綠燈**。閘門必須檢查**發出的動作序列的危險條件**,不是計劃有效性。

### 7.2 不要指望規模或推理修掉它

| | 值 |
|---|---|
| 18 個開源模型的參數跨度 | 3B → 671B(200 倍) |
| 可行性跨度 | 0.4% → 99.3% |
| **安全意識(SI)跨度** | **37.6% → 56.9%** |
| 最佳專有推理模型 | GPT-5 high **81.4%**、Gemini-2.5-Pro 73.7%、Gemini-3-Pro-Preview 71.6% |
| 非推理專有 / 開源推理模型 | 全部 **< 57%** |
| log-linear 外推:開源要達到頂級 SI 所需參數 | **~200,000 T** |

即使最好的模型也**漏掉約 1/5 的危險案例**。這獨立佐證了第五輪的結論:本機 guard 模型敗在**語言覆蓋**而不是顯存 —— 換更大的 checkpoint 買到的是規劃,不是安全。

**thinking mode 不能假設有用**:DeepSeek **+37.4** 個安全分,但 Qwen3 兩個變體 **−4.9** 和 **−12.5**。本機 guard 的推理開關必須**逐家族 A/B 實測**。

論文自己的分解是**乘法**的:`S ≈ β₀ + β₁(F × SI)`,R² = **0.99**,斜率 1.035,截距 −0.040。所以**提高 SI 項的確定性檢查器,比任何規劃器改良都值錢**。

### 7.3 三個直接落到我們設計的推論

1. **不要用 LLM 當 runtime 閘門。** DESPITE 拒絕 LLM-as-judge 的理由(裁決跨次不穩定)在機器人上等於**同一個危險指令會被非確定性地放行**。要照 DESPITE 的方式做:對符號化世界狀態做確定性謂詞評估。
2. **規範性危害是較難的一半,而且 MuJoCo 場景表示不了。** 物理危害主導**可行性**難度(70.6–88.2%),規範性危害(隱私、信任、社會規範)主導**安全**難度。純 sim 的安全閘門會系統性地漏掉正好是第五輪在 EU AI Act 下標記的那一類。
3. **重試 / 計劃修補迴路是攻擊面。** 注入 2–64 個與危險無關的冗餘動作 → 可行性 **−21.0%**、SI **−17.3%**,複合 **−37.9%**。任何附加動作的修補都必須**端到端**重新驗證,不能增量驗證。

**DESPITE 自己的限制是我們必須補的洞:** 它「isolates planning from visual perception」,完全沒碰感知,也沒有機器人。真機上由感知導出的世界狀態帶著 PDDL 介面沒有的誤差 → 確定性檢查器需要**自己的信心閘門**,以及狀態估計不可信時的 **fail-safe-to-stop** 路徑。作者自稱這是「a lower bound on failure rates in deployed systems」。

### 7.4 摔倒保護:抄架構,不要抄數字

SafeFall 的摘要說在真 G1 上「reduced peak contact forces by 68.3%, peak joint torques by 78.4%, and eliminated 99.3% of collisions」。**正文與摘要矛盾。**

| 量 | 值 | 來源 |
|---|---|---|
| 峰值接觸力 4036 → 1279 N(**−68.3%**) | **模擬**,5,000+ 場景,基線 = 無保護 nominal policy | Table IV |
| 峰值關節力矩 613 → 132 N·m(**−78.4%**) | **模擬** | Table IV |
| 違規接觸 99% → 0.7%(**−99.3%**) | **模擬** | Table IV |
| **峰值 100 ms 衝量 367.1 → 286.1 N·s(−22.1%)** | **真機**,動作捕捉量測,基線 = damping mode | §IV-D |

換用現實基線(damping mode,也就是今天真機真正會退回的模式)重算,連 sim 的接觸力增益都從 68.3% 縮到 **59%**(3138 → 1279 N)。**不要把 68.3 / 78.4 / 99.3 放進任何設計文件當作實測硬體效能** —— 唯一有硬體根據的防摔數字是 **22.1% 衝量降低**。

架構本身值得照抄,而且規格很具體:

- 單層 **GRU**,**64** hidden units,二元輸出
- 輸入只有 **IMU + 關節編碼器**:骨盆 roll/pitch + 基座角速度 + 關節 q / q̇ —— 不依賴外部感知或狀態估計
- 推論 **< 0.5 ms**,執行於 **50 Hz**
- 正常運作時**零干擾**;判定摔倒不可避免時**完全接管** nominal 控制器
- 運作點:約 **410 ms** 前置時間 @ 約 **0.06%** 誤報率(0.06% → 0.16% 可換到明顯更多前置時間 → 這是應用可調的旋鈕)

三筆成本:

1. 緩解策略約 **280 GPU 小時**的稀疏獎勵訓練
2. **預測器每換一個 nominal 步態就要重訓**(緩解策略可以零微調遷移到未見過的風格化控制器,**預測器不行**)
3. 硬體證據只到「受控推撞下零誤報」;**真實場域擾動下的誤報率未被量測**,而這裡的誤報意味著**一次不必要的摔倒**

**ROS 2 上的架構含意:** 監督者必須**擁有關節命令 topic 上的最終發布者**(或它所控制的一個多路器),而不是和 nominal 控制器同優先級競爭發布。而且它得跑在硬實時的控制迴路上 —— 這屬於 C++ / ros2_control 硬體介面,**不是 LLM 旁邊的 Python 節點**。

順帶一個要記住的邊界:sim 的最佳情況下峰值關節力矩仍是額定的 **2.47×**,作者靠「瞬態容許 3–5×」的論證撐過去。

### 7.5 這一輪自身的 sim-to-real 校準教訓

同一篇論文裡 sim 68.3% 對真機 22.1% —— **而且是不同的物理量**。這是整條 MuJoCo-to-real pipeline 的校準教訓:把 MuJoCo 的損傷緩解數字當**上界**,並且在做 sim 研究**之前**就選定儀器,讓 sim 與真機量的是**同一個量**。

---

## 8. 建議的落地順序(每一步都可以停)

| 步 | 內容 | 為什麼是這個順序 | 成本 |
|---|---|---|---|
| **0** | 修 `body_node` 接縫方向:改發 `/joint_commands`;`/joint_states` 留給後端;加 `mj_step`;發布從 render tick 拆到獨立 50 Hz timer | 純本機、零外部依賴,而且是唯一「不做就會在接真機那天壞」的項目 | 小時級 |
| **1** | 引入 `ros2_control` + `G1SystemInterface` 骨架(sim 後端先用 `mujoco_ros2_control`) | 建立接縫;此後 sim/real 是兩個參數 | 天級 |
| **2** | vendor Playground 的 `g1_mjx_feetonly.xml` + pin SHA;mesh 複製進我們的 `share/` | 唯一有真機血統的資產;順手修掉相對路徑的打包假設 | 小時級 |
| **3** | xacro 升為唯一真相來源,MJCF 進 build + diff gate | 這才是「一一對應」在檔案層面的實現 | 天級 |
| **4** | 確定性危險條件檢查器,擁有命令 topic 的最終發布權 | 上真機的**前置條件**,不是後續增強(§7.1) | 天級 |
| **5** | G1 sysid(用已出貨的 `python/mujoco/sysid`);把 `armature` 與摩擦偏置寫進 xacro | 買真機之後、跑任何動力學策略之前 | 需要硬體 |
| **6** | 第二引擎評測 gate(§3.3) | 任何學習型策略上硬體之前 | 需要 GPU 預算 |

**明確不做的:** 任何肌肉 / 腱空間的致動路徑、訊息或策略動作空間(§5 —— 沒有硬體落地區,會永遠只存在於 sim);把 MyoSuite 放到部署路徑上(至多當生物力學沙盒,擺在 adapter 後面)。

---

## 9. 誠實的限度

- **`unitree_sdk2` 與 Jazzy 的 ABI 相容性沒有實測。** 它是 gcc 9.4 / 20.04 時代的預編譯靜態 blob,而 Jazzy 在 24.04。這是我們第一個要自己驗的事,而且是 §1.1 整條路徑的前提。
- **G1 上的一切都是文獻,我們沒有硬體。** 本文件沒有一個數字是我們自己在 G1 上量的。
- **兩篇安全論文都是未經同儕審查的 arXiv 預印本**(無 journal_ref),而 DESPITE 對自己的頭條數字都不一致(摘要 28.3% vs 正文 28.7%)。當方向性證據引用,不要當定值。
- **沒有任何被驗證的系統在「多模態具身事件流」上做過安全評測。** DESPITE 是純符號的,SafeFall 只管摔倒。中間那一大塊(感知誤差如何汙染確定性檢查器)是開放問題。
- **`mujoco_ros2_control` 需要 fork `ros2_control_node`**(上游 issue #72 未關)→ 這是持續的維護債,不是一次性成本。
