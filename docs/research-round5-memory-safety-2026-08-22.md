# Round 5 — Episodic Memory & Moral-Boundary Safety: Feasibility

**Date:** 2026-08-22 · **Method:** two independent deep-research workflows (105 + 107 agents; 240 claims extracted, 50 adversarially verified with 3-vote panels, 3 killed) plus one targeted gap-verification pass on the claims the first two rounds extracted but never voted on. Every license, price and specification re-checked against primary sources on 2026-08-22.

**為什麼要第五輪:** 第四輪選了 cognee + Graphiti,但用的理由是「聯想式召回更強」——`docs/design-critique-memory-safety-2026-08-22.md` §1 已指出那是**最弱的證據**。這一輪的任務不是再找新玩具,而是回答三個會改變代碼的問題:episodic 記憶到底存什麼、鞏固與遺忘的**單位成本**是多少、道德邊界的**確定性部分**有沒有已發表的實證。

---

## 0. 決策增量(只列會改代碼的)

| # | 第四輪 / 批判文件的立場 | 第五輪證據 | 現在怎麼做 |
|---|---|---|---|
| 1 | Episode 切分用確定性訊號(靜音、活動轉換、場景變化),因為 Bedrock 拿不到 logprobs | **成立且被雙重證實**:EM-LLM 的 surprise 需要模型自身 token log-probabilities,邊界精修需要每層 attention keys(`refine_from_layer: 20`)——Converse API 兩者都不給。另有一條「可以把切分交給 LLM 自己做」的主張被 **0-3 全票否決** | 確定性切分**定案**,不再視為妥協。surprise 只作為將來可選的本機小模型 perplexity 訊號 |
| 2 | Episode schema 有 provenance / retention / consent | 不夠:Tulving 立場論文(arXiv 2502.06975)定義 episodic = 綁定 **when / where / why**;我們的 schema 只填了 when 與 who | **加 `why`(goal-ref)**;`place` 升為必填。見 §3 |
| 3 | 鞏固走「夜間語義蒸餾」 | **成立,且有經濟學證據**:sleep-time compute(arXiv 2504.13171)在等準確率下把 test-time 計算降 ~5×,一次鞏固攤到多個相關查詢時每查詢成本降 2.5× | 夜間批量**定案**;明確拒絕 A-MEM 的**每次寫入**演化(每次 memory operation ~1,200 tokens,成本隨寫入量線性長) |
| 4 | 鞏固頻率沒有數字 | Generative Agents:重要性累加 **> 150** 觸發反思,實測 **每天 2–3 次** | 反思預算按「每天 2–4 次 LLM 呼叫」規劃,不是每 N 事件 |
| 5 | 遺忘用 Ebbinghaus 衰減 | 機制零 LLM 成本(R = e^(-t/S),召回時 S += 1 且 t 歸零),但 **MemoryBank 論文自稱「exploratory and highly simplified」且沒有任何 decay vs no-decay 的 ablation** | 保留為預設,但**標記為未證實**;A/B 必須包含「不衰減 + 更好排序」這一組 |
| 6 | Graphiti 的理由改為「時序有效性 + 查詢期零 LLM」 | **成立**(四時間戳、`t_invalid` 失效而非刪除、RRF 檢索路徑不呼叫 LLM),但成本比預期高:**`graphiti_core` 裡沒有 Bedrock client**(只有 openai / azure / anthropic-direct / gemini / groq / gliner2),而且**嵌入式 Kuzu 後端已標記 deprecated** | A/B 的 Graphiti 一側先多一項工程債:自寫 Bedrock Converse client 並驗證 tool-use 結構化輸出。**排在扁平方案之後**,不是並列 |
| 7 | 「硬邊界必須在 LLM 之外」是設計律,無外部實證 | **有了**:RoboGuard(IEEE RA-L 2026,UPenn)——root-of-trust LLM **完全看不到使用者輸入**,下游是確定性控制合成閘;最壞情況越獄下不安全計畫執行率從 **>92% 降到 <3%**,且不犧牲正常任務 | L0 動作允許清單從「我們的偏好」升格為**有同儕審查證據的架構**;並照 RoboGuard 補一條:**規則落地的那個模型也不能吃使用者輸入** |
| 8 | 記憶投毒是風險,靠封閉謂詞防 | Letta 的 `read_only` memory block 是 schema 級的可寫性控制——開發者釘住的 persona/values 區塊 agent 自己改不了。MINJA 的 query-only 威脅**只在共享記憶庫下成立**,且良性紀錄從 25 增到 100 條時攻擊成功率從 68.9% 掉到 31.1% | 檔案卡與憲法區塊標記為 **read-only**;記憶庫**按家庭隔離**寫成架構不變量;召回的記憶當**資料**注入,永不進指令區 |
| 9 | 護欄選型待研究(本機 guard 模型 vs 託管) | **本機這條路關閉,但理由不是顯存**:Llama Guard 4/3、ShieldGemma、Granite Guardian **沒有一個官方支援中文**(見 §9.1)。Bedrock Guardrails **Standard tier** 簡中「optimized」、繁中「supported」,價格可忽略(~$0.0006/回合),但唯一的獨立量測是 p50 **749 ms** | 用 **Bedrock Guardrails Standard tier**(Classic 對中文等於沒開);輸入檢查與生成**並行**,輸出走既有句級異步 + 殺聲開關。**粵語官方不支援**,需自建紅隊集 |
| 10 | `affect` 欄位與 salience 用語音能量/笑聲 | **EU AI Act Annex III 1(c)**:從**生物特徵**(語音韻律、臉部)推論情感 = 高風險;純文字情感分析**不算**。Art 50 已於 **2026-08-02 生效** | salience 的**韻律項預設關閉**並獨立開關;`affect` 只從 ASR 文字推,不從聲學特徵推。**新增義務**:Art 50(2) 要求合成音訊有機器可讀標記(我們完全沒做) |
| 11 | 嵌入模型待選 | gte-multilingual-base:Apache-2.0、305M / 611 MB 原生 F16、768 維、C-MTEB Retrieval **71.95**(bge-m3 65.43、mE5-base 61.63) | 選 **gte-multilingual-base**;pin commit(需 `trust_remote_code`);短紀錄延遲 Day-0 自己量 |

---

## 1. Episodic 記憶的已驗證先例(可抄的部分)

| 系統 | 授權 | 存什麼(已驗證) | 鞏固時機 | 對我們的用處 |
|---|---|---|---|---|
| **Generative Agents**(UIST '23,Stanford) | Apache-2.0 | `ConceptNode`:id、type(event/thought/chat)、created + expiration、s-p-o gist 三元組、自然語言描述、embedding key、**poignancy**(重要性/情緒 1–10)、keywords、`filling`(指向證據節點)。三條分型串流 + 關鍵詞倒排索引 | 重要性累加 > 150 → 反思(每天 2–3 次) | **檢索評分的基準線**:recency(每小時 0.995 衰減)+ importance + relevance(餘弦),三者 min-max 正規化後**相加**,α 全為 1 |
| **EM-LLM**(ICLR 2025) | MIT | KV-cache 層的 episodic 事件(token 串流) | 線上,逐 token | 只有**演算法**可移植:surprise 邊界勝過固定長度切分;檢索要**同時取時間相鄰段**。實作不可用(要 logprobs + per-layer attention) |
| **A-MEM**(NeurIPS 2025) | MIT | `MemoryNote`:content、tags、category、`YYYYMMDDHHMM`、自動生成 context、keywords、embedding、links、retrieval_count、evolution_history | **每次寫入**觸發 LLM 演化,改寫最近 k(10–50)個鄰居 | 反面教材:~1,200 tokens/次 memory operation。schema 可抄,寫入策略不可抄 |
| **MemoryBank**(AAAI 2024) | MIT | 分三層:帶時間戳的對話紀錄(episodic)、每日/全域事件摘要(鞏固層)、演化中的使用者人格畫像(semantic) | 每日 1–2 次摘要(含人格提示約 4 次/天) | **最便宜的完整政策**:遺忘是純算術(零 token);檢索是 FAISS,無 LLM |
| **Letta / MemGPT** | 開源 | memory block:label、value、**size limit**(限定它佔多少 context)、description、`read_only` | idle 時的 sleep-time subagent(現稱 dreaming) | 工作記憶層的 schema + **投毒緩解**(read-only 區塊) |
| **Zep / Graphiti** | Apache-2.0 | episode 是一等公民、**非破壞性**原始層(text/message/JSON);實體與關係抽取到語義子圖,靠 `MENTIONS` 邊連回去;每條衍生邊帶 `episodes: list[str]` 血緣 | 寫入時 LLM 抽取 | **時序有效性**(四時間戳)+ 血緣。作者自己把 episode/entity 分裂類比為 Tulving 的 episodic/semantic |
| **Mem0** | 開源 | — | — | 只有延遲數字可用:**memory search p95 = 0.20 s**。準確率行銷不可信(見 §5) |

---

## 2. 切分:為什麼確定性訊號現在是「正解」而非「將就」

三件事同時成立:

1. **surprise 確實比固定切分好**——EM-LLM 的 S/SM/SC 方法在與人工標註事件邊界的對齊上「consistently outperform fixed-interval methods」。這一點沒有爭議。
2. **但它拿不到**:surprise 是模型對下一個 token 的貝氏驚訝度,需要 log-probs;邊界精修(modularity/conductance)需要每層 attention keys。Bedrock Converse 兩者都不暴露。
3. **也不能請 LLM 代勞**:「LLM 已經能把連續經驗切成離散事件」這條主張以 **0-3** 被否決。

所以切分必須是**顯式機制**。我們的機制(靜音 > 45 s / 活動轉換 / 場景變化 / 硬上限 20 turns 或 5 分鐘)在文獻座標裡的位置很清楚:它是**手工編碼的事件邊界**,而遊戲開始/結束、有人進入/離開畫面這些 System-1 已經在偵測的訊號,正好就是家用機器人版本的「beats」。

> 誠實的限度:EM-LLM 的人類對齊證據全部在**文字 token 串流**(播客轉錄)上。沒有任何被驗證的系統在**多模態具身事件流**上做過評測——視覺 + 對話的 surprise 閾值是我們自己的推論,不是被證實的設計。

---

## 3. Schema 修正:補上 `why`,`place` 升必填

Tulving 立場論文(Pink et al., ICML position paper, arXiv 2502.06975)的定義是硬的:

> Episodic memory binds context to its memory content, such as **when, where, and why** an event was encountered.

而且它主張五個性質必須**同時**成立:long-term storage、explicit reasoning、single-shot learning、instance-specific memories、contextual memories。我們原本的 schema 填了 when(`t_start`/`t_end`)與 who(`participants`),`place` 是選填,**`why` 完全沒有**。

```diff
   "kind": "conversation|game|observation|teaching|story",
   "participants": ["user:primary"],
-  "place": "desk",
+  "place": "desk",                      // 必填:where(粗粒度即可)
+  "why": {"goal": "user_requested_joke", "trigger": "utterance|vision_event|schedule"},
+                                        // 必填:why(這段經驗為何發生)——Tulving 的第三個 context slot
   "summary": "...",
```

`why` 不是裝飾:它是**檢索的第二把鑰匙**。「上次我叫你講粵語笑話你講了什麼」這種查詢,靠 summary 的餘弦相似度並不可靠,靠 `why.goal == user_requested_joke` 的結構化過濾則是零成本精確命中。這也是為什麼它必須是**封閉集合的枚舉**,而不是自由文字(見批判文件 §3.5 的設計律)。

---

## 4. 鞏固與遺忘:可以編進預算表的數字

| 動作 | 觸發 | 已驗證成本 | 我們的設定 |
|---|---|---|---|
| 熱層寫入(events.jsonl) | 每個事件 | **0 tokens** | 保持零 LLM |
| Episode 摘要 | 每個 episode 關閉時(異步) | MemoryBank:每日 1–2 次 LLM 摘要即可覆蓋一天 | 便宜模型,離開語音迴路 |
| 語義鞏固(反思) | 重要性累加 > 閾值 | Generative Agents:閾值 150 → **每天 2–3 次** | 夜間批量 + 白天門檻兜底,預算 **2–4 次 Sonnet 呼叫/天** |
| 衰減 / 遺忘 | 每次讀取後更新 | **0 tokens**(R = e^(-t/S),S += 1 且 t 歸零) | 預設開,但列入 A/B 對照(見下) |
| 每次寫入演化(A-MEM 式) | 每次寫入 | ~1,200 tokens/op | **不做**。成本隨寫入量線性成長,對一天講幾百句的機器人不可接受 |

**必須寫進 A/B 的對照組:** MemoryBank 的 Ebbinghaus 衰減**沒有任何 ablation** 證明它提升檢索品質,論文自稱「exploratory and highly simplified」,而且開源實作裡那條規則是否真的被呼叫也未被確認(grep `forget` 零命中)。所以評測集的三組必須是:①衰減 ②重要性加權剪枝 ③不刪除但改善排序。在有自己的數字之前,「遺忘讓記憶更好」只是設計美學。

> 另一個必須分開講的區別:Ebbinghaus 衰減是**機率性的檢索抑制**,不是刪除。它**不能**用來滿足「忘記我說的 X」——那條路必須是確定性的刪除交易(批判文件 §2.5)。把兩者混為一談是這個設計最容易犯的錯。

---

## 5. 延遲:檢索必須留在非 LLM 路徑上

| 量測 | 數字 | 誰量的 |
|---|---|---|
| Mem0 **memory search** p95 | **0.20 s** | 廠商自評 |
| Mem0 端到端 p95(含生成) | 1.44 s | 廠商自評 |
| Zep on LongMemEval 端到端平均 | 2.58 s(gpt-4o)/ 3.20 s(gpt-4o-mini),對照全上下文 ~29 s | 廠商自評 |
| Graphiti 查詢 | 「typically sub-second」,hybrid = 餘弦 + BM25 + 圖 BFS,預設 RRF 路徑**不呼叫 LLM** | README 比較表,**無 benchmark、無 p95** |

**設計後果:**完整的「記憶增強回答」迴路實測落在 **1.4–3.2 s**,已經**達到或超過**我們 2 秒的語音預算。所以:

- 檢索必須是**非 LLM 路徑**(嵌入 + BM25 + 結構化過濾),且與 ASR 尾段/LLM prefill **並行**;
- 寫入(尤其是 Graphiti 的實體抽取)**永遠不在語音迴路裡**;
- 超過 p95 150 ms 就**降級**到只用常駐檔案卡,而不是阻塞。

> 全部四個數字都是廠商量自己、單一資料集、無獨立複現。上線前必須在我們自己的箱子上量。而且 Mem0 自己的論文承認全上下文**更準**(J 72.90% vs 66.88%)——記憶系統換來的是成本與延遲,不是準確率。

---

## 6. Graphiti 在我們這台機器上的真實代價

被逐行驗證過的四項:

1. **需要外部圖資料庫**:Neo4j 5.26 / FalkorDB 1.1.2 / Neptune(還要加 OpenSearch Serverless)。**嵌入式 Kuzu 已標記 deprecated** —— 第四輪「全嵌入式起原型」的前提正在消失。要和 MuJoCo + YOLO + FunASR + Kokoro 一起擠 4 vCPU / 15 GB RAM。
2. **`graphiti_core` 沒有 Bedrock client**:只有 openai、azure、anthropic-direct、gemini、groq、gliner2。`anthropic_client.py` 直接 import `AsyncAnthropic`(直連 API)。要用我們的 Bedrock 路徑,得**自寫一個 Converse client 並驗證 tool-use 能滿足它的 structured-output 要求**——README 明確警告不支援結構化輸出的服務會「result in incorrect output schemas and ingestion failures」。
3. **每個 episode 寫入都要 LLM**:實體/關係抽取、去重、邊失效判定。連本機 GLiNER 路徑也仍然 `raise` 要求一個 `llm_client`。
4. **級聯刪除是不完整的**:`remove_episode` 會留下孤立實體(issue #1083, 2025-11;PR #1130 待處理),而且從已刪除 episode 合成出來的實體摘要**仍然留著**,community 也不更新。

**結論:** Graphiti 的**時序有效性**(四時間戳、`t_invalid` 失效而非刪除、可查「當時是什麼」)確實是我們要的,而且是它獨有的。但把它當成「起原型最快的路」是錯的——它現在是 A/B 裡**成本較高**的那一側,先做扁平方案 + 時間評分,再拿凍結的 30 條查詢去比。

**而且 GDPR 級的刪除必須我們自己做**:沒有任何被調查的開源記憶系統實作了「使用者聲稱 / 機器人推論 / 機器人觀察」的**一等公民 trust 欄位**,Graphiti 的級聯也不完整。所以 `provenance` + `source_episode_ids` + 我們自己的 tombstone/重新推導層不是可選項。

---

## 7. 道德邊界安全:確定性那一半終於有實證

### 7.1 RoboGuard:92% → 3%

IEEE RA-L 2026(UPenn,Ravichandran / Robey / Kumar / Pappas / Hassani)。兩階段:

- **階段一**:一個 root-of-trust LLM,**依設計看不到使用者提示**(只吃預定義安全規則 + 機器人 API + 世界模型更新),把規則落地成時序邏輯規格;
- **階段二**:確定性的 LTL 控制合成(Büchi 自動機模型檢查,**無 LLM**),對規劃 LLM 產出的候選計畫做閘控。

實測:最壞情況越獄攻擊(RoboPAIR 類)下,不安全計畫的執行率從 **>92% 降到 <3%**,且不降低正常計畫的表現。

這對我們有兩個直接後果:

1. L0 的動作允許清單(`play_gesture` 的 dict 查表、停止詞、冷卻)不再只是「我們認為對」——它是有同儕審查證據的架構;
2. **多一條我們原本沒有的規則**:把安全規則落地的那個組件,**也不能吃使用者輸入**。我們的憲法在系統提示裡,和使用者話語在同一個 context 裡——這正是 RoboGuard 明確要隔離的東西。修法:憲法 + 檔案卡標記為 read-only 區塊(Letta 模式),且輸出守衛的判定提示**不含**使用者原文,只含待播句子與規則。

### 7.2 為什麼不能靠模型自己拒絕

RoboPAIR(ICRA 2025)在三個 LLM 控制的機器人系統上(NVIDIA Dolphins 自駕 LLM、GPT-4o 規劃的 Clearpath Jackal、GPT-3.5 整合的 Unitree Go2)越獄,「often achieving 100% attack success rates」。一個具身 agent **不能**把 LLM 自己的拒絕行為當作安全邊界。這條就是我們「硬邊界在 LLM 之外」的外部證據。

### 7.3 記憶帶來的攻擊面(未完全驗證,但方向明確)

第一輪抽取到 MINJA(arXiv 2503.03704,NeurIPS 2025)——**query-only** 威脅模型:攻擊者不需要對記憶庫的寫入權限,只靠正常提問 + 觀察輸出就能植入惡意紀錄。對一個把每次互動都寫回 episodic 記憶的陪伴機器人,這是**直接對應**的風險。這條與相關的數字落在本輪的 gap-verification 裡(見 §9)。

已驗證的緩解只有一個,而且是 schema 級的:**Letta 的 `read_only` 區塊**——開發者釘住的 persona/values 內容,agent 自己的記憶工具改不了。這與我們的封閉謂詞是同一個思路的兩層:**不是把壞句子過濾掉,而是讓它沒有欄位可以落地**。

---

## 8. 被殺掉的主張(3 條)

| 主張 | 票數 | 影響 |
|---|---|---|
| 「LLM 已經能把連續經驗切成人類式離散事件,所以切分可以交給 LLM」 | **0-3** | 切分必須是顯式機制。這條若當真,我們會寫出一個每天呼叫幾百次 LLM 又不可靠的切分器 |
| Mem0 在 LOCOMO 上「相對 OpenAI 記憶系統提升 26%、勝過全部六類基準」 | **0-3** | Mem0 的準確率行銷不可用。只有延遲數字(search p95 0.20 s)倖存,而且同一篇論文承認全上下文更準 |
| Mem0 的「extraction + update 鞏固流水線」框架描述 | **1-2** | 不要照它的流水線敘事設計我們的鞏固層 |

---

## 9. Gap verification:上兩輪抽取但從未投票的主張

> **方法上的重點,值得單獨說:**第一個工作流的 caveats 寫著「Bedrock Guardrails 價格/延遲、NeMo、Llama Guard、MINJA、EU AI Act、嵌入模型比較——**沒有存活的主張**」。但翻 fetch 階段的原始輸出,這些主張**全都被抽取出來了**,只是驗證預算只覆蓋了前 25 條,它們落在 `budgetDropped: 5` 裡從未被投票。
>
> **「未驗證」不等於「不存在」。**把它們當成「研究不到」會讓我們在沒有數字的情況下自己編一個。所以本輪追加了一次針對這 6 條的對抗驗證(每條 2 個不同視角的驗證者)。

結果:**6 條全部存活**(1 CONFIRMED+補齊、2 CONFIRMED、1 CONFIRMED、2 CORRECTED),而且其中兩條**直接翻轉了設計決策**。

### 9.1 本機 guard 模型:被語言判死,不是被 VRAM

這是本輪最重要的單一發現。原本的問題是「Llama Guard 級模型在 T4 上還剩多少 VRAM」——**問錯了**:

| 候選 | 參數 / BF16 權重 | 授權 | 官方語言 |
|---|---|---|---|
| Llama Guard 4 | 12B / **24.00 GB** | Llama 4 Community | En + fr/de/hi/it/pt/es/th —— **無中文** |
| Llama Guard 3 1B | 1.50B / 3.00 GB | Llama 3.2 Community | 同上 8 種 —— **無中文** |
| ShieldGemma 2B / 9B | 2.61B / 5.23 GB · 9.24B / 18.48 GB | Gemma Terms | **僅英文** |
| Granite Guardian 3.1-2b … 3.3-8b | 2.53B / 5.07 GB … 8.17B / 16.34 GB | **Apache-2.0** | 卡片自述 "only trained and tested on **English** data" |

VRAM 那條也確認成立(T4 16 GB 無 BF16 硬體、FP16 同樣 24 GB、13 GB 餘量裝不下 12B;INT8 約 12 GB 只剩 <1 GB 餘裕)。但**這已經不重要了**:一個聽不懂中文的守衛,對一台講普通話與粵語的家用機器人是零價值。

**→ 決策:本機 guard 模型這條路關閉。**理由寫清楚是「語言覆蓋」,不是「顯存不夠」——因為顯存問題會隨硬體改善,語言問題不會。

### 9.2 Bedrock Guardrails:價格不是問題,延遲與粵語才是

| 項目 | 已驗證數字 | 來源 |
|---|---|---|
| Content filters / Denied topics | **$0.15 / 1,000 text units**(1 text unit = 最多 1,000 字元;CJK 按字元計) | AWS pricing page,2026-08-22 |
| 一個完整護欄回合(兩種政策 × 輸入+輸出) | **≈ $0.0006** | 同上算術 |
| 獨立 `InvokeGuardrailChecks` 路徑 | content filters $0.07/1k、prompt attack $0.08/1k、PII $0.10/1k | 同上 |
| 官方 p50/p99 延遲 | **AWS 完全沒有公布**(兩個驗證者掃過所有官方面) | — |
| 唯一的獨立量測(n=15,us-east-1,含 ~243 ms client RTT) | Classic p50 441 / p90 478 ms;**Standard p50 749 / p90 907 / max 1837 ms** | 單一來源,僅供參考 |
| 語言 | Standard:簡中「optimized」、繁中「supported」;**Classic:僅 en/fr/es**;**粵語(yue):AWS 文件零提及** | AWS supported-languages 文件 |
| 中文有害輸入探測(單一來源) | Standard 2/2 攔下;**Classic 0/2** | 驗證者實測 |

**→ 決策三條:**
1. 只能用 **Standard tier**(Classic 對中文等於沒開)。這會強制 cross-region inference。
2. p50 約 750 ms 吃掉 2 秒預算的三分之一 → 輸入檢查**與 LLM 生成並行**,輸出檢查走**既有的句級異步 + 殺聲開關**(批判文件 §3.1),而不是串行等待。
3. **粵語在任何 tier 都不受官方支援**。AWS 明確警告未支援語言的護欄無效。所以粵語路徑必須有**我們自己的紅隊測試集**,而且在有數字之前,不能對外聲稱粵語受保護。

### 9.3 MINJA:數字修正,但更重要的是**前提條件**

| 原始主張 | 修正後 |
|---|---|
| ">95% ISR / 70% ASR" | **98.2% 平均 ISR / 76.8% 平均 ASR**(arXiv 2503.03704 v5, NeurIPS 2025)。「ISR 一致超過 95%」精確;但 **ASR >70% 只在 6 個配置中的 3 個成立**(範圍 57.0–98.9%) |
| "query-only,不需要寫入權限" | **只在共享記憶庫下成立**。作者自己承認:面對 per-user 隔離的記憶庫,攻擊需要身分偽裝或帳號劫持。另外攻擊者必須知道受害者的關鍵詞,且 agent 必須自動寫入紀錄 |

還有兩個數字改變了防線的設計:

- **ASR 隨真實度衰減**:良性紀錄從 25 條增到 100 條時,ASR 從 **68.9% 掉到 31.1%**(EHRAgent/MIMIC-III);對已有污染的記憶庫再攻擊,低 6–20 個百分點。
- **偵測型防禦無效**:GPT-4o 的提示層偵測在 RAP 上 **0/135**、QA Agent 上 **0/90**;時間衰減被作者明確指出可繞過。

**→ 決策:移除攻擊的前提,而不是偵測攻擊。**
1. **記憶庫按家庭/使用者隔離**(這正是作者承認會破壞 query-only 模型的那一點)。我們本來就是單戶部署,把它寫成**明確的架構不變量**而不是巧合。
2. **原始互動紀錄不自動進入語義記憶**——攻擊搭的就是 auto-write 那條路。我們的三級寫入(熱層 → 暖層摘要 → 夜間語義)加上封閉謂詞,正好是一個寫入閘。
3. **不把提示層偵測或時間衰減當主控**(前者實測 0 命中,後者可繞過)。
4. **被召回的記憶是資料,永遠不是指令**——注入 LLM 時必須有結構化邊界,不能拼進系統提示的指令區。

範圍限制要誠實:MINJA 只在 GPT-4/GPT-4o/DeepSeek-R1 的**文字** agent + 樸素 top-k 餘弦檢索上測過(all-MiniLM-L6-v2 / ada-002),論文裡 **Claude、Bedrock、Mem0、語音、機器人、中文全部零出現**。轉移到我們的棧是**推論**,不是實證。

### 9.4 EU AI Act:Article 50 從今天起生效,而且它管到我們的 TTS

| 條款 | 內容 | 對我們 |
|---|---|---|
| Art 50 適用日 | **2026 年 8 月 2 日**(Art 113)—— 已經生效 | 現在就適用 |
| Art 50(1) | 提供者必須確保使用者知道自己在與 AI 互動,「除非顯而易見」;**recital 132 要求以兒童與弱勢群體為標準判斷「顯而易見」** | 我們憲法第一條「永遠誠實說自己是機器人」不只是價值觀,是**法定義務**;而且不能靠「一看就知道是機器人」豁免 |
| Art 50(5) | 揭露最晚在**首次互動時**,且須符合無障礙要求 | 開場即揭露,不是被問才答 |
| Art 50(2) | **合成音訊必須有機器可讀的標記** | **我們完全沒做過這件事**。TTS 輸出(Kokoro + CosyVoice)需要可機器讀的標記 |
| Art 50(3) | **部署者**須告知被情感辨識系統作用的人,並符合 GDPR | 若保留情感推論,demo 的部署者(= 使用者自己)也承擔義務 |
| Art 5(1)(f) | 情感推論**禁令**只涵蓋**職場與教育**(2025-02-02 起生效) | 家用機器人**不在禁令範圍**內 |
| Annex III 1(c) | 從**生物特徵**資料做情感辨識 = **高風險**。語音韻律與臉部符合 Art 3(39)/(34) 的生物特徵定義;**純文字情感分析不算** | ← **這是我們最大的一個設計槓桿** |
| 高風險義務適用日 | 原 2026-08-02,經 **Reg (EU) 2026/1744 延後到 2027-12-02**(單一驗證者,但來源是 EUR-Lex 一次文獻) | 有緩衝,但不是豁免 |
| 罰則 | Art 5 違反 €35M / 7%;Art 50 違反 €15M / 3% | — |

**→ 兩個必須改的設計點:**

1. **`affect` 欄位與 salience 公式裡的「語音能量/笑聲」都是從生物特徵(語音韻律)推論情感** —— 這正好落在 Annex III 1(c)。批判文件 §2.4 的 salience 公式必須改:預設只用**非生物特徵**訊號(明確要求記住、新穎度、互動長度、outcome),韻律項**預設關閉**並做成獨立開關。純文字情感(從 ASR 文字判斷「他說他很開心」)不在範圍內,可以保留。
2. **合成音訊的機器可讀標記**進 build order。這是一個我們完全漏掉的義務,而且技術上很便宜(在 PCM 或容器層加標記),比事後補要容易得多。

### 9.5 嵌入模型:選 gte-multilingual-base

| 模型 | 授權 | 參數 / 檔案 | 維度 / 上限 | 中文檢索 |
|---|---|---|---|---|
| **Alibaba-NLP/gte-multilingual-base** ← 選它 | **Apache-2.0** | 305M / **611 MB 原生 F16** | 768 / 8192 | **C-MTEB Retrieval nDCG@10 = 71.95**;MTEB-zh 平均 62.72 |
| BAAI/bge-m3 | MIT | ~568M / 2.27 GB fp32(**官方沒有 fp16 檔案**,1.14 GB 是載入時轉換) | 1024 / 8192 | C-MTEB 65.43;MTEB-zh 60.80;**但 MIRACL zh 勝出**(62.7 vs 60.8) |
| intfloat/multilingual-e5-base | MIT | 278M | 768 / **僅 512** | C-MTEB 61.63;MTEB-zh 56.21 —— **每個中文基準都最後** |

修正兩處原始說法:gte 是「**over 70 languages**」而不是 100+;multilingual-e5-base 的上下文上限是 **512** 而不是 8192。

**→ 決策:gte-multilingual-base。**授權最乾淨、體積最小(611 MB,CPU 推論可行 → 省 T4 顯存給 YOLO/TTS)、768 維索引更快(faiss 15.07 ms vs 1024 維的 20.35 ms)、中文檢索在兩個基準上領先。兩個要記住的代價:它需要 `trust_remote_code`(供應鏈考量,要 pin commit),而且**沒有任何一手來源涵蓋短紀錄(<100 tokens)的 CPU/GPU 延遲**——這是 Day-0 必須自己量的 30 分鐘工作。

> 排名是**基準相依**的:gte 贏 C-MTEB 與 MTEB-zh,bge-m3 贏 MIRACL zh。唯一穩定的結論是 multilingual-e5-base 墊底。所以這個選擇應該在自己的 30 條查詢評測集上覆核。

### 9.6 NeMo Guardrails(供參考,不進 build order)

Apache-2.0(`LICENSE.md` 的 SPDX 標頭;GitHub API 顯示 NOASSERTION 是偵測誤判),最新 0.23.0,五類 rails(input / dialog / retrieval / execution / output)確認。可以完全不呼叫 LLM 的部分:越獄啟發式(自述 "No LLM calls were made")、Colang core、語言偵測(~12 μs)。但實測延遲不樂觀:越獄啟發式平均 115 ms(Docker GPU)/ **3,227 ms(in-proc CPU)**;dialog rails 約為無護欄呼叫的 **3 倍**延遲與成本(arXiv 2310.10501 §7.2);AlignScore ~45 ms(記憶體內)/ ~220 ms(HTTP)。**沒有公布通用的每-rail 開銷數字。**

**→ 它是編排選項,不是安全模型的替代品。**有用的 ML rails 要搶已經很緊的 T4,在 CPU 上跑則直接炸掉 2 秒預算。我們自己的確定性 L0 + Bedrock Guardrails 已覆蓋同樣的位置。

### 9.7 這一輪之後仍然沒有數字的部分

- Bedrock Guardrails 從 **ap-southeast-1** 的 server-side p50/p95(現存唯一量測是 n=15 的 client-side)
- 粵語護欄有效性的**任何**量化數字
- gte-multilingual-base 在**我們這台 g4dn.xlarge** 上、短紀錄的 CPU 與 GPU 延遲
- MINJA 對 Claude 系模型、生產記憶框架(Mem0 / AgentCore Memory)、語音、非英語 agent 的轉移性(全部未測)
- Reg (EU) 2026/1744 是否還有其他相關修改(只有一個驗證者讀過)

---

## 10. 仍然開放的問題

1. **多模態切分閾值**:surprise 式邊界偵測能否轉移到「VLM 幀描述 + 對話 + 遊戲節拍」的混合流?或者 System-1 的 beats(遊戲開始/結束、人進入/離開)就是答案?這只能在我們自己一天的真實日誌上測。
2. **Graphiti 每 episode 的真實寫入成本**:透過自寫的 Bedrock Converse client,一次 episode ingest(抽取 + 去重 + 邊失效)實際燒多少 Sonnet token,以及 Converse 的 tool-use 是否真的滿足它的 structured-output 要求。
3. **cognee 完全沒有存活的主張**——第四輪兩個候選裡的一個,在兩輪研究後仍然是空白(episode 等價單元、時序模型、每次寫入成本全部未驗證)。所以「Graphiti vs cognee」目前是**單邊比較**。建議:不要為了補齊比較而研究 cognee,直接把扁平方案當對照組(它是我們真正的 fallback)。
4. **遺忘是否真的有幫助**:沒有任何論文做過 decay vs no-decay 的 ablation。這必須在我們自己的檢索軌跡上測,而且它與「被遺忘權」的刪除路徑有交互。
5. 所有延遲與成本數字都是論文或廠商自評,**沒有一個是在我們的 T4 / Bedrock / 中-粵-英 組合上量的**。

---

## 11. 對 build order 的影響

批判文件 §4 的順序**不變**,但兩項的內容被這一輪改寫:

- **第 6 步(Graphiti A/B)**多一項前置工程債:自寫 Bedrock Converse client + 驗證 structured output + 決定圖資料庫(Kuzu 已 deprecated,所以是 FalkorDB 或 Neo4j 進程,要算進 15 GB RAM)。**扁平方案先做完並凍結數字**,再開這一側。
- **第 2 步(憲法 + L0)**加一條 RoboGuard 的規則:把規則落地/判定的那個組件與使用者輸入隔離;憲法與檔案卡以 read-only 區塊承載。

- **第 7 步(句級異步輸出守衛)**現在有數字可以排了:Bedrock Guardrails Standard tier,$0.15/1k text units,p50 約 750 ms(單一獨立量測)。因為 750 ms 遠超一句話的播放間隙容忍度,守衛必須是**異步 + 殺聲**,不能是串行前置。第一件事是在 ap-southeast-1 量自己的 p50/p95。

新增三步:

- **4.5 `why` 欄位的枚舉定案**:`why.goal` 必須是封閉集合(`user_requested_*` / `game_*` / `vision_event` / `scheduled_*`),因為它同時是檢索鍵與安全邊界的一部分——自由文字的 goal 欄位就是另一個讓指令型內容落地的地方。
- **2.5 AI 身分揭露 + 合成音訊標記**(插在憲法之後,因為 Art 50 已生效):開場即揭露「我是機器人」,不依賴「顯而易見」豁免(recital 132 要求以兒童為標準);TTS 輸出加機器可讀標記。兩件都便宜,而且都是**法定義務**而非最佳實務。
- **3.5 salience 公式去生物特徵化**:預設只用非生物特徵訊號;韻律/笑聲項做成預設關閉的獨立開關。改在檔案卡之後、熱層之前,因為公式一旦寫進資料就難改。

---

## 附:來源

Episodic / 記憶架構:
- Generative Agents — https://arxiv.org/abs/2304.03442 · https://github.com/joonspk-research/generative_agents (Apache-2.0)
- EM-LLM — https://arxiv.org/abs/2407.09450 · https://github.com/em-llm/EM-LLM-model (MIT) · ICLR 2025
- Episodic-memory position paper — https://arxiv.org/abs/2502.06975
- A-MEM — https://arxiv.org/abs/2502.12110 · https://github.com/agiresearch/A-mem (MIT)
- MemoryBank — https://arxiv.org/abs/2305.10250 · https://github.com/zhongwanjun/MemoryBank-SiliconFriend (MIT)
- Sleep-time compute — https://arxiv.org/abs/2504.13171 · https://www.letta.com/blog/sleep-time-compute
- Letta memory blocks — https://www.letta.com/blog/memory-blocks · https://docs.letta.com/guides/agents/memory-blocks
- Zep / Graphiti — https://arxiv.org/abs/2501.13956 · https://github.com/getzep/graphiti (Apache-2.0)
- Mem0 — https://arxiv.org/abs/2504.19413

安全:
- RoboGuard — https://arxiv.org/abs/2503.07885 · https://robo-guard.github.io/ · IEEE RA-L 2026
- RoboPAIR — https://robopair.org · https://arxiv.org/abs/2410.13691 · ICRA 2025
- Constitutional Classifiers — https://arxiv.org/abs/2501.18837
- MINJA — https://arxiv.org/abs/2503.03704 · NeurIPS 2025
