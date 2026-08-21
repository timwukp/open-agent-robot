# Round 4 — Agent Brain: Memory / Skills / MCP+OAuth / Critical Thinking / Framework

**Date:** 2026-08-21 · **Method:** deep-research workflow (111 agents: 138 claims extracted, 25 adversarially verified) + targeted gap-verification (12 agents, 3-vote panels on the 3 highest-impact claims). All licenses/specs checked against primary sources on 2026-08-21. Constraint frame: thin `agent_node` on Bedrock Converse (Sonnet 5 global profile), EC2 g4dn (16GB RAM, T4), open-source project.

## 0. Executive recommendations & build order

| # | Build | Pick | Why |
|---|---|---|---|
| 1 (先做,最便宜) | **技能加載器**(~50 行) | Agent Skills 開放標準(agentskills.io) | 無任何特殊 API——系統提示注入技能索引(每技能 ~100 tokens)+ 一個 read_file 工具即可;首個技能 = 你的 critical-thinking-skills |
| 2 | **記憶系統** | 原型用 **cognee**(默認全嵌入式 Kuzu/LanceDB/SQLite);並行評測 **Graphiti+FalkorDB Lite**(聯想式召回更強) | 兩者皆 Apache-2.0;Graphiti 查詢期零 LLM 調用(亞秒級),但嵌入式後端故事在弱化 |
| 3 | **框架** | 漸進採用 **Strands Agents**(Apache-2.0,AWS 出品) | 原生 MCP client + Bedrock 一等公民 + hooks 攔截點(正好掛 System-1 反射/記憶寫入/技能注入);架構本來就和我們的薄層 loop 同構 |
| 4 | **MCP+OAuth** | mcp python-sdk **v2**(MIT)+ **Client Credentials**(機器對機器,無需瀏覽器) | 被駁回的「無頭無解」是錯的——ext-auth 擴展 + `ClientCredentialsOAuthProvider` 是官方無頭路徑 |
| 5 | **批判思維** | 你的 skill 作為**觸發式**技能 + **僅在有外部反饋時做自我修正** | ICLR 證據:盲目自我批判會降低準確率(見 §4,反直覺但三票確認) |

## 1. 記憶系統(角度 1,驗證充分)

**目標**:類皮層聯想記憶網(知識圖譜連結事件)+ 快速搜索,跑在 16GB 盒子上。

| 系統 | 授權 | 嵌入式? | 關鍵驗證事實 |
|---|---|---|---|
| **Graphiti**(getzep,Zep 的核心) | Apache-2.0(3-0) | ⚠️ Kuzu 後端**已棄用**;唯一免伺服器選項 = FalkorDB Lite(pip extra,**需 Python 3.12+**——我們的 venv 正是 3.12 ✓) | 時序知識圖譜(每條事實帶時間有效期);**查詢期零 LLM 調用**,混合 semantic+BM25+圖遍歷,典型亞秒(但查詢期仍有一次 embedding 調用;「P95<200ms」是 Zep 託管服務的數字,不是自架的);**攝入期需要 LLM**(實體抽取/去重,大內容會慢);Bedrock 不在官方 provider 列表——可注入 `AsyncAnthropicBedrock`(鴨子類型相容但未文檔化),embedder 仍需自寫 |
| **cognee** | Apache-2.0 | ✅ 默認全嵌入(Kuzu/LanceDB/SQLite) | remember/recall/forget/improve 語義 + 跨會話持久;**基準證據最弱**(無 LoCoMo/LongMemEval 數字) |
| Mem0 / Mem0-graph | Apache-2.0 | 部分 | 自報「p95 延遲降 91%、token 省 >90%」(vs 全歷史入上下文);**自家數據顯示 graph 版只比非 graph 高 ~2 分**——圖存儲的額外設施須由聯想召回需求證成,不是 QA 準確率 |
| HippoRAG 2 | (海馬體啟發,神經生物學對齊你的訴求) | — | 聯想召回 +7% 是這批證據裡最強的,但由文檔 QA 代理測得,非對話記憶 |
| 基準大戰警示 | — | — | **所有頭條數字都是廠商自報且互撕中**(Zep 稱 LongMemEval +18.5%;Zep 反過來批 LoCoMo 對話僅 16-26k tokens、塞得進上下文窗、缺知識更新題);Letta 主張現有基準全是靜態檢索、提出自家排行榜(也是利益方) |

**判定**:先用 cognee 起原型(零伺服器、一天可跑),同時以我們自己的機器人事件流(對話+視覺事件)做 A/B 評 Graphiti——聯想式「有脈絡的記憶網」是 Graphiti/HippoRAG 路線的主場。

## 2. 技能加載(角度 2,3-0 驗證)

- **Agent Skills 已是跨廠商開放標準**(agentskills.io):一個技能 = 一個文件夾 + `SKILL.md`,frontmatter **只必填 name + description**;約定目錄 `scripts/ references/ assets/`(你的 repo 的 `methods/ templates/ references/` 路由結構完全同構)
- **三級漸進披露**(帶 token 預算):L1 = 每技能 name+description(~50-100 tokens)常駐系統提示;L2 = SKILL.md 正文(建議 <5k tokens / <500 行)在觸發時載入;L3 = 捆綁文件按需讀取
- **沒有任何特殊 API 機制**——連 Anthropic 自家實現也只是提示注入 + 文件讀取。我們的 Bedrock Converse loop 用「系統提示技能索引 + read_skill_file 工具」即可 100% 復刻
- Claude Agent SDK 也支持從文件系統無頭加載技能,但 **`allowed-tools` frontmatter 被忽略**(spec 裡本就標 Experimental)
- 你的 `timwukp/critical-thinking-skills`:MIT ✓,但 **repo 目前是 private**——機器人拉取需要 token,或轉 public
- 技能 vs 系統提示 vs 工具:常駐行為 → 系統提示;確定性計算 → 工具;**按需的方法論/流程知識 → 技能**(token 經濟性最好)

## 3. MCP + OAuth 無頭認證(角度 3;原結論被 3-0 駁回並修正)

規範事實(2025-06-18 與 2025-11-25 修訂,均已核對):
- OAuth 2.1 基座;**PKCE 對 client 是 MUST**(2025-11-25 還要求拒絕不聲明 code_challenge_methods_supported 的 AS);**RFC 8707 resource 參數 MUST**(授權與換 token 兩處);**RFC 9728 發現 MUST**(2025-11-25 起 WWW-Authenticate 可選、可回退 .well-known);2025-11-25 新增 **CIMD**(SHOULD)與 OIDC Discovery,DCR 降為 MAY;stdio 傳輸不用 OAuth(從環境取憑證)
- **修正(好消息)**:「無頭機器人無路可走」是錯的——(1) 官方 **ext-auth 擴展倉庫有 Client Credentials 擴展草案**(核心 spec 明文引用 client_credentials 客戶端)+ 穩定的 Enterprise-Managed Authorization;(2) **python-sdk v2(2026-07-28,MIT)已內建 `ClientCredentialsOAuthProvider`** 等——M2M 無瀏覽器路徑是官方存在的
- 瀏覽器授權碼流(給需要「用戶身份」的 MCP server)仍需一次性人工授權:demo 期在 MacBook 上代辦一次、token 持久化(SDK 只給 TokenStorage 協議,存儲安全自理——放我們 EC2 的加密盤 + 0600)
- 託管替代:**AgentCore Gateway**(inbound/outbound OAuth、Token Vault、工具語義搜索、官方支持 Strands/LangGraph 客戶端)——能力強但屬 AWS 託管服務,開源項目主線保持自管,Gateway 列為企業部署選項

## 4. 批判思維(角度 4)— 反直覺的核心證據(3-0 確認)

- **盲目自我批判有害**:Huang et al.(ICLR 2024)——無外部反饋的 intrinsic self-correction **降低**準確率(GPT-3.5 GSM8K 75.9→75.1→74.7;**CommonSenseQA 75.8→38.1** 才是典型跌幅);模型判斷不了自己推理的對錯,正→錯的翻轉多於錯→正
- Reflexion 的光環要拆開看:推理/QA 的收益依賴 oracle 停止訊號;**代碼任務的收益來自執行反饋**——落在「有效外部反饋則有效」的豁免區
- 多智能體辯論 ≤ 等算力的 self-consistency 多數投票(GSM8K 83.0 vs 88.2)
- 2024-2025 的 SCoRe(ICLR 2025)證明可用 RL **訓練出**內在自我修正(+15.6% MATH)——但那是訓練期注入 oracle,對我們調用凍結 API 模型不適用
- SwiftSage:快/慢混合(小模型直覺 + LLM 深思)勝過純提示與純反思——**佐證我們既有的 System-1/System-2 架構是對的**
- 你自己的 skill 文檔也寫了:深度批判模式要門控,對技術/事實問題反而降質 ✓

**設計結論**:批判思維技能 = **觸發式**(質疑/驗證/紅隊類請求才載入);自我修正**只掛外部反饋源**(工具結果、視覺幀、代碼執行、記憶檢索),絕不做無據自查第二遍。

## 5. Sonnet 5 API 面(3-0,直接可操作——我們的 agent_node 要照此改)

- `thinking:{type:"enabled",budget_tokens}` → **400**;`{type:"disabled"}` 仍接受(Fable 5 連 disabled 都 400);省略 thinking = 默認 adaptive
- **控制推理深度的正確方式**:`thinking:{type:"adaptive"}` + `output_config:{effort:"low"|"medium"|"high"|"xhigh"|"max"}`(默認 high;effort 放 output_config 裡,放錯位置是 ValidationException);Bedrock Converse 經 `additionalModelRequestFields` 傳
- 語音迴路建議:日常對話 effort **low**(降延遲),批判思維技能觸發時升 **high/xhigh**——這就是 System-2 的油門
- 坑:`thinking.display` 默認 omitted(要看推理設 "summarized");**Bedrock 上強制 tool_choice 必須配 thinking disabled**;temperature/top_p/top_k 全部 400(已踩過);切換 thinking 配置會失效 prompt cache
- Strands 注意:默認模型是 Claude 4 Sonnet(us-west-2)——必須顯式傳 `BedrockModel(model_id=...)`;repo 已遷入 monorepo(strands-agents/harness-sdk,PyPI 名不變)

## 6. 目標架構(增量遷移,不推倒重來)

```
agent_node(現有薄層,漸進換 Strands Agent 殼)
 ├─ System-1(不變):game reflex / stop words / barge-in — 永不過 LLM
 ├─ SkillLoader:掃 skills/ 目錄 → L1 索引入系統提示;read_skill_file 工具
 │    └─ skills/critical-thinking(你的 repo,首個)
 ├─ Memory:cognee(嵌入式)寫入對話輪+視覺事件;recall 注入 <memory> 標籤
 │    └─ A/B: Graphiti+FalkorDB Lite(聯想召回)
 ├─ MCP:mcp python-sdk v2 — stdio(本地工具)+ ClientCredentials(M2M 遠端)
 └─ effort 路由:平時 low;技能觸發/複雜推理 high~xhigh(adaptive thinking)
```

## 7. 未決(誠實邊界)
- HippoRAG 2 直接整合成本未評(研究代碼形態);記憶寫入的每輪 LLM 成本要實測(Graphiti 攝入調用數未量化)
- AgentCore Gateway vs 自管的深度對比只驗證了能力面,未驗證成本/延遲
- 你的 critical-thinking-skills repo 需轉 public 或配 token 才能被機器人拉取

*來源:agentskills.io spec、modelcontextprotocol.io(2025-06-18 / 2025-11-25)、getzep/graphiti、topoteretes/cognee、mem0ai、strands-agents、mcp python-sdk v2、Anthropic extended-thinking 與模型遷移文檔、arXiv 2310.01798(Huang et al.)、2409.12917(SCoRe)、2305.17390(SwiftSage)、2501.13956(Zep)。逐條票決記錄:workflows wf_4bb08ad4-f74 + wf_a34bcaa0-c0c。*
