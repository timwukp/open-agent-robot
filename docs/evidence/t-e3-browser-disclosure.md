# T-E3 — Art 50 on the browser face (L5 browser walk-through)

Test item: §6.2 E / T-E3 — "元素可見,且**計算後顏色 ≠ 背景色**(class token 存在不等於顏色生效)".
Level: **H + D** (§6.0). §6.3 places this at **L5**, run by hand with a browser present.
Date of this run: 2026-08-23. Engine: Chromium via Playwright, viewport 1280×720, macOS.

CI (`tests/test_disclosure.py`) only checks *structure*: that the badge is static markup,
that it declares an explicit foreground and background, and that the two rules which keep
it un-occluded exist. Whether a colour reached a pixel can only be settled by a renderer,
so that half is recorded here.

## How to reproduce

```
python3 web/stub_ws_server.py            # http://127.0.0.1:8099/ + ws://127.0.0.1:8765/events
open http://127.0.0.1:8099/?v=2          # the ?v= matters — see "the cache trap" below
```

The stub builds its payloads with the real producers (`safety/disclosure.browser_event()`
and `speech/voices.started_payload()`), never a hand-written copy, so a change in payload
shape breaks the walk-through instead of being papered over by it. Stop it by pid.

## Measured computed styles

Read with `getComputedStyle` after the WebSocket delivered all three messages.

| element | property | computed value |
|---|---|---|
| `body` | `background-color` | `rgb(16, 20, 24)` |
| `#disclosure` | `background-color` | `rgb(43, 33, 19)` |
| `#disclosure` | `color` | `rgb(255, 212, 121)` |
| `#voice` | `background-color` | `rgb(23, 29, 36)` |
| `#voice` | `color` | `rgb(159, 179, 200)` |

All three pairs differ, and both rows differ from the page background — the T-E3 pass
condition. Rendered text, read off the screenshot:

* `#disclosure` — `AI 系統 · 合成語音 / AI system · synthetic voice` (static, present before
  any message arrives; visible with the gateway down, which is the case that matters).
* `#voice` — `合成語音 / synthetic voice — kokoro-82m/zh (zf_xiaoxiao)`, i.e. the Art 50(2)
  marking as produced by `speech/voices.py`, not a placeholder.
* `#events` — the Art 50(1) sentence, both languages, timestamped.

Screenshot: `t-e3-browser-disclosure.png` (viewport, 44,807 bytes).

## Two findings the numbers alone would not have produced

**1. Transparent is not a colour.** As a negative control the `#disclosure` rule was
deleted. The browser then reported `background-color: rgba(0, 0, 0, 0)`, which is *not
equal* to `rgb(16, 20, 24)` as a string while rendering as exactly the page background.
The first visibility predicate compared strings and therefore called the mutant
**visible**. Both predicates were fixed: the browser-side one now walks to the first
ancestor with alpha > 0 before comparing, and the CI-side `_rgba()` in
`tests/test_disclosure.py` normalises 3/4/6/8-digit hex and rejects alpha `00`. The named
mutant `test_mutant_a_transparent_badge_background_is_caught` (`background: #10141800`)
pins it. This defect was one step away from becoming evidence.

**2. `#hud` painted over the Art 50(2) marking.** With no camera stream the `<img>`
collapsed to its alt text, `#stage` collapsed with it, and `#hud` — absolutely positioned
against `#stage` — landed on top of the rows above. The screenshot showed the engine id
cut in half; every computed colour was still correct. Hit test: 25 sample points along
each row, counting those whose `elementFromPoint` is `#hud` or a descendant.

| | `#disclosure` occluded | `#voice` occluded |
|---|---|---|
| before | 1 / 25 | 5 / 25 |
| after `#stage { min-height: 148px }` + `#disclosure, #voice { position: relative; z-index: 2 }` | **0 / 25** | **0 / 25** |

After the fix, `#stage` computes `min-height: 148px` and `#hud` sits at y = 200, below both
rows. Two mutants pin the two declarations
(`test_mutant_dropping_the_stacking_order_is_caught`,
`test_mutant_a_collapsible_stage_is_caught`).

## The cache trap

The first re-measurement after fixing the occlusion returned **identical** numbers — the
reload served the previous stylesheet from cache. Loading `?v=2` gave the true reading. A
cached page is not the edited file, and a false measurement was two minutes from being
written down here as a pass.

## Honest boundary

* This settles that the Art 50(1) notice and the Art 50(2) marking **render legibly and
  un-occluded in one browser at one viewport**. It is not a cross-browser or responsive
  claim, and nothing here measures whether a person *notices* the badge — §6.4 item 1 is
  the audible half and stays a human sign-off.
* The Art 50(2) row displays a **stream-level** label. It is not a watermark surviving
  re-recording; see the boundary recorded in `speech/voices.py`.
* The stub is not the gateway. It proves the browser handles the payloads the real
  producers emit; that the gateway emits them is asserted structurally in
  `tests/test_voices.py`.
