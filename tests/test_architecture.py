"""The two drawings, checked against the code they draw (§6.2 item T-M1..T-M9).

`docs/architecture.svg` draws the standing shape of the ROS 2 graph and
`docs/workflow.md` draws the sequences through it. Both are hand-authored claims about
a graph that changes, and a drawing nobody can execute drifts silently — the way it
drifts is by staying beautiful. So every claim either side makes is derived here from
`infra/ros_census.py`, which executes the runner for its node inventory and walks the
AST of every launched file for its topics.

Three properties of this module are load-bearing:

**Both directions, always.** Every censused topic must appear in the diagrams *and*
every topic-shaped token in the diagrams must be a censused topic. One direction alone
lets a renamed topic sit in a picture forever: the old name is still drawn, and nothing
notices the new one is missing.

**The checks are pure functions over text, so the mutants can be too.** Every finder
below takes the document (or the SVG) as a string and returns a list of findings. A
mutant is then a `.replace()` on that string rather than a copy of the tree, which is
what makes the mutation pass fast enough to actually run — and what lets a mutant name
the exact defect it reintroduces.

**The pixels are not asserted here, they are cited.** No browser runs in this suite.
`docs/evidence/mermaid-render.json` holds the render measurements, and this module
asserts that record is internally consistent, complete, and within its stated ceiling.
The reason there is a record at all is that all seven blocks of the first draft parsed
*and* rendered without error while section 1 was 3141 px wide with a decision subtree
laid outside the subgraph containing its siblings. A chart that parses can still be
unreadable, so parsing is not the oracle — the width is, and a human looking at the
PNG is what found the rest.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "infra"))

import arch_render  # noqa: E402
from ros_census import Topic, census  # noqa: E402

WORKFLOW = "docs/workflow.md"
SVG = "docs/architecture.svg"
GAPS = "infra/graph_gaps.tsv"
RENDER_EVIDENCE = "docs/evidence/mermaid-render.json"
ANIMATION_EVIDENCE = "docs/evidence/architecture-animation.json"

# The closed class set from graph_gaps.tsv's own header. Spelled here so that adding a
# class to the manifest without deciding what it means turns this red.
GAP_CLASSES = {"external", "pending", "defect", "unused"}

# Mermaid diagram kinds this repo allows. A kind we have never rendered is not
# forbidden — it just has to be added deliberately, because "it parsed" has already
# been shown not to mean "it is readable".
DIAGRAM_KINDS = {"flowchart TD", "flowchart LR", "sequenceDiagram", "stateDiagram-v2"}

# GitHub's file-view content column is about 880 px wide, and the scale it applies to
# an over-wide diagram applies to the label text with it: at 3141 px, a 14 px label
# renders around 4 px. 1500 px scales to about 0.59x, which keeps that label above 8 px.
WIDTH_CEILING_PX = 1500

# A topic-shaped token: two or more lowercase segments. Written once, because the
# document side and the census side must agree on what counts as a topic name.
#
# The boundaries are the whole difficulty. Without them this pattern reads
# `docs/evidence/mermaid-render.json` as the topic `/evidence/mermaid` and reports a
# topic the census does not find — a false finding, which erodes the check faster than
# a missing one, because the way people silence it is by loosening it.
TOPIC_TOKEN = re.compile(r"(?<![\w/.\-])/[a-z0-9_]+(?:/[a-z0-9_]+)+(?![\w.\-])")

# A repo path the document names. Hyphens are in the class deliberately:
# `web/robot-view/index.html` exists, and a class without `-` reports it missing.
REPO_PATH = re.compile(r"\b[a-z_][a-z_0-9]*(?:/[a-z_0-9.\-]+)+\.(?:py|c|h|html|json|jsonl|tsv|md|sh|svg)\b")


# ───────────────────────────────── fixtures ─────────────────────────────────
@pytest.fixture(scope="module")
def graph(repo_root: Path):
    """(nodes, edges, topics) — the census, run once for the whole module."""
    return census(repo_root)


@pytest.fixture(scope="module")
def doc(repo_root: Path) -> str:
    return (repo_root / WORKFLOW).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def svg(repo_root: Path) -> str:
    return (repo_root / SVG).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def gaps(repo_root: Path) -> dict[str, dict[str, str]]:
    return arch_render.read_gaps(repo_root)


# ───────────────────────── the finders, as pure functions ─────────────────────────
def mermaid_blocks(doc: str) -> list[str]:
    return re.findall(r"```mermaid\n(.*?)```", doc, re.S)


def fence_findings(doc: str) -> list[str]:
    """An odd fence count means a block bleeds into prose and renders as code."""
    fences = doc.count("```")
    blocks = len(mermaid_blocks(doc))
    out = []
    if fences % 2:
        out.append(f"{fences} code fences — an odd count leaves a block unterminated")
    if fences != 2 * blocks:
        out.append(f"{fences} fences but {blocks} mermaid blocks — "
                   f"{fences - 2 * blocks} fence(s) belong to something else")
    if not blocks:
        out.append("0 mermaid blocks in a workflow chart document")
    return out


def kind_findings(doc: str) -> list[str]:
    out = []
    for i, b in enumerate(mermaid_blocks(doc)):
        head = b.strip().splitlines()[0].strip()
        if head not in DIAGRAM_KINDS:
            out.append(f"block {i} declares {head!r}, which is not in {sorted(DIAGRAM_KINDS)}")
    return out


def undeclared_id_findings(doc: str) -> list[str]:
    """An id used in an edge but never given a label *in the same block*.

    Mermaid does not report this. It draws a box whose visible text is the bare id —
    which is exactly what happened when section 4's disk chart referred to `memory`,
    a node declared in the chart above it. It rendered, it parsed, and it was wrong.
    """
    declared_re = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s*[\[\{\(]")
    out: list[str] = []
    for i, b in enumerate(mermaid_blocks(doc)):
        if not b.strip().startswith("flowchart"):
            continue
        declared = set(declared_re.findall(b))
        stripped = re.sub(r'"[^"]*"', "", re.sub(r"\|[^|]*\|", "|", b))
        used: set[str] = set()
        for line in stripped.splitlines()[1:]:
            line = line.strip()
            if not line or line.startswith(("subgraph", "end", "%%", "style", "classDef", "click")):
                continue
            for tok in re.split(r"-{2,}>|-\.->|={2,}>|\|", line):
                tok = tok.strip().strip("[]{}()")
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", tok):
                    used.add(tok)
        for name in sorted(used - declared):
            out.append(f"block {i} uses node id {name!r} without a label in that block "
                       f"— mermaid draws a box reading {name!r} and reports nothing")
    return out


def coverage_findings(text: str, nodes: dict[str, str], topics: dict[str, Topic],
                      *, where: str) -> list[str]:
    """Both directions between one drawing and the census."""
    out = []
    for t in sorted(topics):
        if t not in text:
            out.append(f"{where} never names censused topic {t}")
    for tok in sorted(set(TOPIC_TOKEN.findall(text))):
        if tok.endswith(".py"):
            continue
        if tok not in topics:
            out.append(f"{where} names {tok}, which the census does not find")
    for path in sorted(set(nodes.values())):
        if path not in text:
            out.append(f"{where} never names deployed node file {path}")
    return out


def path_findings(text: str, root: Path) -> list[str]:
    """Every repo path a drawing names has to exist."""
    return [f"names {p}, which does not exist"
            for p in sorted(set(REPO_PATH.findall(text)))
            if not (root / p).exists()]


def count_claim_findings(doc: str) -> list[str]:
    """This document must quote no totals.

    `infra/doc_counts.py` guards the totals in docs/test-design.md and both READMEs.
    A fourth site it does not read would be a number nothing checks, so the document
    says "every topic the census finds" instead of "all 21 topics".
    """
    pattern = re.compile(r"\b\d+\s+(?:topics|nodes|edges|blocks|cases|mutants|checks)\b")
    return [f"quotes an unguarded total: {m!r}" for m in pattern.findall(doc)]


def gaps_findings(gaps: dict[str, dict[str, str]], topics: dict[str, Topic]) -> list[str]:
    """The manifest, both directions, plus the closed class set.

    A stale row is the finding that matters: it would silently excuse the next topic
    arriving with that name, and it is the row that must be deleted in the same PR
    that wires the missing end.
    """
    out = []
    asym = {n: ("dead_end" if t.dead_end else "orphan")
            for n, t in topics.items() if t.dead_end or t.orphan}
    for name in sorted(asym):
        if name not in gaps:
            out.append(f"{name} is one-ended in the census and has no manifest row")
    for name, row in sorted(gaps.items()):
        if name not in topics:
            out.append(f"manifest row {name} names a topic the census does not find")
        elif name not in asym:
            out.append(f"manifest row {name} is stale — the topic has both ends wired now")
        elif row["asymmetry"] != asym[name]:
            out.append(f"{name} is a {asym[name]} in the census, "
                       f"{row['asymmetry']} in the manifest")
        if row["class"] not in GAP_CLASSES:
            out.append(f"{name} has class {row['class']!r}, outside {sorted(GAP_CLASSES)}")
        if len(row["reason"].split()) < 8:
            out.append(f"{name}'s reason is {len(row['reason'].split())} words — "
                       f"an exemption without an argument")
    return out


def animation_findings(svg: str) -> list[str]:
    """The four defects the architecture render actually had or nearly had, as a checker.

    Each one was measured in a browser, not reasoned about:

    * the *argument* to the support query has to be a path a browser accepts. The
      idiomatic feature-detection spelling `path("")` is **rejected by Chrome 151**:
      `CSS.supports('offset-path', 'path("")')` is false there while
      `path("M 0 0 L 1 1")` is true. An empty-path gate therefore closes on a browser
      that fully supports the feature, every dot stays at `opacity: 0`, and every check
      in this file still passes — so the gate's argument is checked, not just its
      presence. Measured against the pushed bytes in Chrome 151 on 2026-08-25.

    * `@keyframes` nested inside `@supports` parses, lists in `cssRules`, computes an
      `animation-name`, and does not animate (Chrome 141). Measured: 0 of 14 dots moved
      over 5.6 s. So the frames must be at top level and only the name-assigning rules
      inside the support query.
    * without a base `opacity: 0` outside the query, a renderer with no `offset-path`
      shows 14 dots stacked at the canvas origin instead of a static diagram.
    * `cx`/`cy` on an animated circle is *added* to the offset path point, because the
      transform anchors on the user-space origin and not the circle's centre. Measured:
      a dot at cx=580/cy=479 read getBBox (575,474) and getBoundingClientRect (1154,952).
    """
    out = []
    if "@supports (offset-path" not in svg:
        out.append("no @supports gate on offset-path — an older renderer gets 14 dots "
                   "parked at the origin instead of a complete static diagram")
    for m in re.finditer(r"@supports\s*\(\s*offset-path\s*:\s*([^{]*?)\)\s*\)", svg):
        arg = (m.group(1) + ")").strip()
        inner = re.fullmatch(r"path\(\s*(['\"])(.*?)\1\s*\)", arg)
        if not inner or not inner.group(2).strip().upper().startswith("M"):
            out.append("the @supports argument is not a path a browser accepts "
                       f"({arg!r}); Chrome 151 rejects path(\"\"), so the gate closes on "
                       "a browser that supports offset-path and nothing ever moves")
    for m in re.finditer(r"@supports[^{]*\{", svg):
        depth, i = 1, m.end()
        while depth and i < len(svg):
            depth += (svg[i] == "{") - (svg[i] == "}")
            i += 1
        if "@keyframes" in svg[m.end():i]:
            out.append("@keyframes is nested inside @supports — it parses, it lists in "
                       "cssRules, it computes an animation-name, and nothing moves")
    # The base state must be at CSS top level. There is a second `.pkt, .dead
    # { opacity: 0 }` inside the reduced-motion query, so a search that ignores nesting
    # is satisfied by the copy that only applies when motion is already off — the exact
    # case the base state exists to cover.
    if not any(svg[:m.start()].count("{") == svg[:m.start()].count("}")
               for m in re.finditer(r"\.pkt\s*,\s*\.dead\s*\{[^}]*opacity:\s*0", svg)):
        out.append("no top-level opacity:0 for .pkt/.dead — the only copy is inside a "
                   "query, so a renderer without offset-path still shows the dots")
    if "prefers-reduced-motion" not in svg:
        out.append("no prefers-reduced-motion query — motion with no way to stop it")
    for m in re.finditer(r"<circle[^>]*class=\"pkt[^\"]*\"[^>]*>", svg):
        if re.search(r'\bc[xy]=', m.group(0)):
            out.append("an animated circle carries cx/cy, which is added to the offset "
                       f"path point rather than replaced: {m.group(0)[:90]}")
    return out


# ───────────────────────────────── controls ─────────────────────────────────
def test_control_the_census_reads_a_real_graph(graph) -> None:
    """Floors, not counts. A census that reads nothing has the same shape as a clean
    tree, so the empty result must be loud here as well as inside the census."""
    nodes, edges, topics = graph
    assert len(nodes) >= 8, nodes
    assert len(topics) >= 15, sorted(topics)
    assert len(edges) >= 30, len(edges)
    assert all(p.endswith(".py") for p in nodes.values()), nodes


def test_control_the_committed_svg_is_this_generators_output(repo_root: Path, svg: str) -> None:
    """Rebuild and compare, because a stamp with two readings verifies nothing.

    A generated artifact and its generator drift the moment someone edits the file
    instead of the code. Comparing the bytes is the only check that cannot be satisfied
    by a stale artifact that merely looks plausible.
    """
    rebuilt, meta = arch_render.build(repo_root)
    assert meta["findings"] == [], meta["findings"]
    assert rebuilt == svg, (
        f"docs/architecture.svg is {len(svg)} B, the generator now produces "
        f"{len(rebuilt)} B — run infra/arch_render.py")


def test_control_the_real_drawings_have_no_findings(repo_root: Path, doc: str, svg: str,
                                                    graph, gaps) -> None:
    """Asserted before any mutant runs: if this is red, every red below is uninterpretable."""
    nodes, _, topics = graph
    assert fence_findings(doc) == []
    assert kind_findings(doc) == []
    assert undeclared_id_findings(doc) == []
    assert coverage_findings(doc, nodes, topics, where=WORKFLOW) == []
    assert coverage_findings(svg, nodes, topics, where=SVG) == []
    assert path_findings(doc, repo_root) == []
    assert count_claim_findings(doc) == []
    assert gaps_findings(gaps, topics) == []
    assert animation_findings(svg) == []


# ───────────────────────── the drawings against the census ─────────────────────────
def test_every_node_prose_entry_is_a_deployed_node(repo_root: Path, graph) -> None:
    nodes, _, _ = graph
    assert sorted(arch_render.NODE_PROSE) == sorted(nodes)


def test_module_prose_matches_the_derived_module_set(repo_root: Path, graph) -> None:
    nodes, _, _ = graph
    derived = arch_render.module_rows(repo_root, nodes)
    assert sorted(arch_render.MODULE_PROSE) == sorted(derived), (
        "a module with no prose is drawn as an unexplained box, and prose for a module "
        "that no longer exists describes a tree that is gone")


def test_hop_labels_and_hop_paths_are_the_same_length(repo_root: Path) -> None:
    """The numbered list a reduced-motion reader sees and the dots a moving-image
    reader sees are the same eight hops or the diagram tells two stories."""
    _, meta = arch_render.build(repo_root)
    assert len(meta["hop_labels"]) == len(meta["hop_paths"]) >= 8


def test_hop_labels_name_only_censused_topics(graph) -> None:
    _, _, topics = graph
    labels = arch_render.hop_labels(topics)
    for tok in TOPIC_TOKEN.findall(" ".join(labels)):
        assert tok in topics, f"hop label names {tok}, absent from the census"


def test_the_rps_pair_stays_paired(graph, gaps) -> None:
    """One gap counted from both sides. Wiring one end alone would look like progress
    and change nothing, so the manifest must lose both rows together or neither."""
    _, _, topics = graph
    a, b = "/game/rps_round", "/memory/rps_table"
    assert (a in gaps) == (b in gaps), (
        f"{a} in manifest: {a in gaps}, {b}: {b in gaps} — "
        "the round is published to a writer that cannot answer with a table")
    if a in gaps:
        assert topics[a].dead_end and topics[b].orphan
        assert gaps[a]["class"] == gaps[b]["class"] == "pending"


def test_the_workflow_charts_and_the_svg_draw_the_same_topics(doc: str, svg: str,
                                                              graph) -> None:
    """Two readers, one graph: the set of topics each drawing names must be equal.

    Checking each against the census separately leaves the case where one drawing is
    a strict subset — both pass their own check, and the two pictures disagree.
    """
    _, _, topics = graph
    in_doc = {t for t in topics if t in doc}
    in_svg = {t for t in topics if t in svg}
    assert in_doc == in_svg == set(topics), {
        "only in workflow.md": sorted(in_doc - in_svg),
        "only in architecture.svg": sorted(in_svg - in_doc),
        "in neither": sorted(set(topics) - in_doc - in_svg),
    }


# ───────────────────────────── the render evidence ─────────────────────────────
def test_the_render_record_is_complete_and_within_its_ceiling(repo_root: Path,
                                                              doc: str) -> None:
    """The pixels are cited, not re-measured — but the citation has to be checkable.

    No browser runs in this suite, so this asserts the record covers every block in
    the document today, that nothing in it failed, and that its own totals were
    computed from its own rows rather than typed.
    """
    rec = json.loads((repo_root / RENDER_EVIDENCE).read_text(encoding="utf-8"))
    blocks = rec["blocks"]
    assert len(blocks) == len(mermaid_blocks(doc)), (
        f"{len(blocks)} blocks recorded, {len(mermaid_blocks(doc))} in the document — "
        "a record that does not cover the current charts describes an older document")
    assert [b["index"] for b in blocks] == list(range(len(blocks)))
    assert all(b["parsed"] and b["rendered"] and b["error"] is None for b in blocks)
    widths = [b["width_px"] for b in blocks]
    assert max(widths) <= WIDTH_CEILING_PX, dict(zip(range(len(widths)), widths))
    assert rec["totals"]["blocks"] == len(blocks)
    assert rec["totals"]["parsed"] == rec["totals"]["rendered"] == len(blocks)
    assert rec["totals"]["errors"] == 0
    assert rec["totals"]["max_width_px"] == max(widths)
    assert rec["ceiling_px"] == WIDTH_CEILING_PX
    assert blocks[rec["totals"]["max_width_block_index"]]["width_px"] == max(widths)


def test_the_render_record_keeps_the_draft_that_parsed_and_failed(repo_root: Path) -> None:
    """The comparison numbers are the argument for measuring width at all: the first
    draft parsed and rendered with zero errors at 3576 px, and was unusable."""
    rec = json.loads((repo_root / RENDER_EVIDENCE).read_text(encoding="utf-8"))
    draft = rec["first_draft_for_comparison"]
    assert draft["max_width_px"] > WIDTH_CEILING_PX
    assert draft["max_width_px"] == max(draft["widths_px"])
    assert draft["section_1"]["defects_seen_by_eye"]


def test_the_animation_record_quotes_the_gate_this_svg_actually_ships(repo_root: Path,
                                                                     svg: str) -> None:
    """Two readers, one string. The evidence file says which support query was measured
    in a browser; if the generator's gate changes and the record does not, the record
    describes a diagram that is no longer in the repository — and a browser measurement
    of a string nobody ships is worth nothing. The driven-timeline samples are required
    to contain at least one dot that reached a non-zero offset-distance at full opacity,
    because a record of a frozen timeline would otherwise pass as a record of motion."""
    rec = json.loads((repo_root / ANIMATION_EVIDENCE).read_text(encoding="utf-8"))
    gate = rec["the_gate_argument_is_itself_a_claim"]["as_shipped"]
    assert gate in svg, f"the record was measured against {gate!r}, which this svg does not contain"
    assert animation_findings(svg) == []
    samples = rec["the_css_is_effective"]["driven_timeline_samples"]
    assert any(s["offset_distance"] not in ("0%", "100%") and s["opacity"] == "1"
               for s in samples), "no sample shows a dot mid-path and visible"
    assert rec["what_this_pass_could_not_measure"]["still_manual"]


# ────────────────────────── geometry checker discrimination ──────────────────────────
# Seven hand-built layouts, each broken in exactly one way. A checker that has only ever
# seen a clean diagram is indistinguishable from one that returns [].
#   segment: (x1, y1, x2, y2, from_name, to_name)
#   rect:    (x, y, w, h, name)
BROKEN_LAYOUTS = {
    "connectors_cross": (
        [(0, 0, 100, 100, "a", "b"), (0, 100, 100, 0, "c", "d")],
        [(0, 0, 10, 10, "a"), (90, 90, 10, 10, "b"), (0, 90, 10, 10, "c"), (90, 0, 10, 10, "d")],
        200, 200),
    "connector_through_unrelated_box": (
        [(0, 50, 200, 50, "a", "b")],
        [(0, 40, 10, 20, "a"), (190, 40, 10, 20, "b"), (90, 40, 20, 20, "c")],
        300, 200),
    "boxes_overlap": (
        [],
        [(0, 0, 100, 100, "a"), (50, 50, 100, 100, "b")],
        300, 300),
    "box_off_canvas_right": ([], [(280, 10, 100, 40, "a")], 300, 300),
    "box_off_canvas_bottom": ([], [(10, 280, 100, 40, "a")], 300, 300),
    "box_negative_origin": ([], [(-20, 10, 100, 40, "a")], 300, 300),
    "segment_off_canvas": ([(10, 10, 400, 10, "a", "b")],
                           [(0, 0, 20, 20, "a"), (390, 0, 20, 20, "b")], 300, 300),
}


@pytest.mark.parametrize("name", sorted(BROKEN_LAYOUTS))
def test_geometry_checker_finds_each_hand_built_defect(name: str) -> None:
    segs, rects, w, h = BROKEN_LAYOUTS[name]
    found = arch_render.geometry_findings(segs, rects, w, h)
    assert found, f"{name}: the checker reported a broken layout clean"


def test_geometry_checker_passes_a_clean_hand_built_layout() -> None:
    """The no-defect control. Without it, a checker that returns a finding for every
    input would score 7/7 above."""
    segs = [(50, 20, 50, 80, "a", "b")]
    rects = [(20, 0, 60, 20, "a"), (20, 80, 60, 20, "b")]
    assert arch_render.geometry_findings(segs, rects, 200, 200) == []


def test_geometry_checker_finds_a_tightened_copy_of_the_real_layout(repo_root: Path) -> None:
    """Grow every real box by 21 px and the real layout must stop being clean.

    21 px is not a taste: the closest pair of boxes in the shipped layout ('disk' and
    'cloud') sits exactly 20.0 px apart, measured by walking `meta["rects"]`, so 20 px of
    uniform growth makes them touch and 21 px makes them overlap. Both halves are
    asserted — under-growth must stay clean and over-growth must be caught — because a
    checker that finds a defect at every input would pass the second half alone.

    This slack is a different number from the 11.5 px the browser containment pass
    reported: that one is the tightest gap between a text run and its own box, this one
    is the tightest gap between two boxes. Two numbers, two derivations.
    """
    _, meta = arch_render.build(repo_root)

    def grown(px: float):
        return [(x - px / 2, y - px / 2, w + px, h + px, n)
                for (x, y, w, h, n) in meta["rects"]]

    assert arch_render.geometry_findings(
        meta["segments"], grown(20), meta["W"], meta["H"]) == [], (
        "20 px of growth already reports a finding, so the 21 px result below would "
        "prove nothing about the checker's threshold")
    found = arch_render.geometry_findings(
        meta["segments"], grown(21), meta["W"], meta["H"])
    assert found, "21 px of growth on every box changed no verdict"


# ───────────────────────────────── mutants ─────────────────────────────────
# name -> (mutate, the finder it must turn red). The mutants are this repo's real defect
# history: `undeclared_id_in_block` and `keyframes_inside_supports` both shipped into a
# draft, rendered without error, and were caught by looking at the picture.
DOC_MUTANTS = {
    "topic_renamed_in_the_chart": (
        lambda d: d.replace("/speech/asr", "/speech/transcript"),
        lambda d, n, t, r: coverage_findings(d, n, t, where=WORKFLOW)),
    "topic_dropped_from_every_chart": (
        lambda d: d.replace("/speech/tts_done", "a finished-utterance signal"),
        lambda d, n, t, r: coverage_findings(d, n, t, where=WORKFLOW)),
    "node_file_dropped": (
        lambda d: d.replace("memory/memory_node.py", "the memory node"),
        lambda d, n, t, r: coverage_findings(d, n, t, where=WORKFLOW)),
    "path_that_does_not_exist": (
        lambda d: d.replace("memory/recall.py", "memory/retrieval.py"),
        lambda d, n, t, r: path_findings(d, r)),
    "unterminated_fence": (
        lambda d: d.replace("```mermaid\nflowchart TD", "```mermaid\nflowchart TD", 1)
                   .replace("\n```\n", "\n", 1),
        lambda d, n, t, r: fence_findings(d)),
    "undeclared_diagram_kind": (
        lambda d: d.replace("flowchart LR", "graph LR", 1),
        lambda d, n, t, r: kind_findings(d)),
    "undeclared_id_in_block": (
        lambda d: d.replace('memory2["memory_node<br/>the only writer<br/>memory/store.py"]',
                            "memory2", 1),
        lambda d, n, t, r: undeclared_id_findings(d)),
    "unguarded_total_quoted": (
        lambda d: d.replace("every topic the census finds", "all 21 topics", 1),
        lambda d, n, t, r: count_claim_findings(d)),
}


@pytest.mark.mutation
@pytest.mark.parametrize("name", sorted(DOC_MUTANTS))
def test_doc_mutant_is_killed(name: str, doc: str, graph, repo_root: Path) -> None:
    mutate, finder = DOC_MUTANTS[name]
    nodes, _, topics = graph
    broken = mutate(doc)
    assert broken != doc, f"{name}: the mutant anchor is gone, so nothing was mutated"
    assert finder(broken, nodes, topics, repo_root), f"{name}: mutant survived"


@pytest.mark.mutation
def test_doc_control_is_green(doc: str, graph, repo_root: Path) -> None:
    """The unmutated control, asserted in the same pass as the mutants: if the harness
    itself is broken, every "mutant killed" above is a report about nothing."""
    nodes, _, topics = graph
    for _, finder in DOC_MUTANTS.values():
        assert finder(doc, nodes, topics, repo_root) == []


SVG_MUTANTS = {
    "supports_gate_removed": (
        lambda s: s.replace("@supports (offset-path", "@media all and (min-width:0px) /*"),
    ),
    "keyframes_moved_inside_supports": (
        # The exact defect: parses, lists in cssRules, computes animation-name, and
        # 0 of 14 dots move.
        lambda s: re.sub(r"(@supports \(offset-path[^{]*\{)",
                         r"\1 @keyframes hop0 { to { offset-distance: 100% } }", s, count=1),
    ),
    "base_opacity_removed": (
        lambda s: re.sub(r"\.pkt\s*,\s*\.dead\s*\{[^}]*\}", ".pkt, .dead { fill: none }", s,
                         count=1),
    ),
    "reduced_motion_removed": (
        lambda s: s.replace("prefers-reduced-motion", "prefers-color-scheme"),
    ),
    # The gate is still there, still spelled `@supports (offset-path: …)`, and its
    # argument is the one every feature-detection snippet uses. Chrome 151 rejects it,
    # so this mutant is a diagram that is silently static on a current browser.
    "supports_argument_emptied": (
        lambda s: re.sub(r"(@supports \(offset-path:\s*)path\([^)]*\)", r"\1path('')", s,
                         count=1),
    ),
    "cx_cy_back_on_an_animated_circle": (
        lambda s: s.replace('<circle r="5.5"', '<circle cx="580" cy="479" r="5.5"', 1),
    ),
}


@pytest.mark.mutation
@pytest.mark.parametrize("name", sorted(SVG_MUTANTS))
def test_svg_animation_mutant_is_killed(name: str, svg: str) -> None:
    (mutate,) = SVG_MUTANTS[name]
    broken = mutate(svg)
    assert broken != svg, f"{name}: the mutant anchor is gone, so nothing was mutated"
    assert animation_findings(broken), f"{name}: mutant survived"


@pytest.mark.mutation
def test_svg_animation_control_is_green(svg: str) -> None:
    assert animation_findings(svg) == []


GAP_MUTANTS = {
    "row_deleted_for_a_real_dead_end": lambda g: {k: v for k, v in g.items()
                                                  if k != "/speech/tts_done"},
    "stale_row_for_a_wired_topic": lambda g: {**g, "/speech/say": {
        "asymmetry": "dead_end", "class": "pending",
        "reason": "a stale excuse that would cover the next topic of this name silently"}},
    "class_outside_the_closed_set": lambda g: {**g, "/speech/tts_done": {
        **g["/speech/tts_done"], "class": "wontfix"}},
    "asymmetry_direction_flipped": lambda g: {**g, "/memory/rps_table": {
        **g["/memory/rps_table"], "asymmetry": "dead_end"}},
    "reason_reduced_to_a_shrug": lambda g: {**g, "/joint_states": {
        **g["/joint_states"], "reason": "known issue"}},
}


@pytest.mark.mutation
@pytest.mark.parametrize("name", sorted(GAP_MUTANTS))
def test_gap_manifest_mutant_is_killed(name: str, gaps, graph) -> None:
    _, _, topics = graph
    broken = GAP_MUTANTS[name](dict(gaps))
    assert broken != gaps, f"{name}: nothing was mutated"
    assert gaps_findings(broken, topics), f"{name}: mutant survived"


@pytest.mark.mutation
def test_gap_manifest_control_is_green(gaps, graph) -> None:
    _, _, topics = graph
    assert gaps_findings(dict(gaps), topics) == []
