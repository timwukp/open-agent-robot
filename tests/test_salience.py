"""Offline tests for the S3.5 salience formula — T-C6, and the formula-id contract.

**T-C6 needs two halves, and the one the design table asks for is the weaker one.**
The table says: with the prosody switch off, feed various acoustic inputs and the
score must be bit-identical. That assertion is satisfied by a system that ignores
prosody *and equally well* by a system that never reads those fields for any reason
at all — a misspelled field name in this file, an input shape the real node never
sends, a grid whose values all collapse to the same number. Each of those makes the
test green while proving nothing. So the switch-*on* run is not a bonus, it is the
positive control: the same grid must move the score when the term is enabled, which
is what establishes that there was something there to ignore. The grid's keys come
from `salience.ACOUSTIC_FIELDS` rather than from this file, so a rename cannot
quietly empty it either.

That is also why `prosody_term` refuses out-of-range values instead of clamping
them: a clamping version fed realistic pitch in Hz returns 1.0 for every input, so
the positive control would fail while the term was genuinely reading its input. The
refusal keeps the failure attributable — a bad unit is a bad unit, not a dead term.

**Why a `salience` field with no `salience_formula` is refused by the store.**
The value of settling the formula before S4 is not that the code is hard to change
later; it is that stored numbers are only comparable to numbers from the same
formula. If S6 reweights the terms and nothing recorded which formula produced which
value, "these two records disagree" and "these two records are not the same
quantity" become indistinguishable — and unrecoverably so, because the missing
information was never written. `store.RECORD_FIELD_DEPS` refuses the record, which is
the last moment at which it is fixable. The store declares the *pairing* as data and
`memory/salience.py` owns the *id*: two copies of the id would be the defect the
pairing exists to prevent, so there is a one-creator assertion below with a positive
control (a scanner that cannot find a planted second copy is not evidence of one
copy).
"""
from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SALIENCE_PY = REPO_ROOT / "memory" / "salience.py"
STORE_PY = REPO_ROOT / "memory" / "store.py"

from memory import salience as sal  # noqa: E402
from memory import store as st  # noqa: E402

USER = {"role": "user", "text": "我喜歡芫茜,記住"}


def make_store(tmp_path: Path, module=st, **kw):
    """Same visible exemption as tests/test_store.py: tmp_path is ephemeral."""
    return module.Store(tmp_path / "data", _allow_ephemeral=True, **kw)


# --------------------------------------------------------------------- mutants
def load_mutant(tmp_path: Path, target: str, *replacements: tuple[str, str]):
    """Import a text-patched copy of memory/salience.py or memory/store.py.

    Zero replacements gives the no-mutant control required by §0 rule 2. The patched
    copy is imported by path, so it keeps importing the *real* `memory.store` — which
    is what makes the composed mutant below meaningful: a `stamp` that forgets the
    formula id has to be caught by the genuine store, not by a patched one.
    """
    dst = tmp_path / f"mut_{target}_{len(replacements)}"
    dst.mkdir(parents=True, exist_ok=True)
    src = (REPO_ROOT / "memory" / f"{target}.py").read_text()
    for old, new in replacements:
        assert old in src, f"mutant anchor no longer present in {target}.py: {old!r}"
        assert src.count(old) == 1, f"mutant anchor is not unique: {old!r}"
        src = src.replace(old, new)
    out = dst / f"{target}.py"
    out.write_text(src)
    name = f"{target}_smutant_{tmp_path.name}_{len(replacements)}"
    spec = importlib.util.spec_from_file_location(name, out)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


PROJECTION = ('    return {k: turn[k] for k in TEXT_FIELDS if k in turn}\n')
PAIRING = ("        for field, companion in sorted(RECORD_FIELD_DEPS.items()):\n"
           "            if field in record and companion not in record:\n")


# ------------------------------------------------------- the acoustic grid (T-C6)
def acoustic_grid() -> list[dict]:
    """Turns identical in text and differing only in the declared acoustic fields.

    Two shapes, because they fail differently: moving every field together catches a
    term that reads the mean, and moving one field at a time catches a term that
    reads a single field. Values are normalised features in [0, 1] — see the module
    docstring on why raw Hz would make even a live term look constant.
    """
    base = dict(USER)
    grid = []
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        grid.append({**base, **{f: v for f in sal.ACOUSTIC_FIELDS}})
    for field in sal.ACOUSTIC_FIELDS:
        for v in (0.0, 1.0):
            turn = {**base, **{f: 0.5 for f in sal.ACOUSTIC_FIELDS}}
            turn[field] = v
            grid.append(turn)
    return grid


def test_the_acoustic_grid_is_derived_from_the_declared_field_names():
    """§0 rule 5: a grid this test wrote down could not notice a renamed field."""
    grid = acoustic_grid()
    assert len(grid) >= 12, f"the grid collapsed to {len(grid)} turns"
    for turn in grid:
        assert set(sal.ACOUSTIC_FIELDS) <= set(turn), \
            "a grid turn is missing a declared acoustic field"
    varied = {f for f in sal.ACOUSTIC_FIELDS
              if len({turn[f] for turn in grid}) > 1}
    assert varied == set(sal.ACOUSTIC_FIELDS), \
        f"these fields never vary across the grid, so nothing tests them: " \
        f"{sorted(set(sal.ACOUSTIC_FIELDS) - varied)}"


def test_with_the_switch_off_no_acoustic_field_moves_the_score():
    """T-C6 proper: the scores are bit-identical, not merely close."""
    scores = [sal.score(turn, prosody=False) for turn in acoustic_grid()]
    assert len(set(scores)) == 1, \
        f"prosody reached the score with the switch off: {sorted(set(scores))}"


def test_with_the_switch_on_the_same_grid_does_move_the_score():
    """The positive control. Without it the test above cannot fail for the right reason."""
    scores = [sal.score(turn, prosody=True) for turn in acoustic_grid()]
    assert len(set(scores)) >= 5, \
        f"the grid is unreadable even with the term enabled: {sorted(set(scores))}"


def test_the_switch_off_score_ignores_the_fields_entirely():
    """A turn with no acoustic fields at all scores the same as one drenched in them."""
    bare = sal.score(USER, prosody=False)
    for turn in acoustic_grid():
        assert sal.score(turn, prosody=False) == bare


def test_the_projection_admits_exactly_the_text_fields():
    turn = {**USER, **{f: 0.5 for f in sal.ACOUSTIC_FIELDS}, "unknown": 1}
    assert sal.text_only(turn) == {"role": "user", "text": USER["text"]}


def test_the_shipped_default_is_the_switch_off():
    """Read the literal in the pushed source, not the value this process imported.

    A test that asserts `sal.PROSODY_ENABLED is False` after importing the module is
    checking the same statement twice. What matters is what a fresh clone does, so the
    assertion is against the file's text — the same reason the push guard checks the
    real `.push/files.txt`.
    """
    assert "\nPROSODY_ENABLED = False\n" in SALIENCE_PY.read_text()
    assert sal.PROSODY_ENABLED is False
    assert sal.score(USER) == sal.score(USER, prosody=False)
    assert sal.formula_id() == sal.FORMULA


def test_the_formula_id_changes_with_the_prosody_term():
    """Enabling biometric processing must be visible in the data, not only in a config."""
    assert sal.formula_id(False) != sal.formula_id(True)
    assert sal.formula_id(True) == sal.FORMULA_PROSODY


# ---------------------------------------------------------------- the four terms
def test_the_declared_weights_have_no_findings():
    assert sal.weight_findings() == []
    assert math.fsum(w for _, w in sal.WEIGHTS) == 1.0


def test_the_weight_names_are_exactly_the_signal_names():
    assert {n for n, _ in sal.WEIGHTS} == set(sal.signals(USER))


@pytest.mark.parametrize("text", [
    "我喜歡芫茜", "我討厭芫茜", "唔好唔記得我唔食牛", "remember this: I hate coriander",
    "今天天氣不錯", "a" * 400, "我對花生過敏", "你喜歡芫茜嗎", "你記得我嗎",
])
def test_a_score_is_in_the_unit_interval(text):
    value = sal.score({"role": "user", "text": text}, seen=["今天天氣不錯"])
    assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("text", ["我喜歡芫茜", "a" * 4000, "唔好唔記得"])
def test_every_signal_is_in_the_unit_interval(text):
    for name, value in sal.signals({"role": "user", "text": text}).items():
        assert 0.0 <= value <= 1.0, f"{name} escaped the unit interval: {value}"


def test_an_explicit_evaluation_scores_regardless_of_polarity():
    """Statement-hood, not sentiment: 「我討厭X」 is as worth remembering as 「我最愛X」."""
    liked = sal.signals({"role": "user", "text": "我最愛芫茜"})["explicit"]
    hated = sal.signals({"role": "user", "text": "我討厭芫茜"})["explicit"]
    assert liked == hated == 1.0


def test_a_question_about_the_robot_is_not_the_person_evaluating():
    assert sal.signals({"role": "user", "text": "你喜歡芫茜嗎"})["explicit"] == 0.0


@pytest.mark.parametrize("text", [
    "記住我住在新加坡", "唔好唔記得我唔食牛", "please remember my name is Tim",
    "别忘了我的生日", "keep in mind I finish after 19:00",
])
def test_asking_to_remember_scores_in_three_languages(text):
    assert sal.signals({"role": "user", "text": text})["asked"] == 1.0


def test_a_question_about_the_robots_memory_is_not_a_request():
    """Bare 「記得」 is deliberately absent from the lexicon — see the module comment."""
    assert sal.signals({"role": "user", "text": "你記得我嗎"})["asked"] == 0.0


def test_the_first_turn_is_maximally_novel():
    assert sal.novelty("我喜歡芫茜", []) == 1.0


def test_a_restatement_is_not_novel():
    assert sal.novelty("我喜歡芫茜", ["我喜歡芫茜"]) == 0.0


def test_novelty_ignores_whitespace_differences():
    assert sal.novelty("我 好 累", ["我好累"]) == 0.0


def test_novelty_takes_the_closest_match_not_the_last_one():
    """A restatement buried among unrelated turns is still a restatement."""
    seen = ["今天天氣不錯", "我喜歡芫茜", "明天要下雨"]
    assert sal.novelty("我喜歡芫茜", seen) == 0.0


def test_the_length_term_saturates():
    long = sal.signals({"role": "user", "text": "字" * 10_000})["length"]
    at = sal.signals({"role": "user", "text": "字" * sal.LENGTH_SATURATION})["length"]
    assert long == at == 1.0


def test_the_length_term_is_monotone():
    values = [sal.signals({"role": "user", "text": "字" * n})["length"]
              for n in (1, 10, 60, 119, 120, 500)]
    assert values == sorted(values)
    assert values[0] < values[-1]


def test_a_robot_turn_is_refused():
    """The robot saying "I like that" is not the person stating a preference."""
    with pytest.raises(sal.SalienceError, match="only role that is the person"):
        sal.score({"role": "robot", "text": "我喜歡芫茜"})


@pytest.mark.parametrize("text", ["", "   ", None, 7])
def test_a_turn_with_no_text_is_refused(text):
    with pytest.raises(sal.SalienceError, match="non-empty text"):
        sal.score({"role": "user", "text": text})


def test_the_prosody_term_refuses_a_missing_measurement():
    with pytest.raises(sal.SalienceError, match="not a measurement of zero"):
        sal.score(USER, prosody=True)


def test_the_prosody_term_refuses_raw_units():
    """Realistic pitch in Hz is the input that would make a clamping term constant."""
    turn = {**USER, **{f: 0.5 for f in sal.ACOUSTIC_FIELDS}, "pitch_hz_mean": 220.0}
    with pytest.raises(sal.SalienceError, match="not a normalised feature"):
        sal.score(turn, prosody=True)


def test_term_order_is_not_part_of_the_formula():
    """`fsum` rather than a running total, so a rebuild cannot differ by an ulp."""
    parts = sal.signals(USER, seen=["今天天氣不錯"])
    forward = math.fsum(parts[n] * w for n, w in sal.WEIGHTS)
    backward = math.fsum(parts[n] * w for n, w in reversed(sal.WEIGHTS))
    assert forward == backward == sal.score(USER, seen=["今天天氣不錯"])


def test_scoring_the_same_turn_twice_is_bit_identical():
    seen = ["今天天氣不錯", "我喜歡芫茜"]
    assert sal.score(USER, seen) == sal.score(USER, seen)


# ------------------------------------------------- the stamp and the store pairing
def test_stamp_attaches_both_the_score_and_its_formula():
    out = sal.stamp({"kind": "turn"}, USER)
    assert out["salience"] == sal.score(USER)
    assert out["salience_formula"] == sal.FORMULA
    assert out["kind"] == "turn"


def test_stamp_does_not_mutate_the_caller_record():
    record = {"kind": "turn"}
    sal.stamp(record, USER)
    assert record == {"kind": "turn"}


def test_the_store_accepts_a_stamped_record(tmp_path):
    with make_store(tmp_path) as store:
        store.append("events", sal.stamp({"kind": "turn"}, USER))
        store.flush()
        (row,) = store.read("events")
    assert row["salience_formula"] == sal.FORMULA


def test_the_store_refuses_a_score_with_no_formula_id(tmp_path):
    with make_store(tmp_path) as store:
        with pytest.raises(st.StoreError, match="salience_formula"):
            store.append("events", {"kind": "turn", "salience": 0.5})
        store.flush()
        assert store.read("events") == []


def test_the_refusal_names_the_one_way_to_attach_a_score(tmp_path):
    """A refusal that does not say what to do instead is a refusal people work around."""
    with make_store(tmp_path) as store:
        with pytest.raises(st.StoreError, match=r"memory\.salience\.stamp"):
            store.append("events", {"salience": 0.5})


def test_the_pairing_refusal_precedes_the_dead_writer_check(tmp_path):
    """Same ordering claim as the snapshot lock: content, not disk, must be reported.

    `StoreError` covers both outcomes, so the discriminator is the message. A test
    that only asserted "it raises" would be satisfied by the reordered version.
    """
    with make_store(tmp_path) as store:
        store._failure = RuntimeError("disk gone")
        with pytest.raises(st.StoreError, match="salience_formula"):
            store.append("events", {"salience": 0.5})
        with pytest.raises(st.StoreError, match="writer thread died"):
            store.append("events", {"kind": "turn"})
        store._failure = None


def test_a_record_carrying_both_fields_is_untouched_by_the_rule(tmp_path):
    """The rule must not fire on a record that already complies, nor on one with neither."""
    with make_store(tmp_path) as store:
        store.append("events", {"salience": 0.1, "salience_formula": "x"})
        store.append("events", {"kind": "turn"})
        store.flush()
        assert len(store.read("events")) == 2


# --------------------------------------------------------- one creator of the id
def formula_assignments(root: Path) -> list[str]:
    """Every module in `root` that assigns a name starting with `FORMULA`."""
    out = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith("FORMULA"):
                        out.append(str(path.relative_to(root)))
    return sorted(set(out))


def test_the_scanner_can_find_a_planted_formula_id(tmp_path):
    """Positive control: a scan that finds nothing is only evidence if it can find something."""
    (tmp_path / "planted.py").write_text('FORMULA = "somewhere-else"\n')
    assert formula_assignments(tmp_path) == ["planted.py"]


def test_there_is_exactly_one_creator_of_the_formula_id():
    assert formula_assignments(REPO_ROOT / "memory") == ["salience.py"]


def test_the_store_does_not_import_the_salience_module():
    """The pairing is data in the store; the id belongs to the module that computes it.

    A store that imported `memory.salience` would acquire a dependency on every module
    that derives a field, and `memory/salience.py` already imports the store — the
    cycle is the observable form of the same mistake.
    """
    tree = ast.parse(STORE_PY.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "salience" not in (node.module or "")
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "salience" not in alias.name


def test_the_module_needs_nothing_that_only_exists_on_the_instance():
    """§1: a pure module, so T-C6 runs under pytest rather than on the g4dn."""
    forbidden = {"rclpy", "boto3", "torch", "numpy", "librosa", "soundfile",
                 "sounddevice", "webrtcvad", "socket", "requests"}
    tree = ast.parse(SALIENCE_PY.read_text())
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        assert not (set(names) & forbidden), f"{names} does not run on a laptop"


# ------------------------------------------------------------------ L3: mutants
@pytest.mark.mutation
def test_control_the_unmutated_module_passes_every_gate(tmp_path):
    """§0 rule 2: without this, a broken harness reports every mutant as killed."""
    m = load_mutant(tmp_path, "salience")
    assert m.weight_findings() == []
    assert m.PROSODY_ENABLED is False
    assert len({m.score(t, prosody=False) for t in acoustic_grid()}) == 1
    assert len({m.score(t, prosody=True) for t in acoustic_grid()}) >= 5
    assert m.signals({"role": "user", "text": "我討厭芫茜"})["explicit"] == 1.0
    assert m.signals({"role": "user", "text": "你記得我嗎"})["asked"] == 0.0
    assert m.novelty("我喜歡芫茜", []) == 1.0
    assert m.novelty("我喜歡芫茜", ["我喜歡芫茜"]) == 0.0
    assert m.signals({"role": "user", "text": "字" * 10_000})["length"] == 1.0
    assert m.stamp({}, USER)["salience_formula"] == m.FORMULA
    with pytest.raises(st.StoreError):
        m.score({"role": "robot", "text": "我喜歡芫茜"})


@pytest.mark.mutation
def test_mutant_the_projection_lets_prosody_through(tmp_path):
    """The one T-C6 exists for: the switch is off and the acoustics reach the score."""
    m = load_mutant(tmp_path, "salience", (
        PROJECTION,
        '    out = {k: turn[k] for k in TEXT_FIELDS if k in turn}\n'
        '    for f in ACOUSTIC_FIELDS:\n'
        '        if f in turn:\n'
        '            out[f] = turn[f]\n'
        '    return out\n',
    ), (
        '    terms = [parts[name] * weight * scale for name, weight in WEIGHTS]\n',
        '    terms = [parts[name] * weight * scale for name, weight in WEIGHTS]\n'
        '    terms.append(float(text_only(turn).get("energy_rms", 0.0)) * 0.1)\n',
    ))
    scores = {m.score(turn, prosody=False) for turn in acoustic_grid()}
    assert len(scores) > 1, "the mutant survived, so the switch-off assertion is vacuous"


@pytest.mark.mutation
def test_mutant_the_weights_no_longer_sum_to_one(tmp_path):
    """The observable damage is a *collapse*, not an out-of-range value.

    `score` clamps, so an over-weighted formula does not produce 1.1 — it produces
    1.0 for every turn whose terms sum past the top, and two turns that the real
    formula ranks differently become the same number. Asserting "> 1.0" would
    therefore fail against the very mutant it was written for; `weight_findings()` is
    what names the cause, and the collapse is what makes it observable in the data.
    """
    m = load_mutant(tmp_path, "salience",
                    ('    ("explicit", 0.4),', '    ("explicit", 0.5),'))
    assert m.weight_findings(), "a score outside [0, 1] would be reported as fine"
    full = {"role": "user", "text": "我喜歡記住" + "字" * 200}   # all four terms at 1.0
    part = {"role": "user", "text": "我喜歡記住" + "字" * 55}    # length term at 0.5
    assert sal.score(full) != sal.score(part)
    assert m.score(full) == m.score(part) == 1.0


@pytest.mark.mutation
def test_mutant_the_length_term_stops_saturating(tmp_path):
    m = load_mutant(tmp_path, "salience",
                    ('        "length": _clamp(len(text) / LENGTH_SATURATION),',
                     '        "length": len(text) / LENGTH_SATURATION,'))
    assert m.signals({"role": "user", "text": "字" * 10_000})["length"] > 1.0


@pytest.mark.mutation
def test_mutant_novelty_is_inverted(tmp_path):
    m = load_mutant(tmp_path, "salience",
                    ("    return _clamp(1.0 - best)", "    return _clamp(best)"))
    assert m.novelty("我喜歡芫茜", ["我喜歡芫茜"]) == 1.0, \
        "a verbatim restatement scored as maximally novel and nothing noticed"


@pytest.mark.mutation
def test_mutant_an_empty_history_makes_everything_unnovel(tmp_path):
    """The branch a rewrite is most likely to get backwards: nothing seen yet."""
    m = load_mutant(tmp_path, "salience",
                    ("    best = 0.0\n", "    best = 1.0\n"))
    assert m.novelty("我喜歡芫茜", []) == 0.0


@pytest.mark.mutation
def test_mutant_the_explicit_term_becomes_a_polarity_judgement(tmp_path):
    """Keeping only the positive markers turns statement-hood into sentiment."""
    m = load_mutant(tmp_path, "salience",
                    ('    "我最愛", "我最爱", "我最怕", "我受不了", "我不吃", "我不能吃", "我怕",\n',
                     '    "我最愛", "我最爱",\n'))
    src = (REPO_ROOT / "memory" / "salience.py").read_text()
    assert "我討厭" in src  # the marker the mutant keeps; the dropped ones are the point
    assert m.signals({"role": "user", "text": "我受不了芫茜"})["explicit"] == 0.0


@pytest.mark.mutation
def test_mutant_the_asked_lexicon_swallows_a_question(tmp_path):
    """Adding bare 「記得」 scores 「你記得我嗎」 as a request to remember."""
    m = load_mutant(tmp_path, "salience",
                    ('ASKED_MARKERS = (\n    "記住",', 'ASKED_MARKERS = (\n    "記得", "記住",'))
    assert m.signals({"role": "user", "text": "你記得我嗎"})["asked"] == 1.0


@pytest.mark.mutation
def test_mutant_the_shipped_switch_is_on(tmp_path):
    """The default is the whole de-biometricisation; nothing else changes if it flips."""
    m = load_mutant(tmp_path, "salience",
                    ("PROSODY_ENABLED = False", "PROSODY_ENABLED = True"))
    assert m.PROSODY_ENABLED is True
    assert "\nPROSODY_ENABLED = False\n" not in \
        (tmp_path / "mut_salience_1" / "salience.py").read_text()
    turn = {**USER, **{f: 0.5 for f in sal.ACOUSTIC_FIELDS}}
    assert m.score(turn) != m.score(turn, prosody=False)
    assert m.formula_id() == m.FORMULA_PROSODY


@pytest.mark.mutation
def test_mutant_a_missing_measurement_becomes_a_zero(tmp_path):
    m = load_mutant(tmp_path, "salience", (
        '    missing = [f for f in ACOUSTIC_FIELDS if f not in turn]\n',
        '    missing = []\n    turn = {**{f: 0.0 for f in ACOUSTIC_FIELDS}, **turn}\n',
    ))
    assert m.score(USER, prosody=True) == pytest.approx(
        m.score(USER, prosody=False) * (1.0 - m.PROSODY_WEIGHT)), \
        "silence was scored as flat delivery"


@pytest.mark.mutation
def test_mutant_the_prosody_term_clamps_raw_units(tmp_path):
    """A clamping term reads its input and is constant — live-looking and dead."""
    m = load_mutant(tmp_path, "salience", (
        '        if not 0.0 <= v <= 1.0:\n',
        '        v = _clamp(v)\n        if False:\n',
    ))
    hz = [{**USER, **{f: v for f in sal.ACOUSTIC_FIELDS}}
          for v in (110.0, 220.0, 440.0)]
    assert len({m.score(t, prosody=True) for t in hz}) == 1, \
        "the mutant still varied, so this is not the failure mode described"


@pytest.mark.mutation
def test_mutant_the_role_check_is_removed(tmp_path):
    """Without it the robot's own output raises the salience of the robot's output."""
    m = load_mutant(tmp_path, "salience",
                    ("    if role != SPEAKER:\n", "    if False:\n"))
    assert m.score({"role": "robot", "text": "我喜歡芫茜"}) > 0.0


@pytest.mark.mutation
def test_mutant_the_store_drops_the_pairing_rule(tmp_path):
    m = load_mutant(tmp_path, "store", (PAIRING, "        for field, companion in []:\n"
                                                 "            if False:\n"))
    with make_store(tmp_path, module=m) as store:
        store.append("events", {"kind": "turn", "salience": 0.5})
        store.flush()
        (row,) = store.read("events")
    assert "salience_formula" not in row, \
        "an uncomparable score landed on disk and every gate stayed green"


@pytest.mark.mutation
def test_mutant_stamp_forgets_the_formula_id(tmp_path):
    """Composed: the mutated producer is caught by the *real* store, not a patched one."""
    m = load_mutant(tmp_path, "salience",
                    ('    out["salience_formula"] = formula_id(prosody)\n', "\n"))
    record = m.stamp({"kind": "turn"}, USER)
    assert "salience_formula" not in record
    with make_store(tmp_path) as store:
        with pytest.raises(st.StoreError, match="salience_formula"):
            store.append("events", record)
