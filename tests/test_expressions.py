import pytest

from maskcast.expressions import (
    CANONICAL,
    KEYWORD_ALIASES,
    asset_candidates,
    canon,
    canonical_asset_name,
    derive_valence_arousal,
    ema_probs,
    level_code,
    match_keyword,
    select_asset,
    split_assets,
    top_label,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Happiness", "happy"),
        ("scared", "fear"),
        ("surprised", "surprise"),
        (" anger ", "angry"),
        ("", ""),
        (None, ""),
        ("money", "money"),
    ],
)
def test_canon(raw, expected):
    assert canon(raw) == expected


def test_canonical_asset_name_keeps_va_suffix():
    assert canonical_asset_name("Surprised_v0_a+") == "surprise_v0_a+"
    assert canonical_asset_name("scared") == "fear"
    assert canonical_asset_name("wink") == "wink"


@pytest.mark.parametrize("value,code", [(0.9, "+"), (0.5, "+"), (0.2, "0"), (-0.5, "-"), (-1.0, "-")])
def test_level_code(value, code):
    assert level_code(value, 0.5, -0.5) == code


def test_asset_candidates_most_specific_first():
    assert asset_candidates("happy", 0.8, 0.8) == ["happy_v+_a+", "happy", "neutral"]
    assert asset_candidates("neutral", 0.0, -0.9) == ["neutral_v0_a-", "neutral"]


def test_select_asset_prefers_variant_then_base_then_neutral():
    assets = {"happy_v+_a+": "H++", "happy": "H", "neutral": "N"}
    assert select_asset(assets, "happy", 0.9, 0.9) == ("happy_v+_a+", "H++", False)
    assert select_asset(assets, "happy", 0.0, 0.0) == ("happy", "H", True)
    assert select_asset(assets, "sad", -0.9, 0.0) == ("neutral", "N", True)
    assert select_asset({}, "sad", 0.0, 0.0) == ("<none>", None, True)


def test_select_asset_respects_custom_thresholds():
    assets = {"angry_v-_a+": "hot", "angry": "base"}
    assert select_asset(assets, "angry", -0.3, 0.3)[1] == "base"
    assert select_asset(assets, "angry", -0.3, 0.3, v_low=-0.2, a_high=0.2)[1] == "hot"


def test_match_keyword_uses_aliases_and_order():
    assert match_keyword("oh WOW that is incredible", ["amazing", "money"]) == "amazing"
    assert match_keyword("make it rain", ["amazing", "money"]) == "money"
    assert match_keyword("nothing to see", ["amazing"]) is None
    assert match_keyword("", ["amazing"]) is None
    # keywords without an alias list match their own name
    assert match_keyword("say potato", ["potato"]) == "potato"


def test_split_assets():
    emotions, keywords = split_assets(["neutral", "Scared", "happy_v+_a+", "wink", "money"])
    assert emotions == ["fear", "neutral"]
    assert keywords == ["money", "wink"]


def test_ema_probs_blends_sparse_maps():
    out = ema_probs({"happy": 1.0}, {"sad": 1.0}, 0.25)
    assert out == pytest.approx({"happy": 0.75, "sad": 0.25})


def test_derive_valence_arousal():
    v, a = derive_valence_arousal({"happy": 0.7, "sad": 0.1, "neutral": 0.2})
    assert v == pytest.approx(0.6)
    assert a == pytest.approx(0.5)
    assert derive_valence_arousal({"neutral": 1.0}) is None


def test_top_label():
    assert top_label({"Happiness": 0.6, "neutral": 0.4}) == "happy"
    assert top_label({}) == "neutral"


def test_keyword_table_is_clean():
    assert set(KEYWORD_ALIASES).isdisjoint(CANONICAL)
    for aliases in KEYWORD_ALIASES.values():
        assert aliases and all(a == a.lower() for a in aliases)
