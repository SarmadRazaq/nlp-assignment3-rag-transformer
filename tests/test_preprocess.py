from ragpipe.preprocess import clean_text, tokenize


def test_clean_lowercases_and_strips_urls_and_html():
    raw = "GREAT product!<br/>See http://example.com/x for more &amp; info"
    assert clean_text(raw) == "great product! see for more & info"


def test_tokenize_splits_punctuation_and_keeps_negation_contractions():
    assert tokenize("it doesn't work, at all!!") == ["it", "doesn't", "work", ",", "at", "all", "!", "!"]


from ragpipe.preprocess import Vocab


def _vocab():
    train_tokens = [["good", "phone"], ["good", "case"], ["bad", "case"]]
    return Vocab.build(train_tokens, min_freq=2)


def test_vocab_keeps_only_frequent_training_tokens_after_specials():
    v = _vocab()
    assert v.itos[:4] == ["<pad>", "<unk>", "<bos>", "<eos>"]
    assert set(v.itos[4:]) == {"good", "case"}


def test_unseen_tokens_map_to_unk():
    v = _vocab()
    assert v.encode(["good", "battery"]) == [v["good"], v.unk_id]


def test_pad_or_truncate_to_fixed_length_with_mask():
    v = _vocab()
    ids, mask = v.encode_fixed(["good", "case"], max_len=4)
    assert ids == [v["good"], v["case"], v.pad_id, v.pad_id]
    assert mask == [1, 1, 0, 0]
    ids, mask = v.encode_fixed(["good", "case", "good", "case", "good"], max_len=3)
    assert ids == [v["good"], v["case"], v["good"]] and mask == [1, 1, 1]


def test_decode_skips_specials():
    v = _vocab()
    assert v.decode([v.bos_id, v["good"], v["case"], v.eos_id, v.pad_id]) == "good case"


def test_vocab_round_trips_through_dict():
    v = _vocab()
    v2 = Vocab.from_dict(v.to_dict())
    assert v2.itos == v.itos and v2["case"] == v["case"]
