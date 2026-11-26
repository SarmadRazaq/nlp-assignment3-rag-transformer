from ragpipe.explain import (SentimentLexicon, reference_explanation, build_prompt,
                             make_lm_example, PromptConfig, IGNORE_INDEX)


def _lexicon():
    docs = [["great", "love", "the", "case"], ["great", "works", "the"],
            ["broke", "terrible", "the"], ["broke", "waste", "the", "case"],
            ["okay", "average", "the"]]
    labels = [2, 2, 0, 0, 1]
    return SentimentLexicon.fit(docs, labels, n_classes=3)


def test_lexicon_picks_words_most_indicative_of_the_label():
    lex = _lexicon()
    toks = ["the", "case", "broke", "and", "great"]
    assert lex.keywords(toks, label=0, n=1) == ["broke"]
    assert lex.keywords(toks, label=2, n=1) == ["great"]


def test_lexicon_ignores_stopwords_and_punctuation():
    lex = _lexicon()
    assert "the" not in lex.keywords(["the", "!", "great"], label=2, n=3)
    assert "!" not in lex.keywords(["the", "!", "great"], label=2, n=3)


def test_reference_explanation_is_two_sentences_mentioning_label_title_and_keywords():
    text = reference_explanation(summary="Stopped working fast", sentiment=0, helpful=1,
                                 keywords=["broke", "waste"])
    assert text == ('the review is negative because the customer says " stopped working fast " '
                    'and mentions broke and waste . the detailed feedback makes it likely '
                    'helpful to other shoppers .')


def test_reference_explanation_without_title_or_keywords_still_reads_cleanly():
    text = reference_explanation(summary="", sentiment=1, helpful=0, keywords=[])
    assert text == ('the review is neutral because the customer describes mixed experiences '
                    'with the product . it is probably not very helpful to other shoppers .')


CFG = PromptConfig(max_review=5, max_retrieved=3)
RETRIEVED = [{"sentiment": 2, "tokens": ["love", "it", "so", "much"]},
             {"sentiment": 0, "tokens": ["broke"]}]


def test_prompt_orders_fields_and_truncates_each_segment():
    p = build_prompt(["a", "b", "c", "d", "e", "f"], sentiment=2, helpful=1,
                     retrieved=RETRIEVED, cfg=CFG)
    assert p == ["<bos>", "<review>", "a", "b", "c", "d", "e",
                 "<sentiment>", "positive", "<helpful>", "helpful",
                 "<ctx>", "<ex>", "positive", "love", "it", "so",
                 "<ex>", "negative", "broke", "<explain>"]


def test_baseline_prompt_has_no_retrieved_context():
    p = build_prompt(["a"], sentiment=0, helpful=0, retrieved=None, cfg=CFG)
    assert p == ["<bos>", "<review>", "a", "<sentiment>", "negative",
                 "<helpful>", "not_helpful", "<ctx>", "<none>", "<explain>"]


def test_lm_example_only_scores_explanation_tokens():
    inp, labels = make_lm_example(prompt_ids=[2, 10, 11], target_ids=[20, 21], eos_id=3)
    # full sequence = prompt + target + eos; model predicts token t+1 from tokens <= t
    assert inp == [2, 10, 11, 20, 21]
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, 20, 21, 3]
