import math

from ragpipe.training import bleu, rouge_l


def test_bleu_is_one_for_identical_long_enough_sentences():
    s = "the case fits the phone well".split()
    assert math.isclose(bleu(s, s), 1.0)


def test_bleu_does_not_reward_hypotheses_too_short_for_higher_order_ngrams():
    ref = "the review is positive because it works".split()
    assert bleu(["the", "review"], ref) < 0.05


def test_rouge_l_uses_longest_common_subsequence():
    # LCS of "a b c d" and "a c d" is "a c d": P = 3/4, R = 3/3 -> F1 = 6/7
    assert math.isclose(rouge_l("a b c d".split(), "a c d".split()), 6 / 7)
