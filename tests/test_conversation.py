"""Tests for the conversation module."""
import itertools
import random

import pandas as pd
import pytest
import spacy

from ant_widgets.conversation import ConceptSimilarityModel
from ant_widgets.conversation import Conversation


@pytest.fixture
def example_conversation_data() -> pd.DataFrame:
    """Dataframe with basic conversation data."""
    return pd.DataFrame(
        {
            "text": ["Hello.", "How are you?", "Good thanks.", "Great!"],
            "speaker": ["A", "B", "A", "B"],
        }
    )


@pytest.fixture
def example_cooccurrence_counts() -> dict:
    """
    Example occurrence/cooccurrence with 5 items AB, AC, AD, BC, CD
    """
    counts = {
        "total_windows": 5,
        "occurrence": pd.Series({"A": 3, "B": 2, "C": 3, "D": 2}),
        "cooccurrence": pd.DataFrame(0, index=list("ABCD"), columns=list("ABCD")),
    }
    for x, y in ("AB", "AC", "AD", "BC", "CD"):
        counts["cooccurrence"].loc[x, y] += 1
        counts["cooccurrence"].loc[y, x] += 1
    return counts


def get_random_cooccurrence_counts(n: int = 20) -> dict:
    """
    Return a dict with total_windows, occurrence, cooccurrence
    for 20 random items formed from the combinations of A, B, C, D
    """
    possible_pairs = list(itertools.combinations("ABCD", 2))
    items = random.choices(possible_pairs, k=n)
    counts = {
        "total_windows": n,
        "occurrence": pd.Series({k: 0 for k in "ABCD"}),
        "cooccurrence": pd.DataFrame(0, index=list("ABCD"), columns=list("ABCD")),
    }
    for x, y in items:
        counts["occurrence"].loc[x] += 1
        counts["occurrence"].loc[y] += 1
        counts["cooccurrence"].loc[x, y] += 1
        counts["cooccurrence"].loc[y, x] += 1
    return counts


@pytest.fixture(scope="session")
def basic_spacy_nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture
def sherlock_holmes_doc(sherlock_holmes_five_sentences, basic_spacy_nlp):
    return basic_spacy_nlp(sherlock_holmes_five_sentences)


@pytest.fixture
def sherlock_holmes_dummy_conversation(sherlock_holmes_doc):
    """
    Treat each sentence from the Sherlock Holmes example as a turn
    in a conversation, for checking contingency counts etc.
    """
    df = pd.DataFrame(
        {
            "text": [str(sentence) for sentence in sherlock_holmes_doc.sents],
            "speaker": list("ABABA"),
        }
    )
    return Conversation(df)


def test_conversation(example_conversation_data):
    """Check we can create a Conversation object from a dataframe."""
    convo = Conversation(
        data=example_conversation_data, language_model="en_core_web_sm"
    )

    # Test ID column was automatically created
    assert (convo.data["text_id"] == [0, 1, 2, 3]).all()
    assert convo.n_speakers == 2


def test_get_sentence_windows(sherlock_holmes_doc):
    windows = list(
        ConceptSimilarityModel._get_sentence_windows(sherlock_holmes_doc, window_size=3)
    )
    # 5 sentences should give 3 windows (final window is sentences 3, 4, 5)
    assert len(windows) == 3
    assert all(len(w) == 3 for w in windows)

    windows2 = list(
        ConceptSimilarityModel._get_sentence_windows(sherlock_holmes_doc, window_size=2)
    )
    # 5 sentences should give 4 windows (final window is sentences 4, 5)
    assert len(windows2) == 4
    assert all(len(w) == 2 for w in windows2)


def test_get_sentence_windows_short_doc(basic_spacy_nlp):
    """
    When the document has less sentences than the window size,
    we should still get the sentences returned from the document.
    """
    short_doc = basic_spacy_nlp(
        "This document has only two sentences. It's a short document"
    )
    windows = list(
        ConceptSimilarityModel._get_sentence_windows(short_doc, window_size=3)
    )
    assert len(windows) == 1
    sentences = list(windows[0])
    assert len(sentences) == 2


def test_cooccurrence_counts(sherlock_holmes_dummy_conversation):
    concept_model = ConceptSimilarityModel(
        sherlock_holmes_dummy_conversation, n_top_terms=5, sentence_window_size=3
    )
    counts = concept_model.get_cooccurrence_counts()
    cooccurrence = counts["cooccurrence"]
    assert cooccurrence.loc["sherlock", "holmes"] == 1
    assert cooccurrence.loc["sherlock", "abhorrent"] == 0


def test_contingency_counts(example_cooccurrence_counts):
    total_windows = example_cooccurrence_counts["total_windows"]
    occurrence = example_cooccurrence_counts["occurrence"]
    cooccurrence = example_cooccurrence_counts["cooccurrence"]
    counts = ConceptSimilarityModel._get_contingency_counts(
        total_windows=total_windows, occurrence=occurrence, cooccurrence=cooccurrence
    )
    # ("i", "j") is just the original cooccurrence matrix
    print(cooccurrence)
    pd.testing.assert_frame_equal(counts[("i", "j")], cooccurrence)
    assert counts[("i", "j")].loc["A", "B"] == 1
    assert counts[("i", "not_j")].loc["A", "B"] == 2
    assert counts[("not_i", "j")].loc["A", "B"] == 1
    assert counts[("not_i", "not_j")].loc["A", "B"] == 1

    # Check these cases are symmetric
    for contingency in [("i", "j"), ("not_i", "not_j")]:
        table = counts[contingency]
        assert table.loc["A", "B"] == table.loc["B", "A"]


def test_contingency_counts_random_data():
    counts = get_random_cooccurrence_counts(n=20)
    contingency = ConceptSimilarityModel._get_contingency_counts(**counts)
    assert (
        contingency[("i", "j")].loc["A", "B"]
        + contingency[("i", "not_j")].loc["A", "B"]
        == counts["occurrence"].loc["A"]
    )
    assert (
        contingency[("i", "j")].loc["A", "B"]
        + contingency[("not_i", "j")].loc["A", "B"]
        == counts["occurrence"].loc["B"]
    )
    # Should add up to the total items across all 4 contingencies
    assert (sum(contingency.values()) == counts["total_windows"]).all().all()


def test_get_term_similarity_matrix(example_cooccurrence_counts):
    term_similarity = ConceptSimilarityModel.get_term_similarity_matrix(
        **example_cooccurrence_counts
    )
    # Should be (P(i, j) * P(not_i, not_j)) / (P(not_i, j) * P(i, not_j))
    assert term_similarity.loc["A", "B"] == ((1 / 5 * 1 / 5) / (1 / 5 * 2 / 5))
