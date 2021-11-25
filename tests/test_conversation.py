"""Tests for the conversation module."""
import pandas as pd
import pytest
import spacy

from ant_widgets.conversation import ConceptSimilarityModel
from ant_widgets.conversation import Conversation


@pytest.fixture
def example_conversation_data():
    """Dataframe with basic conversation data."""
    return pd.DataFrame(
        {
            "text": ["Hello.", "How are you?", "Good thanks.", "Great!"],
            "speaker": ["A", "B", "A", "B"],
        }
    )


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


def test_contingency_counts():
    # Example occurrence/cooccurrence with 4 items AB, AC, AD, BC
    total_windows = 4
    occurrence = pd.Series({"A": 3, "B": 2, "C": 2, "D": 1})
    cooccurrence = pd.DataFrame(0, index=list("ABCD"), columns=list("ABCD"))
    for x, y in ("AB", "AC", "AD", "BC"):
        cooccurrence.loc[x, y] += 1
        cooccurrence.loc[y, x] += 1
    counts = ConceptSimilarityModel._get_contingency_counts(
        total_windows=total_windows, occurrence=occurrence, cooccurrence=cooccurrence
    )
    # ("i", "j") is just the original cooccurrence matrix
    pd.testing.assert_frame_equal(counts[("i", "j")], cooccurrence)
    assert counts[("i", "j")].loc["A", "B"] == 1
    assert counts[("i", "not_j")].loc["A", "B"] == 2
    assert counts[("not_i", "j")].loc["A", "B"] == 1
    assert counts[("not_i", "not_j")].loc["A", "B"] == 0

    # Check these cases are symmetric
    for contingency in [("i", "j"), ("not_i", "not_j")]:
        table = counts[contingency]
        assert table.loc["A", "B"] == table.loc["B", "A"]
