"""Tests for the conversation module."""
import pandas as pd
import pytest
import spacy

from ant_widgets import conversation


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
    return conversation.Conversation(df)


def test_conversation(example_conversation_data):
    """Check we can create a Conversation object from a dataframe."""
    convo = conversation.Conversation(
        data=example_conversation_data, language_model="en_core_web_sm"
    )

    # Test ID column was automatically created
    assert (convo.data["text_id"] == [0, 1, 2, 3]).all()
    assert convo.n_speakers == 2


def test_get_sentence_windows(sherlock_holmes_doc):
    windows = list(
        conversation.ConceptSimilarityModel._get_sentence_windows(
            sherlock_holmes_doc, window_size=3
        )
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
        conversation.ConceptSimilarityModel._get_sentence_windows(
            short_doc, window_size=3
        )
    )
    assert len(windows) == 1
    sentences = list(windows[0])
    assert len(sentences) == 2


def test_cooccurrence_counts(sherlock_holmes_dummy_conversation):
    concept_model = conversation.ConceptSimilarityModel(
        sherlock_holmes_dummy_conversation, n_top_terms=5, sentence_window_size=3
    )
    counts = concept_model.get_cooccurrence_counts()
    cooccurrence = counts["cooccurrence"]
    assert cooccurrence.loc["sherlock", "holmes"] == 1
    assert cooccurrence.loc["sherlock", "abhorrent"] == 0
