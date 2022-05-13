"""Tests for the conversation module."""
import itertools
import random

import pandas as pd
import pytest

from atap_widgets.conversation import ConceptSimilarityModel
from atap_widgets.conversation import Conversation


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
def example_similarity_scores() -> pd.DataFrame:
    """
    Dataframe with example document-document (or utterance-utterance)
    similarity scores
    """
    df = pd.DataFrame(
        {
            "A": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "B": [0.9, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1],
            "C": [0.8, 0.5, 1.0, 0.1, 0.2, 0.3, 0.4],
            "D": [0.7, 0.4, 0.1, 1.0, 0.5, 0.6, 0.7],
            "E": [0.6, 0.3, 0.2, 0.5, 1.0, 0.2, 0.4],
            "F": [0.5, 0.2, 0.3, 0.6, 0.2, 1.0, 0.5],
            "G": [0.4, 0.1, 0.4, 0.7, 0.4, 0.5, 1.0],
        },
        index=list("ABCDEFG"),
    )
    return df


@pytest.fixture
def example_similarity_conversation() -> Conversation:
    """
    Dummy conversation object to use with example_similarity_scores.
    Text is not meaningful, it just matches the structure
    required to work with the example_similarity_scores matrix
    :return:
    """
    df = pd.DataFrame(
        {
            "text_id": list("ABCDEFG"),
            "text": [
                "Hello",
                "Hi",
                "How are you",
                "I'm good",
                "Good",
                "Bye",
                "Goodbye",
            ],
            "speaker": ["Alice", "Bob", "Alice", "Bob", "Alice", "Bob", "Alice"],
        }
    )
    return Conversation(df, id_column="text_id", language_model="en_core_web_sm")


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


def test_conversation(example_conversation_data):
    """Check we can create a Conversation object from a dataframe."""
    convo = Conversation(
        data=example_conversation_data, language_model="en_core_web_sm"
    )

    # Test ID column was automatically created
    assert (convo.data["text_id"] == [0, 1, 2, 3]).all()
    assert convo.n_speakers == 2


def test_conversation_different_column_names(example_conversation_data):
    """
    Check we can create a Conversation object from a dataframe
    with different column names for text and speaker.
    """
    data = example_conversation_data.copy()
    data = data.rename(columns={"text": "utterance", "speaker": "current.name"})
    assert "current.name" in data.columns
    convo = Conversation(
        data=data,
        text_column="utterance",
        speaker_column="current.name",
        language_model="en_core_web_sm",
    )
    assert "text" in convo.data.columns
    assert "speaker" in convo.data.columns
    assert convo.n_speakers == 2
    assert convo.get_speaker_names() == ["A", "B"]


def test_conversation_clashing_names(example_conversation_data):
    """
    Check we can create a Conversation object from a dataframe
    with different column names for text and speaker.
    """
    data = example_conversation_data.copy()
    data = data.rename(columns={"text": "utterance"})
    # Copy speaker column so we have both speaker and current.name
    data["current.name"] = data["speaker"]
    assert "current.name" in data.columns
    assert "speaker" in data.columns
    with pytest.warns(
        UserWarning, match="we are using 'current.name' as the speaker_column"
    ):
        convo = Conversation(
            data=data,
            text_column="utterance",
            speaker_column="current.name",
            language_model="en_core_web_sm",
        )
    assert "text" in convo.data.columns
    assert "speaker" in convo.data.columns
    assert convo.n_speakers == 2
    assert convo.get_speaker_names() == ["A", "B"]


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
        sherlock_holmes_dummy_conversation, key_terms=5, sentence_window_size=3
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


def test_topic_recurrence_invalid_args(
    example_similarity_scores, example_similarity_conversation
):
    with pytest.raises(ValueError, match="Invalid time_scale"):
        example_similarity_conversation.get_topic_recurrence(
            example_similarity_scores, "very short", "forward", "self"
        )


def test_topic_recurrence_scores(
    example_similarity_scores, example_similarity_conversation
):
    sfs_scores = example_similarity_conversation.get_topic_recurrence(
        similarity=example_similarity_scores,
        time_scale="short",
        direction="forward",
        speaker="self",
    )
    assert sfs_scores["C"] == example_similarity_scores.loc["C", "E"]
    assert sfs_scores["B"] == example_similarity_scores.loc["B", "D"]

    sfo_scores = example_similarity_conversation.get_topic_recurrence(
        similarity=example_similarity_scores,
        time_scale="short",
        direction="forward",
        speaker="other",
    )
    assert sfo_scores["C"] == example_similarity_scores.loc["C", "D"]
    assert sfo_scores["B"] == example_similarity_scores.loc["B", "C"]

    mfs_scores = example_similarity_conversation.get_topic_recurrence(
        similarity=example_similarity_scores,
        time_scale="medium",
        direction="forward",
        speaker="self",
        t_medium=4,
    )
    assert mfs_scores["A"] == (
        example_similarity_scores.loc["A", "C"]
        + example_similarity_scores.loc["A", "E"]
    )

    mfo_scores = example_similarity_conversation.get_topic_recurrence(
        similarity=example_similarity_scores,
        time_scale="medium",
        direction="forward",
        speaker="other",
        t_medium=4,
    )
    assert mfo_scores["B"] == (
        example_similarity_scores.loc["B", "C"]
        + example_similarity_scores.loc["B", "E"]
    )

    lfs_scores = example_similarity_conversation.get_topic_recurrence(
        similarity=example_similarity_scores,
        time_scale="long",
        direction="forward",
        speaker="self",
    )
    assert lfs_scores["A"] == example_similarity_scores.loc["A", ["C", "E", "G"]].sum()

    lfo_scores = example_similarity_conversation.get_topic_recurrence(
        similarity=example_similarity_scores,
        time_scale="long",
        direction="forward",
        speaker="other",
    )
    assert lfo_scores["A"] == example_similarity_scores.loc["A", ["B", "D", "F"]].sum()


def test_topic_recurrence_scores_normalized(
    example_similarity_scores, example_similarity_conversation
):
    sfs_scores = example_similarity_conversation.get_topic_recurrence(
        similarity=example_similarity_scores,
        time_scale="short",
        direction="forward",
        speaker="self",
        normalize=True,
    )
    assert sfs_scores["C"] == example_similarity_scores.loc["C", "E"]
    assert sfs_scores["B"] == example_similarity_scores.loc["B", "D"]

    sfo_scores = example_similarity_conversation.get_topic_recurrence(
        similarity=example_similarity_scores,
        time_scale="short",
        direction="forward",
        speaker="other",
        normalize=True,
    )
    assert sfo_scores["C"] == example_similarity_scores.loc["C", "D"]
    assert sfo_scores["B"] == example_similarity_scores.loc["B", "C"]

    mfs_scores = example_similarity_conversation.get_topic_recurrence(
        similarity=example_similarity_scores,
        time_scale="medium",
        direction="forward",
        speaker="self",
        t_medium=4,
        normalize=True,
    )
    assert mfs_scores["A"] == example_similarity_scores.loc["A", ["C", "E"]].sum() / 2

    mfo_scores = example_similarity_conversation.get_topic_recurrence(
        similarity=example_similarity_scores,
        time_scale="medium",
        direction="forward",
        speaker="other",
        t_medium=4,
        normalize=True,
    )
    assert mfo_scores["B"] == example_similarity_scores.loc["B", ["C", "E"]].sum() / 2

    lfs_scores = example_similarity_conversation.get_topic_recurrence(
        similarity=example_similarity_scores,
        time_scale="long",
        direction="forward",
        speaker="self",
        normalize=True,
    )
    assert (
        lfs_scores["A"] == example_similarity_scores.loc["A", ["C", "E", "G"]].sum() / 3
    )

    lfo_scores = example_similarity_conversation.get_topic_recurrence(
        similarity=example_similarity_scores,
        time_scale="long",
        direction="forward",
        speaker="other",
        normalize=True,
    )
    assert (
        lfo_scores["A"] == example_similarity_scores.loc["A", ["B", "D", "F"]].sum() / 3
    )


def test_all_topic_recurrences(
    example_similarity_scores, example_similarity_conversation
):
    similarity = example_similarity_scores
    all_scores = example_similarity_conversation.get_all_topic_recurrences(similarity)

    # Check we get the same results in the all_scores df as when we get
    #   individual scores
    sfs_scores = example_similarity_conversation.get_topic_recurrence(
        similarity, time_scale="short", direction="forward", speaker="self"
    )
    sfs_from_df = all_scores.query(
        "time_scale == 'short' & direction == 'forward' & speaker == 'self'"
    )
    sfs_series = pd.Series(sfs_from_df["score"].tolist(), index=sfs_from_df["text_id"])
    assert (sfs_series == sfs_scores).all()

    lfo_scores = example_similarity_conversation.get_topic_recurrence(
        similarity, time_scale="long", direction="backward", speaker="other"
    )
    lfo_from_df = all_scores.query(
        "time_scale == 'long' & direction == 'backward' & speaker == 'other'"
    )
    lfo_series = pd.Series(lfo_from_df["score"].tolist(), index=lfo_from_df["text_id"])
    assert (lfo_series == lfo_scores).all()


@pytest.fixture
def example_grouped_recurrence_conversation():
    data = pd.DataFrame(
        {
            "text_id": [1, 2, 3, 4, 5, 6],
            "speaker": ["a", "b", "c", "b", "a", "c"],
            "group": ["G1", "G1", "G2", "G1", "G1", "G2"],
            # Don't need actual text values for this
            "text": ["dummy"] * 6,
        }
    )
    return Conversation(data, id_column="text_id")


@pytest.fixture
def example_grouped_recurrence_similarity():
    df = pd.DataFrame(
        {
            1: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
            2: [0.9, 1.0, 0.5, 0.4, 0.3, 0.2],
            3: [0.8, 0.5, 1.0, 0.1, 0.2, 0.3],
            4: [0.7, 0.4, 0.1, 1.0, 0.5, 0.6],
            5: [0.6, 0.3, 0.2, 0.5, 1.0, 0.2],
            6: [0.5, 0.2, 0.3, 0.6, 0.2, 1.0],
        },
        index=[1, 2, 3, 4, 5, 6],
    )
    return df


def test_grouped_recurrence(
    example_grouped_recurrence_conversation, example_grouped_recurrence_similarity
):
    conversation = example_grouped_recurrence_conversation
    similarity = example_grouped_recurrence_similarity

    person_to_person = conversation.get_grouped_recurrence(
        similarity, grouping_column="speaker"
    )
    assert person_to_person.loc["a", "c"] == (0.8 + 0.5 + 0.2) * 3
    # These scores are non-symmetrical, they only include
    #   cells where the second group speaks after the first group
    assert person_to_person.loc["c", "a"] == (0.2) * 1

    group_to_group = conversation.get_grouped_recurrence(
        similarity, grouping_column="group"
    )
    # For this, for each G1 turn, we need to find all the G2
    #   turns that come after it
    g1_g2_scores = [0.8 + 0.5, 0.5 + 0.2, 0.6, 0.2]
    assert group_to_group.loc["G1", "G2"] == pytest.approx(sum(g1_g2_scores) * 6)
    # There's only one G2 turn with G1 turns after it
    g2_g1_scores = [0.1 + 0.2]
    assert group_to_group.loc["G2", "G1"] == pytest.approx(sum(g2_g1_scores) * 2)
