"""Tests for the conversation module."""
import pandas as pd
import pytest

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


def test_conversation(example_conversation_data):
    """Check we can create a Conversation object from a dataframe."""
    convo = conversation.Conversation(
        data=example_conversation_data, language_model="en_core_web_sm"
    )

    # Test ID column was automatically created
    assert (convo.data["text_id"] == [0, 1, 2, 3]).all()
    assert convo.n_speakers == 2
