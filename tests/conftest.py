import pandas as pd
import pytest
import spacy

from ant_widgets.conversation import Conversation


@pytest.fixture(scope="session")
def sherlock_holmes_five_sentences():
    return """To Sherlock Holmes she is always the woman. I have seldom heard him
    mention her under any other name. In his eyes she eclipses and predominates the
    whole of her sex. It was not that he felt any emotion akin to love for Irene
    Adler. All emotions, and that one particularly, were abhorrent to his cold,
    precise but admirably balanced mind. """


@pytest.fixture(scope="session")
def basic_spacy_nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture
def sherlock_holmes_doc(sherlock_holmes_five_sentences, basic_spacy_nlp):
    return basic_spacy_nlp(sherlock_holmes_five_sentences)


@pytest.fixture
def sherlock_holmes_dummy_df(sherlock_holmes_doc):
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
    return df


@pytest.fixture
def sherlock_holmes_dummy_conversation(sherlock_holmes_dummy_df):
    return Conversation(sherlock_holmes_dummy_df)
