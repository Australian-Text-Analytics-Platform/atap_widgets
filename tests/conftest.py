from sys import stderr

import pytest
import spacy

# Workaround for spacy models being difficult to install
#   via pip
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print(
        "Downloading language model for spaCy\n"
        "(don't worry, this will only happen once)",
        file=stderr,
    )
    from spacy.cli import download

    download("en_core_web_sm")


@pytest.fixture(scope="session")
def sherlock_holmes_five_sentences():
    return """To Sherlock Holmes she is always the woman. I have seldom heard him
    mention her under any other name. In his eyes she eclipses and predominates the
    whole of her sex. It was not that he felt any emotion akin to love for Irene
    Adler. All emotions, and that one particularly, were abhorrent to his cold,
    precise but admirably balanced mind. """
