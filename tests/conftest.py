import pytest


@pytest.fixture(scope="session")
def sherlock_holmes_five_sentences():
    return """To Sherlock Holmes she is always the woman. I have seldom heard him
    mention her under any other name. In his eyes she eclipses and predominates the
    whole of her sex. It was not that he felt any emotion akin to love for Irene
    Adler. All emotions, and that one particularly, were abhorrent to his cold,
    precise but admirably balanced mind. """
