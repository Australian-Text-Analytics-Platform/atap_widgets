import spacy

from ant_widgets.text_search import prepare_text_df
from ant_widgets.text_search import SearchTable


def test_prepare_text_df(sherlock_holmes_dummy_df):
    df = prepare_text_df(sherlock_holmes_dummy_df)
    assert isinstance(df["spacy_doc"].iloc[0], spacy.language.Doc)


def test_text_search(sherlock_holmes_dummy_df):
    df = prepare_text_df(sherlock_holmes_dummy_df)
    table = SearchTable(df, keyword="she")
    results = table.to_dataframe()
    assert (results["text_id"] == [0, 0, 2]).all()
