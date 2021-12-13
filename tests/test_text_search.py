import pandas as pd
import spacy

from ant_widgets.text_search import prepare_text_df
from ant_widgets.text_search import SearchTable


def test_prepare_text_df(sherlock_holmes_dummy_df):
    df = prepare_text_df(sherlock_holmes_dummy_df)
    assert isinstance(df["spacy_doc"].iloc[0], spacy.language.Doc)


def test_text_search(sherlock_holmes_dummy_df):
    df = prepare_text_df(sherlock_holmes_dummy_df)
    # Matches "she" and "Sherlock"
    table = SearchTable(df, keyword="she")
    results = table.to_dataframe()
    assert (results["text_id"] == [0, 0, 2]).all()


def test_text_search_regex():
    df = pd.DataFrame(
        {
            "text_id": [1, 2, 3, 4],
            "text": ["AGGACTTA", "invalid AGGACTA", "XPOAGTC", "TACCA"],
        }
    )
    df = prepare_text_df(df, id_column="text_id")
    # Basic DNA base matcher
    # Only match if the entire string is valid DNA bases (A, C, G, T)
    table = SearchTable(df, keyword=r"^[ACGT]+$", regex=True)
    results = table.to_dataframe()
    assert (results["text_id"] == [1, 4]).all()
