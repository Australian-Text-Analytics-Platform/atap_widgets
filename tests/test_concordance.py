import pandas as pd
import pytest
import spacy

from atap_widgets.concordance import ConcordanceTable
from atap_widgets.concordance import prepare_text_df


def test_prepare_text_df(sherlock_holmes_dummy_df):
    """
    Test we can rename/add to a dataframe to create a standardized
    dataframe
    """
    df = prepare_text_df(sherlock_holmes_dummy_df)
    assert isinstance(df["spacy_doc"].iloc[0], spacy.language.Doc)


def test_text_search(sherlock_holmes_dummy_df):
    """
    Test we can query for matches
    """
    df = prepare_text_df(sherlock_holmes_dummy_df)
    # Matches "she" and "Sherlock"
    table = ConcordanceTable(df, keyword="she")
    results = table.to_dataframe()
    assert (results["text_id"] == [0, 0, 2]).all()

    table.ignore_case = False
    case_sensitive_results = table.to_dataframe()
    # Should no longer match "Sherlock"
    assert (case_sensitive_results["text_id"] == [0, 2]).all()


def test_text_search_regex():
    """
    Test we can search using regular expressions.
    """
    df = pd.DataFrame(
        {
            "text_id": [1, 2, 3, 4],
            "text": ["AGGACTTA", "invalid AGGACTA", "XPOAGTC", "TACCA"],
        }
    )
    df = prepare_text_df(df, id_column="text_id")
    # Basic DNA base matcher
    # Only match if the entire string is valid DNA bases (A, C, G, T)
    table = ConcordanceTable(df, keyword=r"^[ACGT]+$", regex=True)
    results = table.to_dataframe()
    assert (results["text_id"] == [1, 4]).all()


def test_sorting(sortable_text_df):
    df = prepare_text_df(sortable_text_df, id_column="text_id")
    table = ConcordanceTable(df, keyword="pen", sort="text_id")

    # text_id/default sort: original order
    text_id_sorted = table.to_dataframe()
    assert (text_id_sorted["text_id"] == [1, 2, 3]).all()

    # Left context sort: My, The, Your should be 2, 1, 3
    table.sort = "left"
    table.sort_offset = 1
    left_context_sorted = table.to_dataframe()
    assert (left_context_sorted["text_id"] == [2, 1, 3]).all()

    # Right context sort: blue, green, red should be 3, 2, 1
    table.sort = "right"
    # sort_offset = 2 to skip over "is"
    table.sort_offset = 2
    right_context_sorted = table.to_dataframe()
    assert (right_context_sorted["text_id"] == [3, 2, 1]).all()


@pytest.mark.parametrize(
    "sort,offset,expected",
    [
        # 1st left context: apple, carrot, banana should be 1, 3, 2
        ("left", 1, [1, 3, 2]),
        # 2nd left context: banana, apple, carrot should be 2, 1, 3
        ("left", 2, [2, 1, 3]),
        # Carrot, banana, apple should be 3, 2, 1
        ("left", 3, [3, 2, 1]),
        # carrot, apple, banana should be 2, 3, 1
        ("right", 1, [2, 3, 1]),
        # banana, carrot, apple should be 3, 1, 2
        ("right", 2, [3, 1, 2]),
        # a, b, c should be 1, 2, 3
        ("right", 3, [1, 2, 3]),
    ],
)
def test_sorting_by_offsets(sort, offset, expected, multisortable_text_df):
    df = prepare_text_df(multisortable_text_df, id_column="text_id")
    table = ConcordanceTable(df, keyword="search", sort=sort, sort_offset=offset)
    sorted = table.to_dataframe()
    assert (sorted["text_id"] == expected).all()
