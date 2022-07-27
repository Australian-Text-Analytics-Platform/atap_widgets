import pandas as pd
import spacy
#from atap_widgets.concordance import ConcordanceTable
from src.atap_widgets.concordance import ConcordanceTable
from atap_widgets.concordance import prepare_text_df


def test_history(sherlock_holmes_dummy_df):
    """
    Selecting three lines in context should bring up current line with match and two above.
    """
    df = prepare_text_df(sherlock_holmes_dummy_df)
    # Matches "she" and "Sherlock"
    table = ConcordanceTable(df, keyword="she",ignore_case = False,historic_utterances=3)
    results = table.to_dataframe()
    right_answer = """To Sherlock Holmes she is always the woman. I have seldom heard him
    mention her under any other name. In his eyes she eclipses and predominates the
    whole of her sex."""

    assert(results["history"].iloc[1] == right_answer) #returns two lines of context


def test_history_none(sherlock_holmes_dummy_df):
    """
    Selecting no lines in context should bring up a dummy blank string.
    """
    df = prepare_text_df(sherlock_holmes_dummy_df)
    # Matches "she" and "Sherlock"
    table = ConcordanceTable(df, keyword="she",ignore_case = False,historic_utterances=0)
    results = table.to_dataframe()
    assert(results["history"].iloc[1] == " ") #returns two lines of context

