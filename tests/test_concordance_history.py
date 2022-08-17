import pandas as pd
import spacy
from atap_widgets.concordance import ConcordanceTable #when commiting
#from src.atap_widgets.concordance import ConcordanceTable , DataIngest#for dev
from atap_widgets.concordance import prepare_text_df
import os

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


def test_data_ingestion(sherlock_holmes_dummy_df):
    """ dataIngest should treat csv and dataframes equally
    """
    DataDF = DataIngest(type = "dataframe",df_input = sherlock_holmes_dummy_df,chunk = 2)
    df_df = DataDF.get_original_data()
    DataCSV = DataIngest(type = "csv",path = "tests/data/sherlock_for_testing.csv",chunk = 2)
    df_csv = DataCSV.get_original_data()
    assert(pd.testing.assert_frame_equal(df_df, df_csv) == None)
