import pandas as pd
import spacy
#from atap_widgets.concordance import ConcordanceTable, Foo
from src.atap_widgets.concordance import ConcordanceTable
from atap_widgets.concordance import prepare_text_df



def test_history(sherlock_holmes_dummy_df):
    """
    To do : test history
    """
    df = prepare_text_df(sherlock_holmes_dummy_df)
    # Matches "she" and "Sherlock"
    table = ConcordanceTable(df, keyword="she",ignore_case = False,historic_lines=3)
    results = table.to_dataframe()
    print("RESULTS \n",results[["text_id","history"]])

    table = ConcordanceTable(df, keyword="she",ignore_case = False)
    results = table.to_dataframe()
    
