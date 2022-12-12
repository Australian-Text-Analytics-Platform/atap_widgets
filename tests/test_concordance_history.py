import os

import pandas as pd
import spacy
from src.atap_widgets.concordance import ConcordanceLoader
from src.atap_widgets.concordance import ConcordanceTable

from atap_widgets.concordance import prepare_text_df

# from atap_widgets.concordance import ConcordanceTable #when commiting


def test_data_ingestion(sherlock_holmes_dummy_df):
    """ConcordanceLoader should treat csv and dataframes equally"""
    DataDF = ConcordanceLoader(type="dataframe", df_input=sherlock_holmes_dummy_df)
    df_df = DataDF.get_original_data()
    DataCSV = ConcordanceLoader(type="csv", path="tests/data/sherlock_for_testing.csv")
    df_csv = DataCSV.get_original_data()
    assert pd.testing.assert_frame_equal(df_df, df_csv) == None
