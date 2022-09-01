import pandas as pd
import spacy
#from atap_widgets.concordance import ConcordanceTable #when commiting
from src.atap_widgets.concordance import ConcordanceTable , DataIngest #for dev
from atap_widgets.concordance import prepare_text_df
import os

def test_data_ingestion(sherlock_holmes_dummy_df):
    """ dataIngest should treat csv and dataframes equally
    """
    DataDF = DataIngest(type = "dataframe",df_input = sherlock_holmes_dummy_df,chunk = 2)
    df_df = DataDF.get_original_data()
    DataCSV = DataIngest(type = "csv",path = "tests/data/sherlock_for_testing.csv",chunk = 2)
    df_csv = DataCSV.get_original_data()
    assert(pd.testing.assert_frame_equal(df_df, df_csv) == None)
