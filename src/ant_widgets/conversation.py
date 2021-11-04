from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import spacy


def pairwise_cosine_similarity(docs):
    vectors = np.vstack([doc.vector for doc in docs])
    return cosine_similarity(vectors)


class Conversation:
    def __init__(
        self,
        data: pd.DataFrame,
        text_column: str = "text",
        speaker_column: str = "speaker",
        id_column: Optional[str] = None,
        language_model: str = "en_core_web_lg",
    ):
        # Set up data
        self.data = data.copy()
        if id_column is None:
            id_column = "text_id"
            self.data["text_id"] = range(len(self.data))
        self.data = self.data.rename(
            columns={
                text_column: "text",
                speaker_column: "speaker",
                id_column: "text_id",
            }
        )
        self.data.set_index("text_id", drop=False, inplace=True)
        # Apply NLP
        self.nlp = spacy.load(language_model)
        self.data["spacy_doc"] = self.data["text"].apply(self.nlp)

    def __str__(self):
        return "Conversation object:\n" + str(self.data.head())

    @property
    def n_speakers(self) -> int:
        return self.data["speaker"].nunique()

    def get_speaker_names(self) -> list:
        return self.data["speaker"].unique().tolist()

    def similarity_matrix(self) -> pd.DataFrame:
        matrix = pairwise_cosine_similarity(self.data["spacy_doc"])
        return pd.DataFrame(matrix, index=self.data.index, columns=self.data.index)
