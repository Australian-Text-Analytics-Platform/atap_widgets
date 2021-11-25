from collections import defaultdict
from itertools import combinations
from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np
import pandas as pd
import spacy
import textacy
from cytoolz.itertoolz import concat
from cytoolz.itertoolz import sliding_window
from sklearn.metrics.pairwise import cosine_similarity
from textacy.representations import matrix_utils
from textacy.representations.vectorizers import Vectorizer


def vector_cosine_similarity(docs: Sequence[spacy.tokens.Doc]) -> np.ndarray:
    """
    Get the pairwise cosine similarity between each
    document in docs.
    """
    vectors = np.vstack([doc.vector for doc in docs])
    return cosine_similarity(vectors)


class Conversation:
    def __init__(
        self,
        data: pd.DataFrame,
        text_column: str = "text",
        speaker_column: str = "speaker",
        id_column: Optional[str] = None,
        language_model: str = "en_core_web_sm",
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

    def vector_similarity_matrix(self) -> pd.DataFrame:
        """
        Calculate the similarity between each turn of the conversation
        based on spacy word vectors.
        """
        matrix = vector_cosine_similarity(self.data["spacy_doc"])
        return pd.DataFrame(matrix, index=self.data.index, columns=self.data.index)


class ConceptSimilarityModel:
    def __init__(
        self,
        conversation: Conversation,
        n_top_terms: int = 20,
        sentence_window_size: int = 3,
    ):
        self.conversation = conversation
        self.sentence_window_size = sentence_window_size
        self.n_top_terms = n_top_terms
        # Create textacy corpus
        self.corpus = self._create_corpus()

        # Start counting terms
        self.binary_vectorizer = Vectorizer(
            tf_type="binary",
            idf_type=None,
            dl_type=None,
        )
        self.doc_term_matrix = self.binary_vectorizer.fit_transform(
            self._get_filtered_corpus()
        )
        self.top_terms = self._get_top_terms()

    def _create_corpus(self) -> textacy.Corpus:
        return textacy.Corpus(
            lang=self.conversation.nlp, data=self.conversation.data["text"]
        )

    @staticmethod
    def _filter_tokens(doc: spacy.tokens.Doc):
        """
        Filter out stopwords, punctuation and spaces.
        Return a generator that yields tokens, converted
        to lowercase
        """
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space:
                yield token.lower_

    def _get_filtered_corpus(self):
        return (self._filter_tokens(doc) for doc in self.corpus)

    def _get_sentence_windows(self, doc: spacy.tokens.Doc, window_size: int):
        """
        We can use cytoolz.itertoolz.sliding_window
        to generate sentence windows, but it returns
        nothing if the number of sentences is less
        than the window size. Here we just return
        the sentences for documents less than the window size.
        """
        n_sentences = sum(1 for sentence in doc.sents)
        if n_sentences < window_size:
            return [doc.sents]
        else:
            return sliding_window(window_size, (sentence for sentence in doc.sents))

    def _get_top_terms(self):
        # TODO: should this work based on raw frequencies, rather than the binary
        #   per-document count?
        term_counts = matrix_utils.get_term_freqs(self.doc_term_matrix)
        top_n_indices = np.flip(np.argsort(term_counts))[: self.n_top_terms]
        top_n_terms = [
            self.binary_vectorizer.terms_list[index] for index in top_n_indices
        ]
        return top_n_terms

    def get_cooccurrence_counts(self) -> dict:
        """
        Calculate the occurrence count of each term across all sentence
        windows, and the co-occurrence of each pair of terms

        Returns:
            A dictionary with "occurrence": Series of overall occurrence counts,
            "cooccurrence": DataFrame of term-term cooccurrence counts,
            "total_windows":
        """
        pair_counts = defaultdict(int)
        occurrence_counts = defaultdict(int)

        total_windows = 0
        for doc in self.corpus:
            windows = self._get_sentence_windows(
                doc, window_size=self.sentence_window_size
            )
            for window in windows:
                total_windows += 1
                # We only need the unique tokens so use a set
                current_tokens = set(
                    concat(self._filter_tokens(sentence) for sentence in window)
                )
                for token in current_tokens:
                    occurrence_counts[token] += 1

                pairs = combinations(sorted(current_tokens), 2)
                for pair in pairs:
                    pair_counts[pair] += 1

        # Create series and dataframe to store occurrence counts
        #   and cooccurrence counts
        occurrence_series = pd.Series(
            occurrence_counts,
            index=self.binary_vectorizer.terms_list,
            dtype=pd.Int64Dtype(),
        )
        cooccurrence_matrix = pd.DataFrame(
            0,
            index=self.binary_vectorizer.terms_list,
            columns=self.binary_vectorizer.terms_list,
            dtype=pd.Int64Dtype(),
        )
        for (term1, term2), count in pair_counts.items():
            cooccurrence_matrix.loc[term1, term2] = count
            cooccurrence_matrix.loc[term2, term1] = count

        return {
            "occurrence": occurrence_series,
            "cooccurrence": cooccurrence_matrix,
            "total_windows": total_windows,
        }

    @staticmethod
    def _get_contingency_counts(
        total_windows: int, occurrence: pd.Series, cooccurrence: pd.DataFrame
    ) -> Dict[tuple, pd.DataFrame]:
        """
        When calculating association statistics it helps to have the contingency
        counts between a term i and another term j.

        Returns:
            A dictionary mapping from 2-tuples ('i', 'j'), ('i', 'not_j'), etc. to a
            DataFrame of the counts of each.
        """
        return {
            ("i", "j"): cooccurrence,
            ("not_i", "not_j"): total_windows
            - (
                (-1 * cooccurrence)
                .add(occurrence, axis="rows")
                .add(occurrence, axis="columns")
            ),
            ("i", "not_j"): (-1 * cooccurrence).add(occurrence, axis="rows"),
            ("not_i", "j"): (-1 * cooccurrence).add(occurrence, axis="columns"),
        }

    def get_term_similarity_matrix(self) -> pd.DataFrame:
        """
        Calculate the similarity score S(t_i, t_j) between each pair of terms t_i.

        See Angus (2012): https://doi.org/10/b49pvx
        for details of how to calculate these scores.
        """
        counts = self.get_cooccurrence_counts()
        contingency_counts = self._get_contingency_counts(**counts)
        # Convert to probabilities
        contingency_probs = {
            k: v / counts["total_windows"] for k, v in contingency_counts.items()
        }
        # Fixes for zero counts/terms only appearing in one context
        contingency_probs[("i", "not_j")] = contingency_probs[("i", "not_j")].where(
            ~counts["cooccurrence"].eq(counts["occurrence"], axis="rows"), 1
        )
        contingency_probs[("not_i", "j")] = contingency_probs[("not_i", "j")].where(
            ~counts["cooccurrence"].eq(counts["occurrence"], axis="columns"), 1
        )

        similarity_matrix = (
            contingency_probs[("i", "j")] * contingency_probs[("not_i", "not_j")]
        ) / (contingency_probs[("not_i", "j")] * contingency_probs[("i", "not_j")])
        return similarity_matrix

    def get_concept_vectors(self, term_similarity_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Get the feature vectors for each document.

        Returns:
            A dataframe where the rows are the top terms in the corpus,
            and the columns are documents. Values are the feature value for
            that term's concept in each document.
        """
        term_doc_df = pd.DataFrame(
            self.doc_term_matrix.T.todense(),
            index=self.binary_vectorizer.terms_list,
            columns=self.conversation.data["text_id"],
        )
        key_term_similarity = term_similarity_matrix.loc[self.top_terms, :]
        return key_term_similarity @ term_doc_df

    def get_conversation_similarity(self):
        term_similarity_matrix = self.get_term_similarity_matrix()
        concept_vectors = self.get_concept_vectors(term_similarity_matrix)

        doc_doc_cosine = pd.DataFrame(
            cosine_similarity(concept_vectors.T),
            index=self.conversation.data["text_id"],
            columns=self.conversation.data["text_id"],
        )
        return doc_doc_cosine
