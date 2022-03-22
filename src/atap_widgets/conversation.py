import warnings
from collections import defaultdict
from itertools import combinations
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
import pandas as pd
import sentence_transformers
import spacy
import textacy
from cytoolz.itertoolz import concat
from cytoolz.itertoolz import sliding_window
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise
from textacy.representations import matrix_utils
from textacy.representations.vectorizers import Vectorizer


def vector_cosine_similarity(docs: Sequence[spacy.tokens.Doc]) -> np.ndarray:
    """
    Get the pairwise cosine similarity between each
    document in docs.
    """
    vectors = np.vstack([doc.vector for doc in docs])
    return pairwise.cosine_similarity(vectors)


def filter_tokens(doc: spacy.tokens.Doc):
    """
    Filter out stopwords, punctuation and spaces.
    Return a generator that yields tokens, converted
    to lowercase
    """
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            yield token.lower_


def filter_corpus(corpus: textacy.Corpus):
    """
    Filter stopwords, punctuation and spaces out of a corpus
    """
    return (filter_tokens(doc) for doc in corpus)


class Conversation:
    """
    The Conversation object stores a table of conversation
    data and processes it to get it ready for analysis.

    Args:
        data: A dataframe where each row represents one turn in the
            conversation.
        text_column: The name of the column in data which holds
            the actual text of the conversation
        speaker_column: The name of the column in data which
            identifies the speaker for that turn
        id_column: An optional column name with a unique ID for
            each turn. If not provided, a default "text_id" column
            will be created.
    """

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
        if text_column != "text":
            if "text" in self.data.columns:
                warnings.warn(
                    f"The data already contains a column called 'text', but we are "
                    f"using '{text_column}' as the text_column. "
                    f"'text' will be renamed to 'text_original'"
                )
                self.data = self.data.rename(columns={"text": "text_original"})
            self.data = self.data.rename(columns={text_column: "text"})
        if speaker_column != "speaker":
            if "speaker" in self.data.columns:
                warnings.warn(
                    f"The data already contains a column called 'speaker', but we are "
                    f"using '{speaker_column}' as the speaker_column. "
                    f"'speaker' will be renamed to 'speaker_original'"
                )
                self.data = self.data.rename(columns={"speaker": "speaker_original"})
            self.data = self.data.rename(columns={speaker_column: "speaker"})
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
        self.data["spacy_doc"] = self._create_spacy_docs()

    def __str__(self):
        return "Conversation object:\n" + str(self.data.head())

    def _create_spacy_docs(self) -> pd.Series:
        """
        Convert the "text" column of the data to a spacy Doc
        """
        return self.data["text"].apply(self.nlp)

    def add_stopword(self, word: str):
        self.nlp.Defaults.stop_words.add(word)
        for stopword in self.nlp.Defaults.stop_words:
            self.nlp.vocab[stopword].is_stop = True
        # Regenerate spacy docs
        self.data["spacy_doc"] = self._create_spacy_docs()

    @property
    def n_speakers(self) -> int:
        """
        The number of speakers in the conversation.
        """
        return self.data["speaker"].nunique()

    def get_speaker_names(self) -> list:
        """
        Get a list of the speakers in the conversation.
        """
        return self.data["speaker"].unique().tolist()

    def vector_similarity_matrix(self) -> pd.DataFrame:
        """
        Calculate the similarity between each turn of the conversation
        based on spacy word vectors.
        """
        matrix = vector_cosine_similarity(self.data["spacy_doc"])
        return pd.DataFrame(matrix, index=self.data.index, columns=self.data.index)

    def create_corpus(self) -> textacy.Corpus:
        return textacy.Corpus(lang=self.nlp, data=self.data["text"])

    # TODO: allow n-gram frequencies, not just individual words?
    def get_term_frequencies(self, method: str = "overall"):
        """
        Get the term frequencies for different terms in the conversation

        :param method: "overall", for the total number of occurrences,
          "n_turns", for the number of turns each term occurs in
        :return:
        """
        methods = {"overall", "n_turns"}
        if method not in methods:
            raise ValueError(f"{method} must be one of {methods}")

        corpus = self.create_corpus()
        if method == "overall":
            tf_type = "linear"
        elif method == "n_turns":
            tf_type = "binary"
        vectorizer = Vectorizer(
            tf_type=tf_type,
            idf_type=None,
            dl_type=None,
        )
        doc_term_matrix = vectorizer.fit_transform(filter_corpus(corpus))
        term_freqs = matrix_utils.get_term_freqs(doc_term_matrix)
        term_df = pd.DataFrame({"term": vectorizer.terms_list, "frequency": term_freqs})
        term_df.sort_values("frequency", ascending=False, inplace=True)
        return term_df

    def get_most_common_terms(self, n: int = 10, method: str = "overall") -> List[str]:
        """
        Get the n most common terms

        :param n: number of top terms
        :param method: see Conversation.get_term_frequencies()
        :return:
        """
        terms = self.get_term_frequencies(method=method)
        return terms["term"].iloc[:n].tolist()


class BaseSimilarityModel:
    """Common code for similarity models, handles creation of corpus etc."""

    def __init__(
        self,
        conversation: Conversation,
        n_top_terms: int = 20,
        top_terms: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        self.conversation = conversation
        self.corpus = self.conversation.create_corpus()
        # Start counting terms
        self.binary_vectorizer = Vectorizer(
            tf_type="binary",
            idf_type=None,
            dl_type=None,
        )
        self.doc_term_matrix = self.binary_vectorizer.fit_transform(
            filter_corpus(self.corpus)
        )

        if top_terms is None:
            # Get top terms based on number of documents they appear in
            self.top_terms = self.conversation.get_most_common_terms(
                n=n_top_terms, method="n_turns"
            )
        else:
            # Use the user-specified terms list
            self.top_terms = top_terms

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


class VectorSimilarityModel(BaseSimilarityModel):
    """
    Generate similarity scores from word vectors.
    Similar to the algorithm from https://doi.org/10/b49pvx
    but instead of concept vectors based on local co-occurrences
    we use the word vectors from a spacy language model.
    """

    def __init__(
        self,
        conversation: Conversation,
        n_top_terms: int = 20,
        top_terms: Optional[Sequence[str]] = None,
    ):
        super().__init__(
            conversation=conversation, n_top_terms=n_top_terms, top_terms=top_terms
        )

    def get_term_similarity_matrix(self) -> pd.DataFrame:
        """
        Calculate cosine similarity between the word vectors for each
        term in the corpus.
        """
        vectors = np.vstack(
            [
                self.conversation.nlp.vocab.get_vector(term)
                for term in self.binary_vectorizer.terms_list
            ]
        )
        similarity_matrix = pd.DataFrame(
            data=pairwise.cosine_similarity(vectors),
            index=self.binary_vectorizer.terms_list,
            columns=self.binary_vectorizer.terms_list,
        )
        return similarity_matrix

    def get_feature_vectors(self) -> pd.DataFrame:
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
        term_similarity_matrix = self.get_term_similarity_matrix()
        key_term_similarity = term_similarity_matrix.loc[self.top_terms, :]
        return key_term_similarity @ term_doc_df

    def get_conversation_similarity(self) -> pd.DataFrame:
        feature_vectors = self.get_feature_vectors()
        doc_doc_cosine = pd.DataFrame(
            pairwise.cosine_similarity(feature_vectors.T),
            index=self.conversation.data["text_id"],
            columns=self.conversation.data["text_id"],
        )
        return doc_doc_cosine


class ConceptSimilarityModel(BaseSimilarityModel):
    """
    Generate similarity scores from local co-occurrences,
    using the algorithm from https://doi.org/10/b49pvx
    """

    def __init__(
        self,
        conversation: Conversation,
        n_top_terms: int = 20,
        top_terms: Optional[Sequence[str]] = None,
        sentence_window_size: int = 3,
    ):
        super().__init__(
            conversation=conversation, n_top_terms=n_top_terms, top_terms=top_terms
        )
        self.sentence_window_size = sentence_window_size

    @staticmethod
    def _get_sentence_windows(doc: spacy.tokens.Doc, window_size: int):
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
            ("not_i", "not_j"): (
                total_windows
                - (
                    (-1 * cooccurrence)
                    .add(occurrence, axis="rows")
                    .add(occurrence, axis="columns")
                )
            ),
            ("i", "not_j"): (-1 * cooccurrence).add(occurrence, axis="rows"),
            ("not_i", "j"): (-1 * cooccurrence).add(occurrence, axis="columns"),
        }

    @classmethod
    def get_term_similarity_matrix(
        cls, total_windows: int, occurrence: pd.Series, cooccurrence: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate the similarity score S(t_i, t_j) between each pair of terms t_i.

        See Angus (2012): https://doi.org/10/b49pvx
        for details of how to calculate these scores.
        """
        contingency_counts = cls._get_contingency_counts(
            total_windows=total_windows,
            occurrence=occurrence,
            cooccurrence=cooccurrence,
        )
        # Convert to probabilities
        contingency_probs = {
            k: v / total_windows for k, v in contingency_counts.items()
        }
        # Fixes for zero counts/terms only appearing in one context
        contingency_probs[("i", "not_j")] = contingency_probs[("i", "not_j")].where(
            ~cooccurrence.eq(occurrence, axis="rows"), 1
        )
        contingency_probs[("not_i", "j")] = contingency_probs[("not_i", "j")].where(
            ~cooccurrence.eq(occurrence, axis="columns"), 1
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
        counts = self.get_cooccurrence_counts()
        term_similarity_matrix = self.get_term_similarity_matrix(**counts)
        concept_vectors = self.get_concept_vectors(term_similarity_matrix)

        doc_doc_cosine = pd.DataFrame(
            pairwise.cosine_similarity(concept_vectors.T),
            index=self.conversation.data["text_id"],
            columns=self.conversation.data["text_id"],
        )
        return doc_doc_cosine

    @property
    def _term_doc_df(self):
        return pd.DataFrame(
            self.doc_term_matrix.T.todense(),
            index=self.binary_vectorizer.terms_list,
            columns=self.conversation.data["text_id"],
        )

    def get_term_overlaps(self, doc1, doc2):
        top_term_doc_df = self._term_doc_df.loc[self.top_terms, :]
        term_vector1 = top_term_doc_df.loc[:, doc1]
        term_vector2 = top_term_doc_df.loc[:, doc2]
        return term_vector1.dot(term_vector2)

    def get_common_concepts(self, n_concepts: int = 5):
        """
        Get the top n concepts each pair of documents has in common.

        :param n_concepts: Number of concepts to include for each pair
           of documents
        :return:
        """

        def normalize_concept_vector(v):
            # Ignore all zero concept vectors
            if (v == 0).all():
                return v
            return v / np.linalg.norm(v)

        counts = self.get_cooccurrence_counts()
        term_similarity_matrix = self.get_term_similarity_matrix(**counts)
        concept_vectors = (
            self.get_concept_vectors(term_similarity_matrix)
            .apply(normalize_concept_vector)
            .astype("float")
        )
        results = {}
        for doc1, doc2 in combinations(concept_vectors.columns, 2):
            doc_vectors = concept_vectors.loc[:, (doc1, doc2)]
            product = doc_vectors.prod(axis="columns")
            product = product.loc[product > 0]
            top_concepts = product.nlargest(n_concepts).index.tolist()
            key = (doc1, doc2)
            if not doc1 <= doc2:
                key = (doc2, doc1)
            results[key] = top_concepts
        return results


class EmbeddingModel:
    def __init__(
        self, conversation: Conversation, model_name: str = "stsb-roberta-base-v2"
    ):
        self.conversation = conversation
        self.model = SentenceTransformer(model_name)

    def get_conversation_similarity(self):
        encoding = self.model.encode(self.conversation.data["text"])
        similarity = sentence_transformers.util.pytorch_cos_sim(encoding, encoding)
        return pd.DataFrame(
            data=similarity.numpy(),
            index=self.conversation.data["text_id"],
            columns=self.conversation.data["text_id"],
        )
