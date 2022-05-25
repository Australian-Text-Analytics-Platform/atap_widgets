import warnings
from itertools import combinations
from itertools import permutations
from itertools import product
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

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

KeyTerms = Union[Literal["all"], int, Sequence[str]]
# Valid options for Conversation.get_topic_recurrence()
_TIME_SCALES = ("short", "medium", "long")
_DIRECTIONS = ("forward", "backward")
_SPEAKERS = ("self", "other")


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
        language_model: A spacy language model. Either a string giving
           the name of an installed model, or you can pass a model
           instance that you've already loaded and configured.
           When passing a string, we disable the parser by default.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        text_column: str = "text",
        speaker_column: str = "speaker",
        id_column: Optional[str] = None,
        language_model: Union[str, spacy.Language] = "en_core_web_sm",
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
        if isinstance(language_model, str):
            # We don't need full parsing by default, but do need sentence
            #   segmentation, which is faster
            self.nlp: spacy.Language = spacy.load(language_model)
            self.nlp.disable_pipe("parser")
            self.nlp.enable_pipe("senter")
        elif isinstance(language_model, spacy.Language):
            self.nlp = language_model
        self.data["spacy_doc"] = self._create_spacy_docs()

    def __repr__(self):
        if "name" in self.nlp.meta and "lang" in self.nlp.meta:
            model_name = f"{self.nlp.meta['lang']}_{self.nlp.meta['name']}"
        else:
            model_name = "custom"
        return (
            f"Conversation({self.n_utterances} utterances, "
            + f"{self.n_speakers} speakers, language_model='{model_name}')"
        )

    def __str__(self):
        return repr(self)

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
    def n_utterances(self):
        return self.data.shape[0]

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

    def get_all_terms(self) -> List[str]:
        corpus = self.create_corpus()
        vectorizer = Vectorizer(
            tf_type="binary",
            idf_type=None,
            dl_type=None,
        )
        vectorizer.fit_transform(filter_corpus(corpus))
        return vectorizer.terms_list

    def get_most_common_terms(self, n: int = 10, method: str = "overall") -> List[str]:
        """
        Get the n most common terms

        :param n: number of top terms
        :param method: see Conversation.get_term_frequencies()
        :return:
        """
        terms = self.get_term_frequencies(method=method)
        return terms["term"].iloc[:n].tolist()

    @staticmethod
    def _validate_args(name: str, value: str, options: Sequence[str]):
        if value not in options:
            raise ValueError(f"Invalid {name} '{value}': valid options are {options}")

    def get_all_topic_recurrences(
        self,
        similarity: pd.DataFrame,
        normalize: bool = False,
        t_short: Union[Literal["n_speakers"], int] = "n_speakers",
        t_medium: int = 10,
    ) -> pd.DataFrame:
        recurrence_types = product(_TIME_SCALES, _DIRECTIONS, _SPEAKERS)
        results = []
        for time_scale, direction, speaker in recurrence_types:
            recurrence = self.get_topic_recurrence(
                similarity=similarity,
                time_scale=time_scale,
                direction=direction,
                speaker=speaker,
                normalize=normalize,
                t_short=t_short,
                t_medium=t_medium,
            )
            record = {
                "time_scale": time_scale,
                "direction": direction,
                "speaker": speaker,
                "text_id": recurrence.index.tolist(),
                "score": recurrence,
            }
            results.append(record)

        results_df = pd.DataFrame.from_records(results).explode(
            ["text_id", "score"], ignore_index=True
        )
        return results_df

    def get_topic_recurrence(
        self,
        similarity: pd.DataFrame,
        time_scale: Optional[Literal["short", "medium", "long"]] = None,
        direction: Optional[Literal["forward", "backward"]] = None,
        speaker: Optional[Literal["self", "other"]] = None,
        normalize: bool = False,
        t_short: Union[Literal["n_speakers"], int] = "n_speakers",
        t_medium: int = 10,
    ) -> pd.Series:
        """
        Calculate topic recurrence metrics like short-forward-self.

        Args:
            similarity: Utterance-utterance similarity matrix, e.g.
                from ConceptSimilarityModel
            time_scale: "short", "medium" or "long.
                How far to look from the current utterance.
            direction: "forward" or "backward".
                Which direction to look from the current utterance
            speaker: "self" or "other"
                Whether to look at recurrence in utterances from
                the same speaker, or other speaker(s)
            normalize: Should the recurrence metric be normalized
                by dividing by the number of relevant utterances?
            t_short: How far to look for the "short" time_scale.
                By default this is the number of speakers in the
                conversation.
            t_medium: How far to look for the "medium" time_scale
        """
        self._validate_args("time_scale", time_scale, _TIME_SCALES)
        self._validate_args("direction", direction, _DIRECTIONS)
        self._validate_args("speaker", speaker, _SPEAKERS)

        speakers = self.data["speaker"]
        # Default for t_short is number of speakers
        if t_short == "n_speakers":
            t_short = self.n_speakers

        range_lookup = {"short": t_short, "medium": t_medium, "long": len(speakers)}
        recurrence_range = range_lookup[time_scale]

        recurrence = pd.Series(pd.NA, index=speakers.index, dtype=pd.Float64Dtype)
        for text_id in recurrence.index:
            current_speaker = speakers[text_id]
            index = speakers.index.get_loc(text_id)
            if direction == "forward":
                range_start = index + 1
                range_end = range_start + recurrence_range
            elif direction == "backward":
                range_start = index - recurrence_range
                range_end = range_start + recurrence_range
            similarity_vec = similarity.iloc[range_start:range_end, index]
            speaker_vec = speakers.iloc[range_start:range_end]
            if speaker == "self":
                speaker_indicator = speaker_vec == current_speaker
            elif speaker == "other":
                speaker_indicator = speaker_vec != current_speaker
            score = (speaker_indicator * similarity_vec).sum()
            if normalize:
                normalization_factor = speaker_indicator.sum()
                score = score / normalization_factor
            recurrence[text_id] = score

        return recurrence

    def get_grouped_recurrence(
        self, similarity: pd.DataFrame, grouping_column: str = "speaker"
    ) -> pd.DataFrame:
        """Calculate overall recurrence between groups, e.g. person-to-person
        or group-to-group.

        See https://doi.org/10.1063/1.5024809

        Args:
            similarity: matrix of turn-turn similarity scores for the conversation,
               e.g. from
        """
        groups = self.data[grouping_column].unique()
        # Group for each text/turn in the conversation
        current_group = self.data[grouping_column]

        recurrence = pd.DataFrame(
            pd.NA, index=groups, columns=groups, dtype=pd.Float64Dtype
        )
        # Get the upper triangle of the similarity matrix, we want to sum
        # across cells (i, j) where i < j
        in_upper_triangle = pd.DataFrame(
            np.triu(np.ones_like(similarity, dtype=bool), k=1),
            index=similarity.index,
            columns=similarity.index,
        )
        similarity_upper = pd.DataFrame(
            np.triu(similarity, k=1), index=similarity.index, columns=similarity.columns
        )

        pairs = permutations(groups, 2)
        for group_a, group_b in pairs:
            scores = similarity_upper.loc[
                current_group == group_a, current_group == group_b
            ]
            # Number of cells eligible to contribute is the total number with
            #   group_i = a, group_j = b, and in the upper triangle where i < j
            n_cells = in_upper_triangle.loc[scores.index, scores.columns].sum().sum()
            recurrence.loc[group_a, group_b] = n_cells * scores.values.sum()

        return recurrence


class BaseSimilarityModel:
    """Common code for similarity models, handles creation of corpus etc."""

    def __init__(
        self,
        conversation: Conversation,
        key_terms: KeyTerms = 20,
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

        if key_terms == "all":
            self.key_terms = self.binary_vectorizer.terms_list
        elif isinstance(key_terms, int):
            # Get top terms based on number of documents they appear in
            self.key_terms = self.conversation.get_most_common_terms(
                n=key_terms, method="n_turns"
            )
        elif isinstance(key_terms, list):
            # Use the user-specified terms list
            self.key_terms = key_terms
        else:
            raise ValueError(f"Unrecognized type for key_terms: {type(key_terms)}")

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

    def __init__(self, conversation: Conversation, key_terms: KeyTerms = 20):
        super().__init__(
            conversation=conversation,
            key_terms=key_terms,
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
        key_term_similarity = term_similarity_matrix.loc[self.key_terms, :]
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
    using the algorithm from https://doi.org/10/b49pvx.

    Also calculates topic recurrence statistics
    as outlined in https://doi.org/10.1109/TASL.2012.2189566

    Args:
        conversation: a Conversation object
        key_terms: By default, we use the 20 terms that occur in the most
            documents. You can pass a different number (as an integer) to
            use a different number of top terms.
            You can also set key_terms="all" to use all
            terms in the data, or manually pass a list of terms to use as
            strings.
        sentence_window_size: Number of consecutive sentences to look at
            when counting co-occurrence. Terms will be treated as co-occurring
            if they occur in the same window (within a turn).
    """

    def __init__(
        self,
        conversation: Conversation,
        key_terms: KeyTerms = 20,
        sentence_window_size: int = 3,
    ):
        super().__init__(conversation=conversation, key_terms=key_terms)
        self.sentence_window_size = sentence_window_size

    def __repr__(self):
        return (
            f"ConceptSimilarityModel(key_terms={len(self.key_terms)}, "
            + f"sentence_window_size={self.sentence_window_size}"
        )

    def __str__(self):
        return repr(self)

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
        # vectorizer.terms_list is a property that's recomputed on access,
        #   pull it out so we only access it once
        all_terms = self.binary_vectorizer.terms_list
        # Create series and dataframe to store occurrence counts
        #   and cooccurrence counts
        occurrence_series = pd.Series(
            0,
            index=all_terms,
            dtype=pd.Int64Dtype(),
        )
        cooccurrence_matrix = pd.DataFrame(
            0,
            index=all_terms,
            columns=all_terms,
            dtype=pd.Int64Dtype(),
        )

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
                    occurrence_series.at[token] += 1

                pairs = combinations(current_tokens, 2)
                for term1, term2 in pairs:
                    cooccurrence_matrix.at[term1, term2] += 1
                    cooccurrence_matrix.at[term2, term1] += 1

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

    def get_term_similarity_matrix(
        self, cooccurrence_counts: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Calculate the similarity score S(t_i, t_j) between each pair of terms t_i.

        See Angus (2012): https://doi.org/10/b49pvx
        for details of how to calculate these scores.
        """
        if cooccurrence_counts is None:
            cooccurrence_counts = self.get_cooccurrence_counts()
        contingency_counts = self._get_contingency_counts(**cooccurrence_counts)
        total_windows = cooccurrence_counts["total_windows"]
        # Convert to probabilities
        contingency_probs: Dict[Tuple[str, str], pd.DataFrame] = {
            k: v / total_windows for k, v in contingency_counts.items()
        }

        numerator = (
            contingency_probs[("i", "j")] * contingency_probs[("not_i", "not_j")]
        )
        denominator = (
            contingency_probs[("not_i", "j")] * contingency_probs[("i", "not_j")]
        )

        similarity_matrix = numerator / denominator
        # NOTE: in order to divide invalid infinite values, we substitute
        #  0.5 as the similarity value for values > 1. This occurs when
        #  a term only occurs in a single context, such that the denominator
        #  is zero
        similarity_matrix = similarity_matrix.where(similarity_matrix <= 1, 0.5)
        return similarity_matrix

    def get_concept_vectors(
        self, term_similarity_matrix: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Get the feature vectors for each document.

        Args:
            term_similarity_matrix: Term-Term similarity matrix, as
            returned by ConceptSimilarityModel.get_term_similarity_matrix.
            If this is not provided, it will be recalculated, which
            may be slow for larger datasets.
        Returns:
            A dataframe where the rows are the top terms in the corpus,
            and the columns are documents. Values are the feature value for
            that term's concept in each document.
        """
        if term_similarity_matrix is None:
            term_similarity_matrix = self.get_term_similarity_matrix()
        term_doc_df = pd.DataFrame(
            self.doc_term_matrix.T.todense(),
            index=self.binary_vectorizer.terms_list,
            columns=self.conversation.data["text_id"],
        )
        key_term_similarity = term_similarity_matrix.loc[self.key_terms, :]
        return key_term_similarity @ term_doc_df

    def get_conversation_similarity(self):
        concept_vectors = self.get_concept_vectors()
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
        top_term_doc_df = self._term_doc_df.loc[self.key_terms, :]
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
