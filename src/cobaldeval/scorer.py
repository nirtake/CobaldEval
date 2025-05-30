import sys
import json
from collections import defaultdict

from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

from .taxonomy import Taxonomy


NULL = "#NULL"


def is_range_id(x: str) -> bool:
    try:
        a, b = x.split('-')
        return a.isdecimal() and b.isdecimal()
    except ValueError:
        return False

def parse_feats(feats: str) -> dict:
    field_dict = {}
    for item in feats.split('|'):
        key, value = item.split('=')
        field_dict[key] = value
    return field_dict

def jaccard_score(a: set, b: set, aggregate_fn: callable = len) -> float:
    return aggregate_fn(a & b) / aggregate_fn(a | b)


class CobaldScorer:
    def __init__(
        self,
        taxonomy: Taxonomy = None,
        semclasses_out_of_taxonomy: set = {},
        lemma_weights: dict[str, float] = None,
        feats_weights: dict[str, float] = None
    ):
        self.taxonomy = taxonomy
        self.semclasses_out_of_taxonomy = set(semclasses_out_of_taxonomy)
        self.lemma_weights = lemma_weights
        self.feats_weights = feats_weights

    def score_lemma(self, test: str, gold: str) -> float:
        if test["lemma"] is None or gold["lemma"] is None:
            return float(test["lemma"] == gold["lemma"])

        normalize = lambda word: word.lower().replace('ё', 'е')
        score = normalize(test["lemma"]) == normalize(gold["lemma"])

        # 'lemma' scores are weighted, because we want immutable parts of speech
        # to affect lemmatization score less than mutable ones (which are, obviously,
        # harder to lemmatize).
        if self.lemma_weights and gold["upos"]:
            score *= self.lemma_weights[gold["upos"]]
        return score

    def score_upos(self, test: dict, gold: dict) -> float:
        return float(test["upos"] == gold["upos"])

    def score_xpos(self, test: dict, gold: dict) -> float:
        return float(test["xpos"] == gold["xpos"])

    def score_feats(self, test: dict, gold: dict) -> float:
        if not test["feats"] or not gold["feats"]:
            return test["feats"] == gold["feats"]

        assert isinstance(test["feats"], str) and isinstance(gold["feats"], str)
        test_feats = parse_feats(test["feats"])
        gold_feats = parse_feats(gold["feats"])

        if len(test_feats) == 0 or len(gold_feats) == 0:
            return float(test_feats == gold_feats)

        test_feats_items = set(test_feats.items())
        gold_feats_items = set(gold_feats.items())

        sum_weights = lambda feats: sum(self.feats_weights[cat] for cat, _ in feats)
        return jaccard_score(
            test_feats_items,
            gold_feats_items,
            sum_weights if self.feats_weights else len
        )

    def score_head(self, test: dict, gold: dict) -> float:
        return float(test["head"] == gold["head"])

    def score_deprel(self, test: dict, gold: dict) -> float:
        return float(test["head"] == gold["head"] and test["deprel"] == gold["deprel"])

    def score_deps(self, test: dict, gold: dict, use_labels: bool) -> float:
        if not test["deps"] or not gold["deps"]:
            return test["deps"] == gold["deps"]

        assert isinstance(test["deps"], str) and isinstance(gold["deps"], str)
        test_deps = json.loads(test["deps"])
        gold_deps = json.loads(gold["deps"])

        if len(test_deps) == 0 or len(gold_deps) == 0:
            return float(test_deps == gold_deps)

        test_arcs = set(test_deps.items() if use_labels else test_deps.keys())
        gold_arcs = set(gold_deps.items() if use_labels else gold_deps.keys())
        return jaccard_score(test_arcs, gold_arcs)

    def score_misc(self, test: str, gold: str) -> float:
        return float(test["misc"] == gold["misc"])

    def score_deepslot(self, test: dict, gold: dict) -> float:
        return float(test["deepslot"] == gold["deepslot"])

    def score_semclass(self, test: dict, gold: dict) -> float:
        # If gold semclass is out of taxomony, simply compare strings
        if gold["semclass"] in self.semclasses_out_of_taxonomy:
            return float(test["semclass"] == gold["semclass"])

        if not self.taxonomy:
            raise RuntimeError("Taxonomy is missing")
        if not self.taxonomy.has_semclass(gold["semclass"]):
            raise ValueError(f"Unknown gold semclass encountered: {gold["semclass"]}")
        if not self.taxonomy.has_semclass(test["semclass"]):
            return 0.

        semclasses_distance = self.taxonomy.calc_path_length(test["semclass"], gold["semclass"])
        # If distance is 0 then test_semclass == gold_semclass, so score is 1.
        # If they are different, the penalty is proportional to their distance.
        # If they are in different trees, then distance is inf, so score is 0.
        return 1 / (1 + semclasses_distance)

    def score_sentences(self, test_sentences, gold_sentences) -> tuple[float]:
        # name, column, scoring function
        scoring_info = [
            ("lemma weight. accuracy", "lemma", self.score_lemma),
            ("upos accuracy", "upos", self.score_upos),
            ("xpos accuracy", "xpos", self.score_xpos),
            ("feats weight. jaccard", "feats", self.score_feats),
            ("UAS", "head", self.score_head),
            ("LAS", "deprel", self.score_deprel),
            ("EUAS", "deps", lambda t, g: self.score_deps(t, g, use_labels=False)),
            ("ELAS", "deps", lambda t, g: self.score_deps(t, g, use_labels=True)),
            ("misc accuracy", "misc", self.score_misc),
            ("deepslot accuracy", "deepslot", self.score_deepslot),
            ("semclass similarity", "semclass", self.score_semclass)
        ]
        # Grammatical scores.
        # Use lists and 'np.sum' instead of float summation (+=) for numerical stability.
        test_scores = defaultdict(list)
        # Some scores are weighted and can be greater/less than 1, so if we want final
        # averaged scores to be in [0..1] range, we have to scale them by maximum possible
        # (gold) scores.
        gold_scores = defaultdict(list)

        test_nulls, gold_nulls = [], []

        for test_sentence, gold_sentence in tqdm(
            zip(test_sentences, gold_sentences, strict=True), file=sys.stdout
        ):
            # Filter out range tokens.
            test_tokens = [
                token for token in test_sentence["tokens"]
                if not is_range_id(token["id"])
            ]
            gold_tokens = [
                token for token in gold_sentence["tokens"]
                if not is_range_id(token["id"])
            ]
            # Test and gold sentence may have different lengths due to null tokens, so align them.
            try:
                test_tokens_aligned, gold_tokens_aligned = self._align_sentences(
                    test_tokens, gold_tokens
                )
            except Exception as e:
                raise RuntimeError(f"Sentence {gold_sentence.get('sent_id', '?')}:\n{e}")
            
            test_columns = {tag for token in test_tokens for tag in token}
            gold_columns = {tag for token in gold_tokens for tag in token}
            columns = test_columns | gold_columns

            for test_token, gold_token in zip(
                test_tokens_aligned, gold_tokens_aligned, strict=True
            ):
                for score_name, column, score_fn in scoring_info:
                    # Allow partial evaluation (i.e. evaluation on subset of columns)
                    if column not in columns:
                        continue

                    # Token mismatch
                    if test_token is None or gold_token is None:
                        test_score = 0
                        gold_score = 1
                    else:
                        test_score = score_fn(test_token, gold_token)
                        gold_score = score_fn(gold_token, gold_token)
                        assert 0. <= test_score <= 1.0
                        assert 0. <= gold_score <= 1.0
                    test_scores[score_name].append(test_score)
                    gold_scores[score_name].append(gold_score)

                    # Track nulls separately
                    test_nulls.append(test_token["word"] == NULL if test_token else 0)
                    gold_nulls.append(gold_token["word"] == NULL if gold_token else 0)

        average_scores = {
            name: np.mean(test_scores[name]) / np.mean(gold_scores[name])
            for name in test_scores.keys()
        }
        average_scores["null f1"] = f1_score(gold_nulls, test_nulls, zero_division=1.0)
        average_scores["total"] = np.mean([score for score in average_scores.values()])

        for score in average_scores.values():
            assert 0. <= score <= 1.

        return average_scores

    @staticmethod
    def _align_sentences(lhs: list[dict], rhs: list[dict]) -> tuple[list]:
        """
        Aligns two sequences of tokens. None token is inserted where needed.
        Example:
        >>> test_words = ["How", "did", "this", "#NULL", "happen"]
        >>> gold_words = ["How", "#NULL", "did", "this", "happen"]
        >>> _align_sentences(test_words, gold_words)
        (["How",    None, "did", "this", "#NULL", "happen"],
         ["How", "#NULL", "did", "this",    None, "happen"])
        """
        lhs_aligned, rhs_aligned = [], []

        i, j = 0, 0

        i_exhausted, j_exhausted = False, False
        while not i_exhausted and not j_exhausted:
            if lhs[i]["word"] == rhs[j]["word"]:
                lhs_aligned.append(lhs[i])
                rhs_aligned.append(rhs[j])
                i += 1
                j += 1
            elif lhs[i]["word"] == NULL and rhs[j]["word"] != NULL:
                lhs_aligned.append(lhs[i])
                rhs_aligned.append(None)
                i += 1
            elif lhs[i]["word"] != NULL and rhs[j]["word"] == NULL:
                lhs_aligned.append(None)
                rhs_aligned.append(rhs[j])
                j += 1
            else:
                raise RuntimeError(
                    f"Test-gold words mismatch: {lhs[i]['word']} != {rhs[j]['word']}"
                )
            if len(lhs) <= i:
                i_exhausted = True
                i = len(lhs) - 1
            if len(rhs) <= j:
                j_exhausted = True
                j = len(rhs) - 1

        assert len(lhs_aligned) == len(rhs_aligned)
        return lhs_aligned, rhs_aligned