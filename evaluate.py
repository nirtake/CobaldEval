import os
import argparse
import json

from DatasetTools.parsing import parse_incr, OPTIONAL_TAGS
from src.scorer import CobaldScorer
from src.taxonomy import Taxonomy


def load_dict_from_json(json_filepath: str) -> dict:
    with open(json_filepath, "r") as file:
        data = json.load(file)
    return data


def main(
    test_filepath: str,
    gold_filepath: str,
    taxonomy_file: str,
    lemma_weights_file: str,
    feats_weights_file: str,
    optional_tags_to_parse: list[str]
) -> tuple[float]:

    print(f"Loading taxonomy from {taxonomy_file}")
    semclass_taxonomy = Taxonomy(taxonomy_file)

    print(f"Loading lemma weights from {lemma_weights_file}")
    lemma_weights = load_dict_from_json(lemma_weights_file)

    print(f"Loading feats weights from {feats_weights_file}")
    feats_weights = load_dict_from_json(feats_weights_file)

    scorer = CobaldScorer(
        semclass_taxonomy,
        semclasses_out_of_taxonomy={None},
        lemma_weights=lemma_weights,
        feats_weights=feats_weights
    )

    print("Evaluating")
    test_sentences = parse_incr(test_filepath, optional_tags_to_parse)
    gold_sentences = parse_incr(gold_filepath, optional_tags_to_parse)
    scores = scorer.score_sentences(test_sentences, gold_sentences)

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoBaLD evaluation script")

    parser.add_argument(
        'test_file',
        type=str,
        help='Test file in CoBaLD format with predicted tags.'
    )
    parser.add_argument(
        'gold_file',
        type=str,
        help=(
            "Gold file in CoBaLD format with true tags.\n"
            "For example, train.conllu."
        )
    )
    script_dir = os.path.dirname(__file__)
    parser.add_argument(
        '--taxonomy_file',
        type=str,
        help="File in CSV format with semantic class taxonomy.",
        default=os.path.join(script_dir, "res", "hyperonims_hierarchy.csv")
    )
    parser.add_argument(
        '--lemma_weights_file',
        type=str,
        help="JSON file with 'POS' -> 'lemma weight for this POS' relations.",
        default=os.path.join(script_dir, "res", "lemma_weights.json")
    )
    parser.add_argument(
        '--feats_weights_file',
        type=str,
        help="JSON file with 'grammatical category' -> 'weight of this category' relations.",
        default=os.path.join(script_dir, "res", "feats_weights.json")
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        type=str,
        default=OPTIONAL_TAGS,
        choices=OPTIONAL_TAGS,
        help=(
            "Tags to include in dataset, e.g. `heads deprels deps`."
            "By default, all CoBaLD tags are used."
        )
    )
    parser.add_argument(
        '--output_precision',
        type=int,
        help="Output scores precision",
        default=4
    )
    args = parser.parse_args()

    scores = main(
        args.test_file,
        args.gold_file,
        args.taxonomy_file,
        args.lemma_weights_file,
        args.feats_weights_file,
        args.tags,
    )

    print()
    print(f"======== SCORES =========")
    for score_name, score_value in scores.items():
        print(f"{score_name}: {score_value:.{args.output_precision}}")