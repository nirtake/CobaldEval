import os
import argparse
import json

from DatasetTools.src.parsing import parse_incr, OPTIONAL_TAGS
from cobaldeval.scorer import CobaldScorer
from cobaldeval.taxonomy import Taxonomy


def load_dict_from_json(json_filepath: str) -> dict:
    with open(json_filepath, "r") as file:
        data = json.load(file)
    return data


def main():
    parser = argparse.ArgumentParser(description="CoBaLD evaluation script")

    parser.add_argument(
        'test_file',
        type=str,
        help='Test file in CoBaLD format with predicted tags.'
    )
    parser.add_argument(
        'gold_file',
        type=str,
        help="Gold file in CoBaLD format with true tags."
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Output JSON file where scoring results will be written to.'
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
        nargs="?",
        type=str,
        default=OPTIONAL_TAGS,
        choices=OPTIONAL_TAGS,
        help=(
            "Tags to include in dataset, e.g. `heads deprels deps`."
            "By default, all CoBaLD tags are used."
        )
    )
    args = parser.parse_args()

    print(f"Loading taxonomy from {args.taxonomy_file}")
    semclass_taxonomy = Taxonomy(args.taxonomy_file)

    print(f"Loading lemma weights from {args.lemma_weights_file}")
    lemma_weights = load_dict_from_json(args.lemma_weights_file)

    print(f"Loading feats weights from {args.feats_weights_file}")
    feats_weights = load_dict_from_json(args.feats_weights_file)

    scorer = CobaldScorer(
        semclass_taxonomy,
        semclasses_out_of_taxonomy={None},
        lemma_weights=lemma_weights,
        feats_weights=feats_weights
    )

    print("Evaluating")
    test_sentences = parse_incr(
        args.test_file,
        args.tags or [],
        validate=False
    )
    gold_sentences = parse_incr(
        args.gold_file,
        args.tags or [],
        validate=False
    )
    scores = scorer.score_sentences(test_sentences, gold_sentences)

    with open(args.output_file, 'w') as f:
        json.dump(scores, f)

    print(f"Done, results written to {args.output_file}")


if __name__ == "__main__":
    main()