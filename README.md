## Install

```
pip install cobaldeval
```

## Usage

```
$ cobaldeval -h
usage: cobaldeval [-h]
                  [--taxonomy_file TAXONOMY_FILE]
                  [--lemma_weights_file LEMMA_WEIGHTS_FILE]
                  [--feats_weights_file FEATS_WEIGHTS_FILE]
                  [--tags [{lemma,upos,xpos,feats,head,deprel,deps,misc,deepslot,semclass}]]
                  test_file gold_file output_file

CoBaLD evaluation script

positional arguments:
  test_file             Test file in CoBaLD format with predicted tags.
  gold_file             Gold file in CoBaLD format with true tags.
  output_file           Output JSON file where scoring results will be written to.

options:
  -h, --help            show this help message and exit
  --taxonomy_file TAXONOMY_FILE
                        File in CSV format with semantic class taxonomy.
  --lemma_weights_file LEMMA_WEIGHTS_FILE
                        JSON file with 'POS' -> 'lemma weight for this POS' relations.
  --feats_weights_file FEATS_WEIGHTS_FILE
                        JSON file with 'grammatical category' -> 'weight of this category' relations.
  --tags [{lemma,upos,xpos,feats,head,deprel,deps,misc,deepslot,semclass}]
                        Tags to include in dataset, e.g. `head deprel deps`.
                        By default, all CoBaLD tags are used.
```

**Example**

```
cobaldeval predictions.conllu reference.conllu results.json --tags head deprel deps
```
