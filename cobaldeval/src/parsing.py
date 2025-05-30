import json


ROOT_HEAD = 0
NULL = "#NULL"

ID = "id"
WORD = "word"
LEMMA = "lemma"
UPOS = "upos"
XPOS = "xpos"
FEATS = "feats"
HEAD = "head"
DEPREL = "deprel"
DEPS = "deps"
MISC = "misc"
DEEPSLOT = "deepslot"
SEMCLASS = "semclass"

MANDATORY_TAGS = [ID, WORD]
OPTIONAL_TAGS = [LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC, DEEPSLOT, SEMCLASS]

# ========================= Field level =========================

def parse_nullable(field: str) -> str | None:
    if field in ["", "_"]:
        return None
    return field


def is_null_index(x: str) -> bool:
    try:
        a, b = x.split('.')
        return a.isdecimal() and b.isdecimal()
    except ValueError:
        return False

def is_range_index(x: str) -> bool:
    try:
        a, b = x.split('-')
        return a.isdecimal() and b.isdecimal()
    except ValueError:
        return False

def parse_id(field: str) -> str:
    if not field.isdecimal() and not is_null_index(field) and not is_range_index(field):
        raise ValueError(f"Incorrect token id: {field}.")
    return field


def parse_word(field: str) -> str:
    if field == "":
        raise ValueError(f"Token form cannot be empty.")
    return field


def parse_joint_field(field: str, inner_sep: str, outer_sep: str) -> dict[str, str]:
    """
    Parse joint field with two kinds of separators (inner and outer) and return a dict.

    E.g.
    >>> parse_joint_field("Mood=Ind|Number=Sing|Person=3", '=', '|')
    {'Mood': 'Ind', 'Number': 'Sing', 'Person': '3'}
    >>> parse_joint_field("26:conj|18.1:advcl:while", ':', '|')
    {'26': 'conj', '18.1': 'advcl:while'}
    """
    if inner_sep not in field:
        raise ValueError(f"Non-empty field {field} must contain {inner_sep} separator.")

    field_dict = {}
    for item in field.split(outer_sep):
        key, value = item.split(inner_sep, 1)
        if key in field_dict:
            raise ValueError(f"Field {field} has duplicate key {key}.")
        field_dict[key] = value
    return field_dict


def parse_feats(field: str) -> str | None:
    if field in ["", "_"]:
        return None

    # validate feats are parsable
    parse_joint_field(field, inner_sep='=', outer_sep='|')
    
    return field


def parse_head(field: str) -> int | None:
    if field in ["", "_"]:
        return None

    if not field.isdecimal():
        raise ValueError(f"Non-empty head must be a decimal number, not {field}.")

    head = int(field)
    if head < 0:
        raise ValueError(f"Non-empty head must be a positive integer, not {head}.")

    return head


def parse_deps(field: str) -> str | None:
    if field in ["", "_"]:
        return None

    token_deps = parse_joint_field(field, inner_sep=':', outer_sep='|')

    # Validate deps.
    if len(token_deps) == 0:
        raise ValueError(f"Empty deps are not allowed: {field}.")
    for head in token_deps:
        if not head.isdecimal() and not is_null_index(head):
            raise ValueError(f"Deps head must be either integer or float (x.1), not {head}.")
    return json.dumps(token_deps)

# ========================= Token level =========================

Token = dict[str, str]

def validate_null_token(token: Token):
    if token[WORD] != NULL:
        raise ValueError(f"Null token must have #NULL form, not {token[WORD]}.")
    if HEAD in token and token[HEAD] is not None:
        raise ValueError(f"Null token must have no head, but found {token[HEAD]}.")
    if DEPREL in token and token[DEPREL] is not None:
        raise ValueError(f"Null token must have no deprel, but found {token[DEPREL]}.")
    if MISC in token and token[MISC] != 'ellipsis':
        raise ValueError(f"Null token must have 'ellipsis' misc, not {token[MISC]}.")

def validate_range_token(token: Token):
    if LEMMA in token and token[LEMMA] != '_':
        raise ValueError(f"Range token lemma must be _, but found {token[LEMMA]}.")
    if UPOS in token and token[UPOS] is not None:
        raise ValueError(f"Range token upos must be _, but found {token[UPOS]}.")
    if XPOS in token and token[XPOS] is not None:
        raise ValueError(f"Range token xpos must be _, but found {token[XPOS]}.")
    if FEATS in token and token[FEATS] is not None:
        raise ValueError(f"Range token feats must be _, but found {token[FEATS]}.")
    if HEAD in token and token[HEAD] is not None:
        raise ValueError(f"Range token head must be _, but found {token[HEAD]}.")
    if DEPREL in token and token[DEPREL] is not None:
        raise ValueError(f"Range token deprel must be _, but found {token[DEPREL]}.")
    if MISC in token and token[MISC] is not None:
        raise ValueError(f"Range token misc must be _, but found {token[MISC]}.")
    if DEEPSLOT in token and token[DEEPSLOT] is not None:
        raise ValueError(f"Range token deepslot must be _, but found {token[DEEPSLOT]}.")
    if SEMCLASS in token and token[SEMCLASS] is not None:
        raise ValueError(f"Range token semclass must be _, but found {token[SEMCLASS]}.")

def validate_regular_token(token: Token):
    if HEAD in token and token[HEAD] == token[ID]:
        raise ValueError(f"Self-loop detected in head.")
    if DEPS in token and token[DEPS] is not None:
        for head in json.loads(token[DEPS]):
            if head == token[ID]:
                raise ValueError(f"Self-loop detected in deps. head: {head}, id: {token[ID]}")

def validate_token(token: Token):
    if token[ID].isdecimal():
        validate_regular_token(token)
    elif is_range_index(token[ID]):
        validate_range_token(token)
    elif is_null_index(token[ID]):
        validate_null_token(token)
    else:
        raise ValueError(f"Incorrect token id: {token[ID]}.")


def parse_token(line: str, tags_to_parse: list[str], validate: bool) -> Token:
    try:
        fields = [field.strip() for field in line.split("\t")]
        if len(fields) < len(tags_to_parse):
            raise SyntaxError(
                f"line {line} must have at leasts {len(tags_to_parse)} fields, not {len(fields)}"
            )

        token = {}
        if ID in tags_to_parse:
            token[ID] = parse_id(fields[0])
        if WORD in tags_to_parse:
            token[WORD] = parse_word(fields[1])
        if LEMMA in tags_to_parse:
            token[LEMMA] = fields[2] if fields[2] != "" else None
        if UPOS in tags_to_parse:
            token[UPOS] = parse_nullable(fields[3])
        if XPOS in tags_to_parse:
            token[XPOS] = parse_nullable(fields[4])
        if FEATS in tags_to_parse:
            token[FEATS] = parse_feats(fields[5])
        if HEAD in tags_to_parse:
            token[HEAD] = parse_head(fields[6])
        if DEPREL in tags_to_parse:
            token[DEPREL] = parse_nullable(fields[7])
        if DEPS in tags_to_parse:
            token[DEPS] = parse_deps(fields[8])
        if MISC in tags_to_parse:
            token[MISC] = parse_nullable(fields[9])
        if DEEPSLOT in tags_to_parse:
            token[DEEPSLOT] = parse_nullable(fields[10])
        if SEMCLASS in tags_to_parse:
            token[SEMCLASS] = parse_nullable(fields[11])

        if validate:
            validate_token(token)

    except Exception as e:
        raise RuntimeError(f"Validation failed on token {fields[0]}:\n{e}")

    return token

# ========================= Sentence level =========================

Sentence = dict[str, list[str]]

def validate_sentence(sentence: Sentence):
    # Ensure the sentence is not empty
    num_words = len(sentence[WORD])
    if num_words == 0:
        raise ValueError("Empty sentence.")

    for tag in OPTIONAL_TAGS:
        if tag in sentence and len(sentence[tag]) != num_words:
            raise RuntimeError(f"Field {tag} has inconsistent length.")

    # Validate sentence heads and ids agreement.
    ids = set(sentence[ID])
    int_ids = {int(idtag) for idtag in sentence[ID] if idtag.isdecimal()}

    contiguous_int_ids = set(range(min(int_ids), max(int_ids) + 1))
    if int_ids != contiguous_int_ids:
        raise RuntimeError(f"ids are not contiguous: {sorted(int_ids)}.")

    if HEAD in sentence:
        # Check 1
        has_labels = any(head is not None for head in sentence[HEAD])
        roots_count = sum(head == ROOT_HEAD for head in sentence[HEAD])
        if has_labels and roots_count != 1:
            raise RuntimeError(
                f"There must be exactly one ROOT in a sentence, but found {roots_count}."
            )
        # Check 2
        sentence_heads = set(
            head
            for head in sentence[HEAD]
            if head is not None and head != ROOT_HEAD
        )
        excess_heads = sentence_heads - int_ids
        if excess_heads:
            raise RuntimeError(
                f"Heads are inconsistent with sentence ids. "
                f"Excessive heads: {excess_heads}."
            )

    if DEPS in sentence:
        sentence_deps_heads = {
            head
            for deps in sentence[DEPS] if deps is not None
            for head in json.loads(deps) if head != str(ROOT_HEAD)
        }
        excess_deps_heads = sentence_deps_heads - ids
        if excess_deps_heads:
            raise RuntimeError(
                f"Deps heads are inconsistent with sentence ids. "
                f"Excessive heads: {excess_deps_heads}."
            )


def parse_sentence(
    token_lines: list[str],
    metadata: dict,
    tags_to_parse: list[str],
    validate: bool
) -> Sentence:
    try:
        tokens = [
            parse_token(token_line, tags_to_parse, validate)
            for token_line in token_lines
        ]
        if validate:
            sentence = {
                tag: [token[tag] for token in tokens]
                for tag in tags_to_parse
            }
            validate_sentence(sentence)

    except Exception as e:
        raise RuntimeError(f"Validation failed on sentence {metadata.get('sent_id', None)}:\n{e}")

    return metadata | {"tokens": tokens}


def parse_incr(filepath: str, optional_tags_to_parse: list[str], validate: bool):
    """
    Generator that parses a CoNLL-U Plus file in CoBaLD format and yields one sentence at a time.
    
    Each sentence is represented as a dictionary:
    {
        "sent_id": <sent_id from metadata or None>,
        "text": <text from metadata or None>,
        ID: list of sentence tokens ids,
        WORD: list of tokens forms,
        ...
        <optional tags to parse, specified in `optional_tags_to_parse` argument>
        ...
    }

    The input file must have metadata lines starting with '#' (e.g., "# sent_id = 1")
    and token lines. Sentence blocks must be separated by blank lines.
    """
    metadata = {}
    token_lines = []

    for tag in optional_tags_to_parse:
        if tag not in OPTIONAL_TAGS:
            raise ValueError(f"Invalid tags: {tag}, must be one of {OPTIONAL_TAGS}")

    tags_to_parse = MANDATORY_TAGS + optional_tags_to_parse

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                # End of a sentence block.
                if token_lines:
                    yield parse_sentence(token_lines, metadata, tags_to_parse, validate)
                    metadata = {}
                    token_lines = []
                continue
            if line.startswith("#"):
                # Process metadata lines.
                if "=" in line:
                    # Remove the '#' and split by the first '='.
                    key, value = line[1:].split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key in ("sent_id", "text"):
                        metadata[key] = value
                continue
            # Accumulate token lines.
            token_lines.append(line)

    # Yield any remaining sentence at the end of the file.
    if token_lines:
        yield parse_sentence(token_lines, metadata, tags_to_parse, validate)