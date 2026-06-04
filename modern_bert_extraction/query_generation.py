from __future__ import annotations

import collections
from pathlib import Path
import random
from typing import Sequence

from modern_bert_extraction.glue import normalize_task_name


def sentencize_wikitext(raw_path: str | Path, output_path: str | Path) -> list[str]:
    with Path(raw_path).open("r", encoding="utf-8") as handle:
        data = handle.read().strip().split("\n")

    paragraphs = [line.split(" . ") for line in data if line.strip() and line.strip()[0] != "="]
    sentences = [sentence + "." for paragraph in paragraphs for sentence in paragraph]
    text = "\n".join(sentences)
    text = text.replace(" @.@ ", ".").replace(" @-@ ", "-").replace(" ,", ",")
    text = text.replace(" \'", "\'").replace(" )", ")").replace("( ", "(")
    text = text.replace(" ;", ";")
    output_sentences = [line for line in text.split("\n") if len(line.split()) > 3]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output_sentences), encoding="utf-8")
    return output_sentences


def load_or_prepare_wikitext_sentences(raw_path: str | Path, sentences_path: str | Path) -> list[str]:
    output_path = Path(sentences_path)
    if output_path.exists():
        return [line.strip() for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sentencize_wikitext(raw_path, output_path)


def build_top_k_vocab(sentences: Sequence[str], top_k: int) -> list[str]:
    full_vocab = collections.Counter()
    for sentence in sentences:
        full_vocab.update(sentence.split())
    return [token for token, _ in full_vocab.most_common(top_k)]


def sanitize_sentence(tokens: Sequence[str], vocab: Sequence[str], vocab_set: set[str], rng: random.Random) -> list[str]:
    return [token if token in vocab_set else rng.choice(vocab) for token in tokens]


def sample_thief_sentence(
    thief_sentences: Sequence[str],
    threshold: int,
    sanitize: bool,
    vocab: Sequence[str],
    vocab_set: set[str],
    rng: random.Random,
) -> list[str]:
    sample = rng.choice(thief_sentences)
    while len(sample.split()) > threshold:
        sample = rng.choice(thief_sentences)
    tokens = sample.split()
    if sanitize:
        tokens = sanitize_sentence(tokens, vocab=vocab, vocab_set=vocab_set, rng=rng)
    return tokens


def random_length(max_query_length: int, rng: random.Random) -> int:
    return rng.randint(1, max_query_length)


def sample_random_sequence(vocab: Sequence[str], length: int, rng: random.Random) -> list[str]:
    return [rng.choice(vocab) for _ in range(length)]


def token_replace(tokens: Sequence[str], vocab: Sequence[str], num_changes: int, rng: random.Random) -> list[str]:
    output = list(tokens)
    for _ in range(num_changes):
        random_index = rng.randint(0, len(output) - 1)
        output[random_index] = rng.choice(vocab)
    return output


def resize_rows(rows: Sequence[dict[str, str]], dataset_size: int | None) -> list[dict[str, str]]:
    if dataset_size is None:
        return [dict(row) for row in rows]
    rows = list(rows)
    if not rows:
        return []
    output: list[dict[str, str]] = []
    points_remaining = dataset_size
    while points_remaining > len(rows):
        output.extend(dict(row) for row in rows)
        points_remaining -= len(rows)
    output.extend(dict(row) for row in rows[:points_remaining])
    return output


def generate_queries(
    task: str,
    scheme: str,
    base_rows: Sequence[dict[str, str]],
    thief_sentences: Sequence[str],
    vocab: Sequence[str],
    max_query_length: int,
    thief_sentence_threshold: int,
    ed1_changes: int,
    dataset_size: int | None,
    augmentations: int,
    sanitize_samples: bool,
    seed: int,
) -> list[dict[str, str]]:
    task_name = normalize_task_name(task)
    resized_rows = resize_rows(base_rows, dataset_size)
    vocab_set = set(vocab)
    rng = random.Random(seed)
    output_rows: list[dict[str, str]] = []

    for _ in range(augmentations):
        for row in resized_rows:
            updated = dict(row)
            if scheme == "random":
                if task_name == "mnli":
                    premise_tokens = sample_random_sequence(
                        vocab=vocab,
                        length=random_length(max_query_length=max_query_length, rng=rng),
                        rng=rng,
                    )
                    hypothesis_tokens = token_replace(
                        premise_tokens, vocab=vocab, num_changes=ed1_changes, rng=rng
                    )
                    updated["sentence1"] = " ".join(premise_tokens)
                    updated["sentence2"] = " ".join(hypothesis_tokens)
                else:
                    sentence_tokens = sample_random_sequence(
                        vocab=vocab,
                        length=random_length(max_query_length=max_query_length, rng=rng),
                        rng=rng,
                    )
                    updated["sentence"] = " ".join(sentence_tokens)
            elif scheme == "wiki":
                if task_name == "mnli":
                    premise_tokens = sample_thief_sentence(
                        thief_sentences=thief_sentences,
                        threshold=thief_sentence_threshold,
                        sanitize=sanitize_samples,
                        vocab=vocab,
                        vocab_set=vocab_set,
                        rng=rng,
                    )
                    hypothesis_tokens = token_replace(
                        premise_tokens, vocab=vocab, num_changes=ed1_changes, rng=rng
                    )
                    updated["sentence1"] = " ".join(premise_tokens)
                    updated["sentence2"] = " ".join(hypothesis_tokens)
                else:
                    sentence_tokens = sample_thief_sentence(
                        thief_sentences=thief_sentences,
                        threshold=thief_sentence_threshold,
                        sanitize=sanitize_samples,
                        vocab=vocab,
                        vocab_set=vocab_set,
                        rng=rng,
                    )
                    updated["sentence"] = " ".join(sentence_tokens)
            else:
                raise ValueError("Unsupported scheme: {}".format(scheme))
            output_rows.append(updated)
    return output_rows
