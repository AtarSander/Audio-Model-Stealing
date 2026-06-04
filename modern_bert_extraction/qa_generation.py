from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
from typing import Any

from loguru import logger
import numpy as np


def normalize_wikitext_line(line: str) -> str:
    """Port the WikiText paragraph cleanup used by the original QA extraction code."""
    text = line.strip()
    replacements = {
        " @.@ ": ".",
        " @-@ ": "-",
        " ,": ",",
        " '": "'",
        " )": ")",
        "( ": "(",
        " ;": ";",
        " .": ".",
        " :": ":",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def load_or_prepare_wikitext_paragraphs(
    *,
    raw_path: str | Path,
    paragraphs_path: str | Path,
    min_words: int = 21,
) -> list[str]:
    output_path = Path(paragraphs_path)
    if output_path.exists():
        paragraphs = [
            line.strip()
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        logger.info("Loaded {} WikiText paragraphs from {}", len(paragraphs), output_path)
        return paragraphs

    input_path = Path(raw_path)
    if not input_path.exists():
        raise FileNotFoundError(
            "WikiText raw file not found at {}. Run `make download_wikitext_hf` first.".format(
                input_path
            )
        )

    paragraphs: list[str] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("="):
            continue
        paragraph = normalize_wikitext_line(stripped)
        if len(paragraph.split()) >= min_words:
            paragraphs.append(paragraph)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(paragraphs) + "\n", encoding="utf-8")
    logger.info("Prepared {} WikiText paragraphs at {}", len(paragraphs), output_path)
    return paragraphs


def get_squad_question_starters() -> tuple[list[str], np.ndarray]:
    tokens = [
        "Why",
        "On",
        "Along",
        "During",
        "At",
        "A",
        "For",
        "According",
        "What",
        "How",
        "Who",
        "When",
        "In",
        "Which",
        "Where",
        "The",
        "To",
        "From",
        "By",
        "what",
        "After",
        "Whose",
        "What's",
    ]
    return tokens, np.ones(len(tokens), dtype=np.float64) / len(tokens)


def get_boolq_question_starters() -> tuple[list[str], np.ndarray]:
    tokens = ["is", "can", "does", "are", "do", "did", "was", "has", "will", "the", "have"]
    return tokens, np.ones(len(tokens), dtype=np.float64) / len(tokens)


def build_thief_vocab(thief_paragraphs: list[str]) -> tuple[list[str], np.ndarray, list[int]]:
    counts: Counter[str] = Counter()
    paragraph_lengths: list[int] = []
    for paragraph in thief_paragraphs:
        tokens = paragraph.split()
        paragraph_lengths.append(len(tokens))
        counts.update(tokens)

    if not counts:
        raise ValueError("Cannot build thief vocabulary from an empty paragraph list.")

    sorted_items = counts.most_common()
    tokens = [token for token, _ in sorted_items]
    frequencies = np.array([count for _, count in sorted_items], dtype=np.float64)
    probabilities = frequencies / frequencies.sum()
    return tokens, probabilities, paragraph_lengths


def _sample_paragraph_length(np_rng: np.random.RandomState) -> int:
    return int(np_rng.randint(75, 500))


def uniform_sampling_paragraph(
    vocab: list[str],
    *,
    rng: random.Random,
    np_rng: np.random.RandomState,
    paragraph_length: int | None = None,
) -> str:
    length = paragraph_length or _sample_paragraph_length(np_rng)
    return " ".join(rng.choice(vocab) for _ in range(length))


def frequency_sampling_paragraph(
    vocab: list[str],
    probabilities: np.ndarray,
    *,
    rng: random.Random,
    np_rng: np.random.RandomState,
    paragraph_length: int | None = None,
) -> str:
    length = paragraph_length or _sample_paragraph_length(np_rng)
    counts = np_rng.multinomial(length, probabilities)
    sampled_words: list[str] = []
    for word, count in zip(vocab, counts):
        sampled_words.extend([word] * int(count))
    rng.shuffle(sampled_words)
    return " ".join(sampled_words)


def choose_random_question(
    paragraph: str,
    *,
    sampling_scheme: str,
    np_rng: np.random.RandomState,
) -> str:
    words = paragraph.split()
    if not words:
        return ""

    question_length = int(np_rng.randint(5, 15))
    if sampling_scheme == "anchor_gaussian":
        anchor = int(np_rng.randint(0, len(words)))
        sampled_indexes = [
            min(max(int(index), 0), len(words) - 1)
            for index in np_rng.normal(anchor, 5, question_length)
        ]
        return " ".join(words[index] for index in sampled_indexes)

    if "random" in sampling_scheme:
        sampled_indexes = np_rng.randint(0, len(words), question_length)
        return " ".join(words[int(index)] for index in sampled_indexes)

    raise ValueError("Unsupported question sampling scheme: {}".format(sampling_scheme))


def postprocess_question(
    question: str,
    q_tokens: list[str],
    q_probs: np.ndarray,
    *,
    sampling_scheme: str,
    rng: random.Random,
    np_rng: np.random.RandomState,
) -> str:
    if "uniform" in sampling_scheme:
        question_starter = rng.choice(q_tokens)
    else:
        question_starter = np_rng.choice(q_tokens, p=q_probs)
    return "{} {}?".format(question_starter, question).strip()


def postprocess_boolq_question(
    question: str,
    q_tokens: list[str],
    q_probs: np.ndarray,
    *,
    sampling_scheme: str,
    rng: random.Random,
    np_rng: np.random.RandomState,
) -> str:
    if "uniform" in sampling_scheme:
        question_starter = rng.choice(q_tokens)
    else:
        question_starter = np_rng.choice(q_tokens, p=q_probs)
    return "{} {}".format(question_starter, question).strip().lower()


def _paragraph_from_scheme(
    *,
    para_scheme: str,
    original_paragraph: str,
    thief_paragraphs: list[str],
    vocab: list[str],
    probabilities: np.ndarray,
    paragraph_lengths: list[int],
    index: int,
    rng: random.Random,
    np_rng: np.random.RandomState,
) -> str:
    original_length = len(original_paragraph.split())
    if para_scheme == "original_para":
        return original_paragraph.strip()
    if para_scheme == "thief_para":
        return rng.choice(thief_paragraphs).strip()
    if para_scheme == "uniform_sampling":
        return uniform_sampling_paragraph(vocab, rng=rng, np_rng=np_rng).strip()
    if para_scheme == "uniform_sampling_orig_length":
        return uniform_sampling_paragraph(
            vocab,
            rng=rng,
            np_rng=np_rng,
            paragraph_length=original_length,
        ).strip()
    if para_scheme == "uniform_sampling_sample_length":
        paragraph_length = paragraph_lengths[index % len(paragraph_lengths)]
        return uniform_sampling_paragraph(
            vocab,
            rng=rng,
            np_rng=np_rng,
            paragraph_length=paragraph_length,
        ).strip()
    if para_scheme == "frequency_sampling":
        return frequency_sampling_paragraph(
            vocab,
            probabilities,
            rng=rng,
            np_rng=np_rng,
        ).strip()
    if para_scheme == "frequency_sampling_orig_length":
        return frequency_sampling_paragraph(
            vocab,
            probabilities,
            rng=rng,
            np_rng=np_rng,
            paragraph_length=original_length,
        ).strip()
    if para_scheme == "frequency_sampling_sample_length":
        paragraph_length = paragraph_lengths[index % len(paragraph_lengths)]
        return frequency_sampling_paragraph(
            vocab,
            probabilities,
            rng=rng,
            np_rng=np_rng,
            paragraph_length=paragraph_length,
        ).strip()
    raise ValueError("Unsupported paragraph sampling scheme: {}".format(para_scheme))


def paragraph_scheme_for_extraction_scheme(scheme: str) -> str:
    normalized = scheme.lower()
    if normalized == "wiki":
        return "thief_para"
    if normalized == "random":
        return "frequency_sampling_sample_length"
    raise ValueError("Unsupported extraction scheme: {}".format(scheme))


def _select_question_ids(
    question_ids: list[str],
    *,
    fraction: float | None,
    dataset_size: int | None,
    rng: random.Random,
) -> set[str]:
    selected_ids = list(question_ids)
    rng.shuffle(selected_ids)
    if fraction is not None:
        selected_count = int(len(selected_ids) * fraction)
        selected_ids = selected_ids[:selected_count]
    if dataset_size is not None:
        selected_ids = selected_ids[: int(dataset_size)]
    return set(selected_ids)


def generate_squad_queries(
    *,
    source_data: dict[str, Any],
    thief_paragraphs: list[str],
    scheme: str,
    question_sampling_scheme: str,
    augmentations: int,
    seed: int,
    fraction: float | None = None,
    dataset_size: int | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    para_scheme = paragraph_scheme_for_extraction_scheme(scheme)
    q_tokens, q_probs = get_squad_question_starters()
    vocab, probabilities, paragraph_lengths = build_thief_vocab(thief_paragraphs)

    all_question_ids = [
        qa["id"]
        for article in source_data["data"]
        for paragraph in article["paragraphs"]
        for qa in paragraph["qas"]
    ]
    selected_ids = _select_question_ids(
        all_question_ids,
        fraction=fraction,
        dataset_size=dataset_size,
        rng=rng,
    )

    generated = {
        "version": source_data.get("version", "1.1"),
        "data": [],
    }
    paragraph_index = 0
    total_qas = 0
    for augmentation_index in range(int(augmentations)):
        for article in source_data["data"]:
            generated_article = {
                "title": article.get("title", ""),
                "paragraphs": [],
            }
            for paragraph in article["paragraphs"]:
                selected_qas = [qa for qa in paragraph["qas"] if qa["id"] in selected_ids]
                if not selected_qas:
                    continue
                original_paragraph = paragraph["context"]
                para_text = _paragraph_from_scheme(
                    para_scheme=para_scheme,
                    original_paragraph=original_paragraph,
                    thief_paragraphs=thief_paragraphs,
                    vocab=vocab,
                    probabilities=probabilities,
                    paragraph_lengths=paragraph_lengths,
                    index=paragraph_index,
                    rng=rng,
                    np_rng=np_rng,
                )
                paragraph_index += 1
                words = para_text.split()
                if not words:
                    continue
                generated_paragraph = {
                    "context": para_text,
                    "qas": [],
                }
                for qa in selected_qas:
                    question = choose_random_question(
                        para_text,
                        sampling_scheme=question_sampling_scheme,
                        np_rng=np_rng,
                    )
                    question = postprocess_question(
                        question,
                        q_tokens,
                        q_probs,
                        sampling_scheme=question_sampling_scheme,
                        rng=rng,
                        np_rng=np_rng,
                    )
                    answer_text = words[0]
                    generated_paragraph["qas"].append(
                        {
                            "answers": [{"answer_start": 0, "text": answer_text}],
                            "question": question,
                            "id": "{}-aug{}".format(qa["id"], augmentation_index),
                            "is_impossible": False,
                        }
                    )
                    total_qas += 1
                if generated_paragraph["qas"]:
                    generated_article["paragraphs"].append(generated_paragraph)
            if generated_article["paragraphs"]:
                generated["data"].append(generated_article)

    logger.info(
        "Generated {} SQuAD-style extraction queries with scheme {}",
        total_qas,
        scheme,
    )
    return generated


def generate_boolq_queries(
    *,
    source_rows: list[dict[str, Any]],
    thief_paragraphs: list[str],
    scheme: str,
    question_sampling_scheme: str,
    augmentations: int,
    seed: int,
    dataset_size: int | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    para_scheme = paragraph_scheme_for_extraction_scheme(scheme)
    q_tokens, q_probs = get_boolq_question_starters()
    vocab, probabilities, paragraph_lengths = build_thief_vocab(thief_paragraphs)

    rows = list(source_rows)
    rng.shuffle(rows)
    if dataset_size is not None:
        rows = rows[: int(dataset_size)]

    generated_rows: list[dict[str, Any]] = []
    paragraph_index = 0
    for _ in range(int(augmentations)):
        for row in rows:
            para_text = _paragraph_from_scheme(
                para_scheme=para_scheme,
                original_paragraph=row["passage"],
                thief_paragraphs=thief_paragraphs,
                vocab=vocab,
                probabilities=probabilities,
                paragraph_lengths=paragraph_lengths,
                index=paragraph_index,
                rng=rng,
                np_rng=np_rng,
            )
            paragraph_index += 1
            question = choose_random_question(
                para_text,
                sampling_scheme=question_sampling_scheme,
                np_rng=np_rng,
            )
            question = postprocess_boolq_question(
                question,
                q_tokens,
                q_probs,
                sampling_scheme=question_sampling_scheme,
                rng=rng,
                np_rng=np_rng,
            )
            generated_rows.append(
                {
                    "title": row.get("title", "empty"),
                    "passage": para_text,
                    "question": question,
                    "answer": False,
                }
            )

    logger.info(
        "Generated {} BoolQ extraction queries with scheme {}",
        len(generated_rows),
        scheme,
    )
    return generated_rows


def combine_squad_predictions_with_queries(
    *,
    generated_queries: dict[str, Any],
    predictions: dict[str, str],
) -> dict[str, Any]:
    distilled = {
        "version": generated_queries.get("version", "1.1"),
        "data": [],
    }
    skipped = 0
    kept = 0
    for article in generated_queries["data"]:
        output_article = {
            "title": article.get("title", ""),
            "paragraphs": [],
        }
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            output_paragraph = {
                "context": context,
                "qas": [],
            }
            for qa in paragraph["qas"]:
                answer_text = predictions.get(qa["id"], "").strip()
                if not answer_text:
                    skipped += 1
                    continue
                answer_start = context.find(answer_text)
                if answer_start < 0:
                    first_word = answer_text.split()[0] if answer_text.split() else ""
                    answer_start = context.find(first_word) if first_word else -1
                    if answer_start >= 0:
                        answer_text = first_word
                if answer_start < 0:
                    skipped += 1
                    continue
                output_qa = deepcopy(qa)
                output_qa["answers"] = [
                    {
                        "answer_start": answer_start,
                        "text": answer_text,
                    }
                ]
                output_qa["is_impossible"] = False
                output_paragraph["qas"].append(output_qa)
                kept += 1
            if output_paragraph["qas"]:
                output_article["paragraphs"].append(output_paragraph)
        if output_article["paragraphs"]:
            distilled["data"].append(output_article)

    logger.info(
        "Combined victim SQuAD predictions with generated queries: kept={}, skipped={}",
        kept,
        skipped,
    )
    return distilled


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
