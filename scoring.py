# Portions of this file are adapted from M3DocRAG's evaluation code
# (src/m3docrag/datasets/m3_docvqa/evaluate.py and common_utils.py),
# Copyright 2024 Bloomberg Finance L.P., licensed under the Apache License,
# Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0), itself adapted
# from https://github.com/allenai/multimodalqa/blob/master/baselines/evaluate.py
#
# Vendored here (instead of importing the `m3docrag` package) so this
# lightweight results viewer doesn't need to install torch / transformers /
# colpali just to reuse ~150 lines of pure-python answer scoring.

import re
import string
from typing import List, Set, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from w2n import word_to_num

MULTI_HOP_QUESTION_TYPES = [
    # text as second hop
    "Compare(Compose(TableQ,ImageQ),Compose(TableQ,TextQ))",
    "Compose(TextQ,ImageListQ)",
    "Compose(TextQ,TableQ)",
    # table as second hop
    "Compare(Compose(TableQ,ImageQ),TableQ)",
    "Compare(TableQ,Compose(TableQ,TextQ))",
    "Compose(TableQ,ImageListQ)",
    "Compose(TableQ,TextQ)",
    "Intersect(ImageListQ,TableQ)",
    "Intersect(TableQ,TextQ)",
    # image as second hop
    "Compose(ImageQ,TableQ)",
    "Compose(ImageQ,TextQ)",
    "Intersect(ImageListQ,TextQ)",
]


def _remove_articles(text: str) -> str:
    return re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


_EXCLUDE = set(string.punctuation)


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _is_word_number(text: str) -> bool:
    try:
        word_to_num(text)
        return True
    except ValueError:
        return False


def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in _EXCLUDE)
    return text


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    if _is_word_number(text):
        return str(float(word_to_num(text)))
    return text


def _tokenize(text: str) -> List[str]:
    return re.split(" |-", text)


def _normalize_answer(text: str) -> str:
    parts = [
        _white_space_fix(_remove_articles(_normalize_number(_remove_punc(token.lower()))))
        for token in _tokenize(text)
    ]
    return " ".join(p for p in parts if p.strip()).strip()


def _answer_to_bags(answer: Union[str, List[str], Tuple[str, ...]]):
    raw_spans = answer if isinstance(answer, (list, tuple)) else [answer]
    normalized_spans, token_bags = [], []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = {w for w in gold_bag if _is_number(w)}
    predicted_numbers = {w for w in predicted_bag if _is_number(w)}
    return (not gold_numbers) or bool(gold_numbers & predicted_numbers)


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag & predicted_bag)
    precision = 1.0 if not predicted_bag else intersection / len(predicted_bag)
    recall = 1.0 if not gold_bag else intersection / len(gold_bag)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    scores = np.zeros([len(gold), len(predicted)])
    for gi, gold_item in enumerate(gold):
        for pi, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gi, pi] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)
    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, col in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, col])
    return max_scores


def list_em(predicted, gold) -> float:
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        return 1.0
    return 0.0


def list_f1(predicted, gold) -> float:
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    return round(float(np.mean(f1_per_bag)), 2)
