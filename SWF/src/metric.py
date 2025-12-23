from collections import Counter
import re
from typing import List, Union

common_words = {
    "in", "on", "at", "to", "for", "with", "by", "from", "about",
    "a", "an", "the",
    "it", "they", "we", "you", "he", "she", "i", "me", "my", "mine", "ours", "us", "your", "yours", "his", "hers",
    "their", "theirs",
    "and", "or", "but", "because", "if", "then", "than", "as",
    "is", "are", "was", "were", "do", "does", "did", "have", "has", "had", "having", "be", "been", "being",
    "not", "no", "nor", "none",
    "what", "where", "when", "who", "why", "how", "which", "whom", "whose",
    ".", ",", "!", "?", ";", ":", "-", "(", ")", "[", "]", "{", "}", "\"", "'", "...", "--", "/", "\\", "|", "<", ">",
    "=", "+", "*", "&", "^", "%", "$", "#", "@", "~", "`",
    "of", "that", "this", "these", "those", "such", "there", "here", "all", "any", "both", "each", "few", "more",
    "some", "most", "other", "another", "every", "either", "neither"
}


def rounder(num):
    return round(num, 3)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return ' '.join(text.split())

    # def remove_punc(text):
    #     return re_punc.sub(' ', text)  # convert punctuation to spaces

    def remove_punc(text):
        # exclude = set(string.punctuation) # TODO
        exclude = common_words
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# ==================================================================================================
# EM Score
# ==================================================================================================


def eval_em(preds: List[str], answers: List[List[str]]):
    scores = []
    for pred, answer in zip(preds, answers):
        pred = normalize_answer(pred)
        if any([normalize_answer(ans) == pred for ans in answer]):  # TODO
            scores.append(1)
        else:
            scores.append(0)
    return rounder(sum(scores) * 100 / len(scores))


# ==================================================================================================
# Acc Score
# ==================================================================================================


def eval_acc(preds, answers):
    scores = []
    for pred, answer in zip(preds, answers):
        pred = normalize_answer(pred)
        if any([normalize_answer(ans) in pred for ans in answer]):  # TODO
            scores.append(1)
        else:
            scores.append(0)
    return rounder(sum(scores) * 100 / len(scores))


# ==================================================================================================
# F1 Score
# ==================================================================================================

def prec_recall_f1_score(pred_items, gold_items):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.

    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values

    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def f1_score(guess, answers):
    if guess is None or answers is None:
        return 0
    g_tokens = normalize_answer(guess).split()
    scores = [
        prec_recall_f1_score(g_tokens, normalize_answer(a).split()) for a in answers
    ]
    return max(f1 for p, r, f1 in scores)


def eval_f1(preds, answers):
    f1 = 0.
    for predict, answer in zip(preds, answers):
        # answer = answer.split('\t')
        f1 += f1_score(predict, answer)
    return rounder(f1 * 100 / len(answers))


class Reward:
    MATH = ['math', 'Algebra', 'Counting & Probability',
            'Geometry', 'Intermediate Algebra', 'Prealgebra',
            'Number Theory', 'Precalculus']
    QA = ['hotpotqa', 'qa', 'musique', 'musiqueqa']

    @staticmethod
    def reward_fn(domain, predict: str, answer: Union[str, List]):
        if domain in Reward.MATH:
            return Reward.eval_math(predict, answer)
        elif domain in Reward.QA:
            return Reward.eval_acc(predict, answer)
        else:
            raise ValueError(f'Unknown domain: {domain}')

    @staticmethod
    def eval_math(predict: str, answer: str):
        pattern = r"\\boxed\{(.*?)\}"

        def _parse(content):
            items = re.findall(pattern, content)
            return items[-1] if len(items) else content

        if '\\boxed' in predict:
            predict = _parse(predict)
        if '\\boxed' in answer:
            answer = _parse(answer)

        return (predict == answer) * 100

    @staticmethod
    def eval_f1(predict: str, answer: Union[str, List]):
        _answer = answer if isinstance(answer, list) else [answer]
        a = eval_f1([predict], [_answer])
        # print(a)
        return a

    @staticmethod
    def eval_acc(predict: str, answer: Union[str, List]):
        _answer = answer if isinstance(answer, list) else [answer]
        return eval_acc([predict], [_answer])

    @staticmethod
    def eval_em(predict: str, answer: Union[str, List]):
        _answer = answer if isinstance(answer, list) else [answer]
        return eval_em([predict], [_answer])



