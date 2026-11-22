"""The controller and retriever cannot access gold labels or future evidence."""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re

STOPWORDS = set('a an the is are was were be been being of in on at for to from by with and or that this '
                'which what who when where how did does do have has had as than'.split())


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def tokens(text):
    return [w for w in re.findall(r'\w+', text.lower()) if w not in STOPWORDS]


def rank(task):
    """No answer, supporting-fact annotations, question type or model scores used."""
    if set(task) != {'id', 'question', 'documents'}:
        raise ValueError('Only public task schema is allowed')
    docs = task['documents']
    words = [tokens(d['title'] + ' ' + ''.join(d['sentences'])) for d in docs]
    counts = list(map(Counter, words))
    df = Counter(t for w in words for t in set(w))
    avg = sum(map(len, words)) / len(words)
    query = tokens(task['question'])
    scores = []
    for w, c in zip(words, counts):
        score = 0.0
        for t in query:
            idf = math.log(1 + (len(words) - df[t] + 0.5) / (df[t] + 0.5))
            tf = c[t]
            score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * len(w) / avg))
        scores.append(score)
    return sorted(range(len(docs)), key=lambda i: (-scores[i], i)), scores
