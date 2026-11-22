"""Pinned data/label contract. No teacher annotations are inserted into prompts."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False))
    t.replace(p)


SPECS = {
    'sufficiency': (
        'Do the retrieved documents contain enough evidence to answer the ORIGINAL user question completely? '
        'Judge the supplied evidence only, not prior model knowledge or the quality of a draft answer.',
        [('sufficient', 'The supplied evidence is sufficient: stop further retrieval and generate the final answer from this evidence.'),
         ('insufficient', 'Essential evidence is missing: continue investigation before generating the final answer.')]),
    'necessity': (
        'Should the agent execute this proposed follow-up question to improve its answer to the original user question? '
        'Judge the value of investigating missing facts using only evidence already obtained.',
        [('execute', 'Investigate: this follow-up could establish facts needed for the original answer that the retrieved evidence does not yet establish.'),
         ('skip_covered', 'Skip: the retrieved evidence already establishes the facts this follow-up asks for.'),
         ('skip_unhelpful', 'Skip: answering this follow-up would not help resolve the original question.')])}


def example(kind, state, label, uid, task_id, source):
    q, options = SPECS[kind]
    values = [v for v, _ in options]
    assert label in values
    # Gold answer, decomposition, teacher explanation and condition_id are metadata,
    # never part of the state passed to the student.
    forbidden = {'gold_answer', 'supporting_facts', 'label', 'teacher', 'decomposition', 'sufficient'}
    assert not (set(state) & forbidden)
    return {'id': uid, 'task_id': task_id, 'kind': kind, 'state': state,
            'state_sha256': digest(state), 'question': q,
            'options': [{'value': v, 'description': d} for v, d in options],
            'label': label, 'target': values.index(label), 'source': source}


def compile_prompt(row):
    from mentat_sys1.inference.contracts import ChoiceField, Option
    from mentat_sys1.inference.scoring import compile_question
    field = ChoiceField(id=row['kind'], type='choice', question=row['question'],
                        options=[Option(**o) for o in row['options']])
    header, choices, texts = compile_question(field, row['state'])
    assert len(choices) == len(row['options']) + 1  # native unknown retained in prompt
    return header + '\n'.join(f'{chr(65 + i)}: {s}' for i, s in enumerate(texts))


def check_frozen_source():
    m = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
    for relative, sha in m.items():
        assert file_sha(ROOT / relative) == sha, relative


def assert_single_task(rows, kind):
    assert kind in SPECS and rows
    assert {row['kind'] for row in rows} == {kind}, 'Cross-task training data is forbidden'
    assert all(row.get('loss_weight', 1.0) == 1.0 for row in rows), 'Single task uses unit sample weight'
