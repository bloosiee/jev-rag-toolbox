"""One GPU per configuration; continue the original Mentat adapter and native readout."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import traceback
from common import ROOT, atomic, check_frozen_source, compile_prompt, digest, file_sha, assert_single_task


def tokenize():
    from transformers import AutoProcessor
    from mentat_sys1.inference.backend import TorchDecisionEngine
    from safetensors.torch import load_file
    cfg = json.loads((ROOT / 'config.json').read_text())
    assert json.loads((ROOT / 'DATASET_AUDIT.json').read_text())['PASS']
    rows = json.loads((ROOT / 'data/train.json').read_text())
    assert_single_task(rows, cfg['task_kind'])
    split = json.loads((ROOT/'SPLIT_MANIFEST.json').read_text())
    assert split['question_counts'] == {'train': 697, 'val': 140, 'test': 150}
    assert file_sha(ROOT/'data/train.json') == split['data_sha256']['train']
    assert {r['task_id'] for r in rows} == set(split['question_ids']['train'])
    proc = AutoProcessor.from_pretrained(cfg['base_model'], local_files_only=True)
    codes = json.loads((Path(cfg['initial_model']) / 'decision_readout.json').read_text())['codes']
    # _prepare needs no model or readout: use the very same native prompt/token binding.
    engine = TorchDecisionEngine(torch_module=None, processor=proc, model=None, readout=None,
        codebook=tuple((x['code'], x['token_id']) for x in codes), device='cpu',
        max_length=cfg['max_length'], load_seconds=0)
    engine._verify_codebook(0)
    counts = Counter(r['kind'] for r in rows)
    output, lengths = [], []
    for r in rows:
        assert r['state_sha256'] == digest(r['state'])
        assert r['options'][r['target']]['value'] == r['label']
        prompt = compile_prompt(r)
        k = len(r['options'])
        rendered, inputs, _ = engine._prepare([], prompt, [x['code'] for x in codes[:k + 1]])
        assert set(inputs) <= {'input_ids', 'attention_mask', 'token_type_ids', 'mm_token_type_ids'}, set(inputs)
        assert all(v.shape == inputs['input_ids'].shape for v in inputs.values()), 'Expected text-only input tensors'
        ids = inputs['input_ids'][0].tolist()
        output.append({'id': r['id'], 'task_id': r['task_id'], 'kind': r['kind'], 'target': r['target'], 'count': k,
                       'input_ids': ids, 'model_inputs': {key: v[0].tolist() for key, v in inputs.items()},
                       'prompt_sha256': digest(prompt),
                       'rendered_sha256': digest(rendered),
                       'loss_weight': 1.0})
        lengths.append(len(ids))
    atomic(ROOT / 'data/tokenized.json', output)
    audit = {'PASS': True, 'n': len(rows), 'tasks': dict(counts), 'min_tokens': min(lengths),
             'max_tokens': max(lengths), 'mean_tokens': sum(lengths) / len(lengths),
             'truncated': 0, 'train_sha256': file_sha(ROOT / 'data/train.json'),
             'tokenized_sha256': file_sha(ROOT / 'data/tokenized.json'),
             'native_unknown_retained_in_prompt_excluded_from_loss_softmax': True,
             'task_weighted_epoch_objective': 'mean(loss) on this single task only',
             'task_kind': cfg['task_kind'],
             'all_examples_used_once_per_epoch': True}
    audit['split_manifest_sha256'] = file_sha(ROOT/'SPLIT_MANIFEST.json')
    atomic(ROOT / 'TOKENIZATION_AUDIT.json', audit)
    return audit


def run(job, smoke=False):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    cfg = json.loads((ROOT / 'config.json').read_text())
    run = ROOT / ('smoke' if smoke else 'training') / job['id']
    run.mkdir(parents=True, exist_ok=True)
    assert not (run / 'COMPLETE.json').exists(), 'Do not rerun completed configurations'
    check_frozen_source()
    for name, sha in json.loads((ROOT / 'MODEL_INPUT_MANIFEST.json').read_text()).items():
        assert file_sha(name) == sha, name
    ta = json.loads((ROOT / 'TOKENIZATION_AUDIT.json').read_text())
    assert ta['PASS'] and file_sha(ROOT / 'data/tokenized.json') == ta['tokenized_sha256']
    rows = json.loads((ROOT / 'data/tokenized.json').read_text())
    assert_single_task(rows, cfg['task_kind'])
    split = json.loads((ROOT/'SPLIT_MANIFEST.json').read_text())
    assert file_sha(ROOT/'SPLIT_MANIFEST.json') == ta['split_manifest_sha256']
    assert {r['task_id'] for r in rows} == set(split['question_ids']['train'])
    assert not ({r['task_id'] for r in rows} &
                (set(split['question_ids']['val']) | set(split['question_ids']['test'])))
    if smoke:
        # Longest example of each label exercises this task's memory and mask.
        rows = [max((r for r in rows if r['target'] == k), key=lambda r: len(r['input_ids']))
                for k in sorted({r['target'] for r in rows})]
    seed = cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    started = time.time()
    initial = Path(cfg['initial_model'])
    checkpoint = run / 'latest'
    resumed = checkpoint.exists()
    load_dir = checkpoint if resumed else initial
    atomic(run / 'status.json', {'stage': 'loading', 'job': job, 'resumed': resumed, 'time': time.time()})
    processor = AutoProcessor.from_pretrained(cfg['base_model'], local_files_only=True)
    bare = Qwen3_5ForConditionalGeneration.from_pretrained(cfg['base_model'],
               local_files_only=True, dtype=torch.bfloat16).to('cuda:0')
    model = PeftModel.from_pretrained(bare, str(load_dir), is_trainable=True, local_files_only=True)
    adapter = model.peft_config['default']
    assert adapter.r == 64 and adapter.lora_alpha == 128 and adapter.lora_dropout == 0
    w = load_file(str(load_dir / 'decision_readout.safetensors'))['weight']
    head = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=torch.float32, device='cuda:0')
    with torch.no_grad():
        head.weight.copy_(w.to('cuda:0').float())
    head.requires_grad_(True)
    base = model.get_base_model()
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    base.enable_input_require_grads()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in n for n, _ in trainable)
    frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    params = [p for _, p in trainable] + [head.weight]
    optimizer = torch.optim.AdamW([{'params': [p for _, p in trainable], 'weight_decay': .01},
                                   {'params': [head.weight], 'weight_decay': 0.0}], lr=job['lr'])
    order = list(range(len(rows))); random.Random(seed).shuffle(order)
    batch = 2 if smoke else job['batch']
    updates = math.ceil(len(rows) / batch)
    warmup = max(1, math.ceil(updates * .03))
    step, losses, seen = 0, 0.0, 0
    if resumed:
        saved = torch.load(checkpoint / 'optimizer.pt', map_location='cpu', weights_only=False)
        assert saved['job'] == job and saved['tokenized_sha256'] == ta['tokenized_sha256']
        assert saved['order'] == order
        optimizer.load_state_dict(saved['optimizer'])
        step, losses, seen = saved['step'], saved['losses'], saved['seen']
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
    atomic(run / 'setting.json', {'job': job, 'task_kind': cfg['task_kind'], 'seed': seed, 'epoch': 1,
           'micro_batch': 1, 'effective_batch': batch, 'rank': 64, 'lora_alpha': 128,
           'initial_adapter': str(initial), 'initial_readout': str(initial / 'decision_readout.safetensors'),
           'loss': 'hard-label CE + 0.5 * sum((softmax(logits)-onehot)^2)',
           'task_loss': 'single-task mean; no other task data or checkpoint is used',
           'trainable': 'existing LoRA + native float32 decision readout; base frozen',
           'scheduler': 'cosine; 3% warmup', 'max_grad_norm': 1.0,
           'unknown_in_prompt': True, 'softmax': 'given options only',
           'precision': 'BF16; reduced-precision reduction OFF, TF32 OFF',
           'checkpoint_policy': 'latest optimizer state every25updates + final adapter',
           'selection': cfg['selection_rule'],
           'trainable_parameters': sum(p.numel() for p in params), 'torch': torch.__version__,
           'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'smoke': smoke})

    def logits(row):
        inputs = {key: torch.tensor([v], device='cuda:0') for key, v in row['model_inputs'].items()}
        assert inputs['input_ids'][0].tolist() == row['input_ids']
        h = base.model(**inputs, use_cache=False).last_hidden_state[0, -1]
        z = head(h.float())[:row['count']]
        assert torch.isfinite(z).all(), row['id']
        return z

    def save(step):
        assert shutil.disk_usage(ROOT).free > 3 * 1024**3, 'Insufficient free disk for atomic checkpoint; keep last checkpoint'
        temp = run / 'checkpoint.tmp'
        assert not temp.exists(), 'Interrupted checkpoint temp exists: inspect before recovery'
        model.save_pretrained(temp, safe_serialization=True, save_embedding_layers=False)
        save_file({'weight': head.weight.detach().contiguous()}, str(temp / 'decision_readout.safetensors'))
        shutil.copy2(initial / 'decision_readout.json', temp / 'decision_readout.json')
        torch.save({'optimizer': optimizer.state_dict(), 'step': step, 'job': job, 'order': order,
                    'tokenized_sha256': ta['tokenized_sha256'], 'losses': losses, 'seen': seen,
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                    'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()}, temp / 'optimizer.pt')
        old = run / 'previous'
        if old.exists():
            shutil.rmtree(old)
        if checkpoint.exists():
            checkpoint.rename(old)
        temp.rename(checkpoint)
        if old.exists():
            shutil.rmtree(old)

    model.train(); head.train()
    for begin in range(step * batch, len(order), batch):
        chunk = order[begin:begin + batch]
        factor = (step + 1) / warmup if step < warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, updates - warmup)))
        for group in optimizer.param_groups:
            group['lr'] = job['lr'] * factor
        optimizer.zero_grad(set_to_none=True)
        local_loss = 0.0
        for ix in chunk:
            row = rows[ix]
            z = logits(row)
            target = torch.tensor([row['target']], device='cuda:0')
            p = z.softmax(-1)
            onehot = F.one_hot(target[0], row['count']).float()
            ce = F.cross_entropy(z[None], target)
            brier = ((p - onehot) ** 2).sum()
            loss = (ce + .5 * brier) * row['loss_weight']
            (loss / len(chunk)).backward()
            local_loss += float(loss.detach())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        assert math.isfinite(grad_norm) and grad_norm > 0, 'Missing or invalid trainable gradients'
        optimizer.step()
        step += 1; seen += len(chunk); losses += local_loss
        status = {'stage': 'training', 'step': step, 'total_steps': updates, 'seen': seen,
                  'total_examples': len(rows), 'last_batch_loss': local_loss / len(chunk),
                  'mean_loss': losses / seen, 'learning_rate': optimizer.param_groups[0]['lr'],
                  'grad_norm': grad_norm, 'elapsed_seconds_this_process': time.time() - started,
                  'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1e9, 'time': time.time()}
        atomic(run / 'status.json', status)
        with (run / 'train.jsonl').open('a') as f:
            f.write(json.dumps(status) + '\n')
        print(json.dumps(status), flush=True)
        if step % 25 == 0 or step == updates:
            save(step)
    assert seen == len(rows)
    assert all(p._version == frozen_versions[n] for n, p in model.named_parameters() if n in frozen_versions)
    # Reload saved tensors into the actual active modules and verify real logits,
    # not just file existence. These probe rows are training rows, not a benchmark.
    model.eval(); head.eval()
    with torch.inference_mode():
        before = [logits(r).cpu() for r in rows[:2]]
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, load_file(str(checkpoint / 'adapter_model.safetensors')))
    with torch.no_grad():
        head.weight.copy_(load_file(str(checkpoint / 'decision_readout.safetensors'))['weight'].to('cuda:0'))
    with torch.inference_mode():
        after = [logits(r).cpu() for r in rows[:2]]
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    audit = {'PASS': True, 'job': job, 'seen': seen, 'step': step, 'frozen_base_unchanged': True,
             'checkpoint_reload_logits_exactly_equal': True, 'final_model': str(checkpoint),
             'input_manifest_sha256': file_sha(ROOT / 'MODEL_INPUT_MANIFEST.json'),
             'tokenized_sha256': ta['tokenized_sha256'], 'time': time.time()}
    # Completed configurations need the adapter/readout for evaluation, not Adam
    # moments. Keep optimizer/RNG only while a job is resumable, preventing 18
    # independent final models from filling the shared data disk with unused state.
    audit['final_optimizer_removed_after_verified_reload'] = False
    atomic(run / 'COMPLETE.json', audit)
    try:
        (checkpoint / 'optimizer.pt').unlink()
        audit['final_optimizer_removed_after_verified_reload'] = True
    except OSError as exc:
        audit['optimizer_cleanup_error'] = str(exc)
    atomic(run / 'COMPLETE.json', audit)
    atomic(run / 'status.json', {'stage': 'COMPLETE', **audit})
    return audit


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--job'); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--tokenize', action='store_true'); args = ap.parse_args()
    if args.tokenize:
        print(json.dumps(tokenize()))
    else:
        job = json.loads(args.job)
        try:
            run(job, args.smoke)
        except Exception:
            folder = ROOT / ('smoke' if args.smoke else 'training') / job['id']
            atomic(folder / 'FAILED.json', {'traceback': traceback.format_exc(), 'time': time.time()})
            raise
