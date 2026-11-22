"""One prefill per decision; no extra hidden-state extraction before inference."""
import json
from pathlib import Path
import time

from .protocol import atomic, digest, answer_prompt, decision_prompt, spec


class Runtime:
    def __init__(self, config, output):
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        from peft import PeftModel
        from safetensors.torch import load_file
        from mentat_sys1.inference.backend import TorchDecisionEngine
        self.torch = torch
        self.config = config
        self.output = Path(output)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        t = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(config['answer_model'], local_files_only=True)
        self.tok = self.processor.tokenizer
        bare = Qwen3_5ForConditionalGeneration.from_pretrained(
            config['answer_model'], local_files_only=True, dtype=torch.bfloat16).to('cuda:0').eval()
        check = self.tok('A short calibration-free base parity preflight.', return_tensors='pt').to('cuda:0')
        with torch.inference_mode():
            before = bare.model(**check, use_cache=False).last_hidden_state[0, -1].float().cpu()
        adapter = Path(config['jev_model'])
        self.model = PeftModel.from_pretrained(bare, str(adapter), is_trainable=False,
                                               local_files_only=True).eval()
        with self.model.disable_adapter(), torch.inference_mode():
            after = self.model.get_base_model().model(**check, use_cache=False).last_hidden_state[0, -1].float().cpu()
        if not torch.equal(before, after):
            raise RuntimeError('Disabled adapter does not reproduce bare base hidden state')
        weight = load_file(str(adapter / 'decision_readout.safetensors'), device='cuda:0')['weight'].float()
        head = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, device='cuda:0', dtype=torch.float32)
        head.weight.data.copy_(weight)
        head.weight.requires_grad_(False)
        codes = json.loads((adapter / 'decision_readout.json').read_text())['codes']
        self.engine = TorchDecisionEngine(torch_module=torch, processor=self.processor,
            model=self.model, readout=head.eval(),
            codebook=tuple((r['code'], r['token_id']) for r in codes),
            device='cuda:0', max_length=16384, load_seconds=0)
        self.engine._verify_codebook(0)
        torch.cuda.synchronize()
        atomic(self.output / 'runtime.json', {
            'load_seconds_excluded': time.perf_counter() - t,
            'bare_vs_disabled_adapter_hidden_equal': True,
            'torch': torch.__version__, 'device': torch.cuda.get_device_name(0),
            'precision': 'BF16, BF16 reduced precision reduction disabled',
            'answer_model': config['answer_model'], 'jev_model': config['jev_model'],
        })

    def hooks(self, module):
        torch = self.torch
        pending, events = [], []
        def pre(m, inputs):
            t = torch.cuda.Event(enable_timing=True)
            t.record()
            pending.append(t)
        def post(m, inputs, output):
            t = torch.cuda.Event(enable_timing=True)
            t.record()
            events.append((pending.pop(), t))
        return events, [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]

    def decision(self, evidence, arm, record, kind):
        torch = self.torch
        request = {'state': evidence, 'kind': kind, 'arm': arm}
        start = time.perf_counter()
        options = spec(kind)[1]
        if arm == 'qwen' and kind == 'necessity':
            options = [options[0],
                ('skip_covered', 'Skip: the retrieved evidence already establishes the facts this follow-up asks for.'),
                ('skip_unhelpful', 'Skip: answering this follow-up would not help resolve the original question.')]
        prompt = decision_prompt(evidence, kind, options)
        k=len(options)
        labels=[chr(65+i) for i in range(k+1)]
        rendered, raw_inputs, token_ids = self.engine._prepare([], prompt, labels)
        inputs = {k: v.to('cuda:0') for k, v in raw_inputs.items()}
        module = self.engine.readout if arm == 'jev' else self.model.get_base_model().get_output_embeddings()
        events, handles = self.hooks(module)
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        try:
            if arm == 'jev':
                with torch.inference_mode():
                    h = self.model.get_base_model().model(**inputs, use_cache=False).last_hidden_state[0, -1]
                    logits = self.engine.readout(h.float())[:k+1]
                output_tokens, raw, forwards = 0, None, len(events)
            elif arm == 'qwen':
                # Same rendered input as Jev; one generated legal letter, no reasoning or JSON overhead.
                def allowed(batch, ids):
                    return token_ids[:k]
                with self.model.disable_adapter(), torch.inference_mode():
                    output = self.model.generate(**inputs, max_new_tokens=1, do_sample=False,
                        use_cache=True, logits_to_keep=1, prefix_allowed_tokens_fn=allowed,
                        return_dict_in_generate=True, output_scores=True)
                logits = output.scores[0][0, token_ids[:k]].float()
                output_tokens = int(output.sequences.shape[1] - inputs['input_ids'].shape[1])
                raw = self.tok.decode(output.sequences[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                forwards = len(events)
            else:
                raise ValueError(arm)
            torch.cuda.synchronize()
            end = time.perf_counter()
        finally:
            for handle in handles:
                handle.remove()
        raw_logits = logits[:k].double()
        temperature = self.config['temperatures'][kind] if arm == 'jev' else 1.0
        p = torch.softmax(raw_logits / temperature, dim=-1).cpu().tolist()
        uncalibrated = torch.softmax(raw_logits, dim=-1).cpu().tolist()
        index = max(range(k), key=p.__getitem__)
        if arm == 'qwen' and raw != labels[index]:
            raise RuntimeError('Constrained generation disagrees with candidate argmax')
        if forwards != 1:
            raise RuntimeError(f'Decision used {forwards} head calls instead of one')
        result = {
            'kind': 'decision', 'arm': arm, 'request': request, 'request_sha256': digest(request),
            'prompt': prompt, 'rendered_sha256': digest(rendered),
            'input_tokens': int(inputs['input_ids'].shape[1]), 'output_tokens': output_tokens,
            'wall_seconds': end - start, 'model_seconds': end - forward_start,
            'head_gpu_ms': sum(a.elapsed_time(b) for a, b in events), 'head_calls': forwards,
            'raw': raw, 'probabilities': p, 'uncalibrated_probabilities': uncalibrated,
            'raw_option_logits': raw_logits.cpu().tolist(), 'temperature': temperature,
            'calibration_sha256': self.config['calibration_sha256'] if arm == 'jev' else None,
            'model_inputs_sha256': digest({key:value[0].tolist() for key,value in raw_inputs.items()}),
            'native_mm_token_type_ids_retained': 'mm_token_type_ids' in raw_inputs,
            'choice': options[index][0], 'module':kind,
            'native_unknown_logit': float(logits[k]) if arm == 'jev' else None,
            'native_unknown_excluded_from_given_options': True,
            'cache_hit': False,
        }
        if arm == 'jev':
            from .confidence_policy import route
            threshold = self.config['confidence_threshold']
            low_policy = self.config['low_confidence_policy']
            routed = route(kind, p, [x[0] for x in options], threshold, low_policy)
            if routed['choice'] is None:
                raise RuntimeError('This run authorizes conservative continuation, not Qwen fallback')
            result.update(routed, confidence_threshold=threshold, low_confidence_policy=low_policy)
            result['request'].update(confidence_threshold=threshold, low_confidence_policy=low_policy, temperature=temperature, calibration_sha256=self.config['calibration_sha256'])
            result['request_sha256'] = digest(result['request'])
        atomic(record, result)
        return result

    def generate_text(self, prompt, kind, limit, record, schema=None):
        """Plain greedy Qwen generation for both candidate generation and branch execution."""
        torch = self.torch
        start = time.perf_counter()
        rendered = self.tok.apply_chat_template([{'role': 'user', 'content': prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tok(rendered, return_tensors='pt').to('cuda:0')
        if inputs.input_ids.shape[1] + limit > 16384:
            raise RuntimeError('Generation input exceeds limit; no silent truncation')
        events, handles = self.hooks(self.model.get_base_model().get_output_embeddings())
        kwargs={}
        if schema is not None:
            from .format_compat import make_json_prefix, json_stopper
            kwargs['prefix_allowed_tokens_fn']=make_json_prefix(self.tok,schema)
            kwargs['stopping_criteria']=json_stopper(self.tok,int(inputs.input_ids.shape[1]),False)
        torch.cuda.synchronize()
        fstart = time.perf_counter()
        try:
            with self.model.disable_adapter(), torch.inference_mode():
                output = self.model.generate(**inputs, do_sample=False, use_cache=True,
                    max_new_tokens=limit, logits_to_keep=1, **kwargs)
            torch.cuda.synchronize()
            end = time.perf_counter()
        finally:
            for handle in handles:
                handle.remove()
        tail = output[0, inputs.input_ids.shape[1]:]
        raw = self.tok.decode(tail, skip_special_tokens=True)
        json_complete=False
        if schema is not None:
            try:
                json_complete=isinstance(json.loads(raw.strip()),dict)
            except ValueError:
                pass
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        result = {'kind': kind, 'model': 'qwen3.5-4b-base-adapter-disabled',
            'prompt': prompt, 'request_sha256': digest(prompt), 'rendered_sha256': digest(rendered),
            'input_tokens': int(inputs.input_ids.shape[1]), 'output_tokens': len(tail),
            'wall_seconds': end - start, 'model_seconds': end - fstart,
            'head_gpu_ms': sum(a.elapsed_time(b) for a,b in events), 'head_calls': len(events),
            'raw': raw, 'truncated': int(tail[-1]) not in eos and not json_complete, 'cache_hit': False,
            'json_schema_constrained': schema is not None}
        atomic(record, result)
        if result['truncated']:
            raise RuntimeError(f'{kind} hit output cap; inspect instead of silently parsing a partial result')
        return result

    def answer(self, evidence, record):
        torch = self.torch
        start = time.perf_counter()
        prompt = answer_prompt(evidence)
        rendered = self.tok.apply_chat_template([{'role': 'user', 'content': prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tok(rendered, return_tensors='pt').to('cuda:0')
        if inputs.input_ids.shape[1] > 16384:
            raise RuntimeError('Answer input too long; do not silently truncate')
        events, handles = self.hooks(self.model.get_base_model().get_output_embeddings())
        torch.cuda.synchronize()
        fstart = time.perf_counter()
        try:
            with self.model.disable_adapter(), torch.inference_mode():
                output = self.model.generate(**inputs, do_sample=False, use_cache=True,
                    max_new_tokens=self.config['max_answer_tokens'], logits_to_keep=1)
            torch.cuda.synchronize()
            end = time.perf_counter()
        finally:
            for handle in handles:
                handle.remove()
        tail = output[0, inputs.input_ids.shape[1]:]
        raw = self.tok.decode(tail, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        result = {
            'kind': 'answer', 'model': 'qwen3.5-4b-base-adapter-disabled', 'state': evidence,
            'request_sha256': digest(evidence), 'prompt': prompt, 'rendered_sha256': digest(rendered),
            'input_tokens': int(inputs.input_ids.shape[1]), 'output_tokens': len(tail),
            'wall_seconds': end - start, 'model_seconds': end - fstart,
            'head_gpu_ms': sum(a.elapsed_time(b) for a, b in events), 'head_calls': len(events),
            'raw': raw, 'answer': raw.strip(), 'truncated': int(tail[-1]) not in eos,
            'cache_hit': False,
        }
        atomic(record, result)
        return result
