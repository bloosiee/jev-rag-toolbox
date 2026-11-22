"""Process-local import alias for LMFE 0.11.3 with the installed Transformers.

No site-packages files or model math are changed. LMFE imports a base tokenizer
class from an older re-export location; the class itself has not changed.
"""
import json
def complete_json(text):
 text=text.strip()
 if not text.endswith('}'):return False
 try:return isinstance(json.loads(text),dict)
 except ValueError:return False
def json_body(text,thinking=False):
 if thinking:
  if '</think>' not in text:return None
  return text.rsplit('</think>',1)[-1]
 return text
def json_stopper(tokenizer,input_length,thinking=False):
 from transformers import StoppingCriteria,StoppingCriteriaList
 class CompleteObject(StoppingCriteria):
  def __call__(self,input_ids,scores,**kwargs):
   body=json_body(tokenizer.decode(input_ids[0,input_length:],skip_special_tokens=True),thinking)
   return body is not None and complete_json(body)
 return StoppingCriteriaList([CompleteObject()])
def imports():
 import transformers.tokenization_utils as legacy
 from transformers.tokenization_utils_base import PreTrainedTokenizerBase
 if not hasattr(legacy,'PreTrainedTokenizerBase'):legacy.PreTrainedTokenizerBase=PreTrainedTokenizerBase
 from lmformatenforcer import JsonSchemaParser
 from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data,build_transformers_prefix_allowed_tokens_fn
 return JsonSchemaParser,build_token_enforcer_tokenizer_data,build_transformers_prefix_allowed_tokens_fn
_data={}
def make_json_prefix(tokenizer,schema):
 parser,build,fn=imports()
 if id(tokenizer) not in _data:_data[id(tokenizer)]=build(tokenizer)
 return fn(_data[id(tokenizer)],parser(schema))
