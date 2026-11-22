"""A confidence threshold is explicit and separate from option argmax."""
import math

def route(kind,probabilities,options,threshold,low_policy):
 if kind not in ['sufficiency','necessity']:raise ValueError(kind)
 if not 0<threshold<=1:raise ValueError('Invalid threshold')
 if len(probabilities)!=len(options) or not probabilities:raise ValueError('Invalid choice space')
 if any(not math.isfinite(p) or not 0<=p<=1 for p in probabilities) or abs(sum(probabilities)-1)>1e-8:
  raise ValueError('Invalid probabilities')
 top=max(range(len(probabilities)),key=probabilities.__getitem__)
 confidence=probabilities[top];raw=options[top]
 if confidence>=threshold:return {'selected_by':'jev','choice':raw,'confidence':confidence,'raw_jev_choice':raw}
 if low_policy=='qwen_fallback':return {'selected_by':'qwen_fallback','choice':None,'confidence':confidence,'raw_jev_choice':raw}
 if low_policy=='conservative_continue':
  return {'selected_by':'conservative_default','choice':'continue' if kind=='sufficiency' else 'execute','confidence':confidence,'raw_jev_choice':raw}
 raise ValueError('Unknown low-confidence policy')
