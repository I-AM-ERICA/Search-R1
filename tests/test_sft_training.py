import json
from pathlib import Path
import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from policy_r1.config import Config
from policy_r1.models import prepare_backbone,action_log_probs
from policy_r1.sft_training import action_positions,inspect_bundle,sha,CODE_FILES,FILES
from policy_r1.policy_sft import fingerprint,content_digest,compile_record
from policy_r1.inference import POLICY_SYSTEM_PROMPT
from test_policy_search_v2 import demonstration
from test_rollout import CharacterTokenizer


def bundle(tmp_path):
    root=Path(__file__).resolve().parents[1]
    data=tmp_path/'data';data.mkdir();tokenizer=tmp_path/'tokenizer';tokenizer.mkdir()
    (tokenizer/'fixture.json').write_text('{}')
    row,snapshots=demonstration()
    cfg=Config(max_seq_length=2000,max_prompt_tokens=200,max_new_tokens=180,max_observation_tokens=400,
               observation_strategy='focused_v1',model_revision='a'*40)
    compiled=compile_record(row,snapshots,CharacterTokenizer(),cfg,POLICY_SYSTEM_PROMPT)
    (data/'train.jsonl').write_text(json.dumps(compiled)+'\n')
    (data/'val.jsonl').write_text('');(data/'test.jsonl').write_text('')
    (data/'reviewed-records.json').write_text(json.dumps([row]))
    settings={'model_environment':cfg.to_dict(),'data_gates':{'min_train_records':120,'min_val_records':30,'min_supported_two_search_records':30}}
    report=dict(tokenized=True,config=settings,ready_for_training=False,reviewed_data_sha256=fingerprint([row]),
                prompt_sha256=fingerprint(POLICY_SYSTEM_PROMPT),output_sha256={n:sha(data/n) for n in FILES},
                compiler_code_sha256={n:sha(root/n) for n in CODE_FILES},tokenizer_sha256={'fixture.json':sha(tokenizer/'fixture.json')})
    (data/'report.json').write_text(json.dumps(report))
    return data,tokenizer,root


def test_preflight_enforces_actual_data_gates_and_detects_mutation(tmp_path):
    data,tok,root=bundle(tmp_path)
    result,*_=inspect_bundle(data,tok,root)
    assert result['ready_for_training'] is False and result['counts']['train']==1
    with (data/'train.jsonl').open('a') as f:f.write(' ')
    with pytest.raises(ValueError,match='hash mismatch'):inspect_bundle(data,tok,root)


def test_mask_violation_is_rejected_even_with_updated_file_hash(tmp_path):
    data,tok,root=bundle(tmp_path)
    row=json.loads((data/'train.jsonl').read_text());row['labels'][0]=row['input_ids'][0]
    (data/'train.jsonl').write_text(json.dumps(row)+'\n')
    report=json.loads((data/'report.json').read_text());report['output_sha256']['train.jsonl']=sha(data/'train.jsonl')
    (data/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='-100'):inspect_bundle(data,tok,root)


def test_actor_only_action_loss_matches_shifted_cross_entropy():
    torch.manual_seed(7)
    base=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2))
    actor=prepare_backbone(base,Config(load_in_4bit=False,gradient_checkpointing=False,lora_rank=2),'CAUSAL_LM')
    ids=torch.tensor([[1,2,3,4,5,6]])
    row={'action_mask':[False,False,True,False,True,True]}
    positions=torch.tensor(action_positions(row))
    labels=ids.clone();labels[0,torch.tensor([not v for v in row['action_mask']])]=-100
    expected=actor(input_ids=ids,labels=labels).loss
    actual=-action_log_probs(actor,ids,positions).mean()
    torch.testing.assert_close(actual,expected)
    before={n:v.detach().clone() for n,v in actor.named_parameters() if not v.requires_grad}
    opt=torch.optim.AdamW([v for v in actor.parameters() if v.requires_grad],lr=1e-3)
    actual.backward();opt.step()
    for n,v in actor.named_parameters():
        if n in before:torch.testing.assert_close(before[n],v)
