#!/usr/bin/env python3
"""Explicit, fresh-base actor LoRA SFT. Default is CPU-only preflight; no PPO/resume."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from policy_r1.sft_training import inspect_bundle,action_positions,sha
from policy_r1.sft_pilot import execution_plan,probe_inputs,probe_summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--tokenizer',required=True)
    p.add_argument('--output-dir')
    p.add_argument('--execute',action='store_true',help='Explicitly start training only if all data gates pass')
    p.add_argument('--pilot-small-data',action='store_true',help='Explicit bounded pilot; standard data readiness remains unchanged')
    p.add_argument('--epochs',type=int,help='Pilot only: 1 to 3 epochs')
    p.add_argument('--learning-rate',type=float,help='Pilot only: explicit LR arm, capped at 1e-4')
    p.add_argument('--comparison-checkpoint',help='Optional existing PPO actor for the same offline validation probes')
    args=p.parse_args()
    status,splits,settings,report=inspect_bundle(args.data,args.tokenizer,ROOT)
    print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)
    if not args.execute and not args.pilot_small_data:return
    try:
        plan=execution_plan(status,splits,settings,pilot=args.pilot_small_data,epochs=args.epochs,learning_rate=args.learning_rate)
    except ValueError as error:p.error(str(error))
    comparison=None
    if args.comparison_checkpoint:
        cp=Path(args.comparison_checkpoint).resolve()
        meta=json.loads((cp/'metadata.json').read_text())
        if meta['resolved_model_revision']!=settings['model_environment']['model_revision']:
            p.error('Comparison checkpoint base revision differs')
        comparison=dict(path=str(cp),adapter_sha256=sha(cp/'actor/adapter_model.safetensors'))
    print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
    if not args.execute:return
    if not args.output_dir:p.error('--output-dir is required with --execute')
    output=Path(args.output_dir).resolve()
    if output.exists() or any(part.startswith('checkpoint-') for part in output.parts):
        p.error('Use a new output directory outside existing checkpoints')
    with socket.socket() as sock:
        sock.settimeout(.3)
        if sock.connect_ex(('127.0.0.1',50051))==0:
            p.error('Stop GPU serving yourself before training; no process was stopped')
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from policy_r1.models import prepare_backbone,action_log_probs
    from policy_r1.config import Config
    if not torch.cuda.is_available():p.error('CUDA GPU required')
    cfg=Config(**settings['model_environment']).validate()
    spec=plan['training_spec']
    if settings['initialization']!='fresh_base_lora; do not resume checkpoint-00150':
        p.error('Only fresh-base LoRA is supported')
    if not (spec['learning_rate']>0 and type(spec['epochs']) is int and spec['epochs']>0
            and type(spec['gradient_accumulation']) is int and spec['gradient_accumulation']>0
            and spec['max_grad_norm']>0):p.error('Invalid SFT hyperparameters')
    torch.manual_seed(cfg.seed)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    kwargs=dict(revision=cfg.model_revision,torch_dtype=dtype,device_map={'':torch.cuda.current_device()},attn_implementation='sdpa')
    if cfg.load_in_4bit:
        kwargs['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    base=AutoModelForCausalLM.from_pretrained(cfg.model_name,**kwargs)
    if cfg.max_seq_length>base.config.max_position_embeddings:p.error('Model context limit exceeded')
    actor=prepare_backbone(base,cfg,'CAUSAL_LM')
    optimizer=torch.optim.AdamW([v for v in actor.parameters() if v.requires_grad],lr=spec['learning_rate'])
    output.mkdir(parents=True,exist_ok=False)
    metadata=dict(checkpoint_kind='policy_search_sft',resolved_model_revision=cfg.model_revision,
        initialization=settings['initialization'],compiled_report_sha256=sha(Path(args.data)/'report.json'),
        prompt_sha256=report['prompt_sha256'],
        source_config=settings,execution_plan=plan,comparison=comparison,
        training_code_sha256=sha(__file__),pilot_code_sha256=sha(ROOT/'policy_r1/sft_pilot.py'),
        created_at=datetime.now(timezone.utc).isoformat())
    (output/'run.json').write_text(json.dumps(metadata,indent=2))
    train=splits['train'];accum=spec['gradient_accumulation'];update=0
    def save_checkpoint(epoch):
        final=output/f'checkpoint-{update:05d}'
        if final.exists():return
        temp=output/f'.checkpoint-{update:05d}.tmp';temp.mkdir()
        actor.save_pretrained(temp/'actor',safe_serialization=True)
        tokenizer.save_pretrained(temp/'tokenizer')
        (temp/'config.json').write_text(json.dumps(cfg.to_dict(),indent=2))
        (temp/'metadata.json').write_text(json.dumps(dict(metadata,completed_updates=update,completed_epoch=epoch),indent=2))
        torch.save(dict(optimizer=optimizer.state_dict(),updates=update),temp/'optimizer.pt')
        temp.rename(final)
    def nll(row):
        device=next(actor.parameters()).device
        ids=torch.tensor([row['input_ids']],dtype=torch.long,device=device)
        positions=torch.tensor(action_positions(row),dtype=torch.long,device=device)
        return -action_log_probs(actor,ids,positions).sum()
    def validate(epoch):
        actor.eval()
        with torch.inference_mode():
            total=sum(float(nll(row)) for row in splits['val'])
        count=sum(sum(row['action_mask']) for row in splits['val'])
        metric=dict(epoch=epoch,update=update,action_nll=total/count,
                    note='Teacher-forced NLL, not search/answer accuracy')
        with (output/'validation.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
        print('VALIDATION '+json.dumps(metric),flush=True)
    def probe(label,variant):
        from scripts.compare_policy_summary import generate_one
        from policy_r1.protocol import parse_action
        from policy_r1.policy_sft import fingerprint
        prepared=probe_inputs(splits['val']);results=[]
        # Disable training checkpointing for generation; restore before updates.
        actor.gradient_checkpointing_disable()
        try:
            with (output/(label+'-probe.jsonl')).open('x') as stream:
                for item in prepared:
                    result=generate_one(actor,tokenizer,item['input_ids'],variant,'tagged',cfg.max_new_tokens,cfg.seed)
                    action,content=parse_action(result['output'])
                    result.update(variant=label,case_id=item['case_id'],stage=item['stage'],
                        expected_action=item['expected_action'],action=action,action_content=content,
                        format_valid=action in ('search','answer'),completed=action in ('search','answer'),
                        input_ids_sha256=fingerprint(item['input_ids']),input_tokens=len(item['input_ids']),
                        semantic_grade=None)
                    results.append(result);stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
                    print(f"PROBE {label} {len(results)}/{len(prepared)} {item['stage']}: {action}",flush=True)
            (output/(label+'-probe-summary.json')).write_text(json.dumps(probe_summary(results),indent=2))
        finally:
            if cfg.gradient_checkpointing:
                actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    try:
        if args.pilot_small_data:
            (output/'probe-inputs.json').write_text(json.dumps(probe_inputs(splits['val'])))
            validate(0)
            probe('base','base')
        for epoch in range(spec['epochs']):
            order=list(range(len(train)));random.Random(cfg.seed+epoch).shuffle(order)
            actor.train()
            for start in range(0,len(order),accum):
                batch=[train[i] for i in order[start:start+accum]]
                denom=sum(sum(r['action_mask']) for r in batch);optimizer.zero_grad(set_to_none=True)
                began=time.monotonic();loss_value=0.0
                for row in batch:
                    loss=nll(row)/denom
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite SFT loss')
                    loss.backward();loss_value+=float(loss.detach())
                norm=torch.nn.utils.clip_grad_norm_([v for v in actor.parameters() if v.requires_grad],spec['max_grad_norm'],error_if_nonfinite=True)
                optimizer.step();update+=1
                metric=dict(update=update,epoch=epoch+1,action_nll=loss_value,action_tokens=denom,grad_norm=float(norm),seconds=time.monotonic()-began)
                with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
                print(json.dumps(metric),flush=True)
                if update%10==0 and not args.pilot_small_data:save_checkpoint(epoch+1 if start+accum>=len(order) else epoch)
            save_checkpoint(epoch+1)
            validate(epoch+1)
        (output/'TRAINING_COMPLETE.json').write_text(json.dumps(dict(updates=update,epochs=spec['epochs'],checkpoint=str(output/f'checkpoint-{update:05d}'))))
        if args.pilot_small_data:
            probe('sft','trained')
            if comparison:
                cp=Path(comparison['path'])
                if sha(cp/'actor/adapter_model.safetensors')!=comparison['adapter_sha256']:
                    raise ValueError('Comparison adapter changed during training')
                actor.load_adapter(cp/'actor',adapter_name='ppo150',is_trainable=False)
                actor.set_adapter('ppo150')
                probe('ppo150','trained')
                actor.set_adapter('default')
                if sha(cp/'actor/adapter_model.safetensors')!=comparison['adapter_sha256']:
                    raise ValueError('Comparison adapter changed during probes')
    except BaseException:
        (output/'INCOMPLETE.txt').write_text('Interrupted/failed. Only completed atomic checkpoints may be used. No automatic resume.\n')
        raise
    print('SFT COMPLETE:',output/f'checkpoint-{update:05d}')


if __name__=='__main__':main()
