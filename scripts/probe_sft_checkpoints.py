"""Read-only conditional probes for already-trained SFT checkpoints. No training/search."""
import argparse
import json
from pathlib import Path
import socket
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy_r1.sft_training import inspect_bundle, sha
from policy_r1.sft_pilot import probe_inputs, probe_summary
from policy_r1.policy_sft import fingerprint
from policy_r1.protocol import parse_action


def selected_inputs(inputs, entry):
    if 'probe_keys' not in entry:
        return inputs
    lookup = {(r['case_id'],r['stage']):r for r in inputs}
    keys = [tuple(k) for k in entry['probe_keys']]
    if not keys or len(set(keys)) != len(keys) or any(k not in lookup for k in keys):
        raise ValueError('Invalid probe subset')
    return [lookup[k] for k in keys]


def verify_source(source, reference):
    source = Path(source).resolve()
    for name, expected in reference['files'].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source):
            raise ValueError('Source reference escapes experiment')
        if sha(path) != expected:
            raise ValueError('Source file changed: ' + name)
    if (source / 'exit-code.txt').read_text().strip() != '0':
        raise ValueError('Source experiment did not complete')
    package = source / 'package'
    _, splits, settings, report = inspect_bundle(package / 'data', package / 'tokenizer', package / 'runtime')
    inputs = json.loads((source / 'training/probe-inputs.json').read_text())
    if inputs != probe_inputs(splits['val']):
        raise ValueError('Probe inputs differ from validation-only compiled prefixes')
    cfg = settings['model_environment']
    for label, item in reference['checkpoints'].items():
        selected_inputs(inputs, item)
        checkpoint = source / item['path']
        meta = json.loads((checkpoint / 'metadata.json').read_text())
        checkpoint_config = json.loads((checkpoint / 'config.json').read_text())
        if (meta['completed_epoch'] != item['epoch'] or meta['completed_updates'] != item['updates']
                or meta['resolved_model_revision'] != cfg['model_revision']
                or meta['prompt_sha256'] != report['prompt_sha256']
                or meta['compiled_report_sha256'] != sha(package / 'data/report.json')
                or any(checkpoint_config[k] != v for k, v in cfg.items())):
            raise ValueError('Checkpoint metadata/config mismatch: ' + label)
        actual_tokenizer = {p.name: sha(p) for p in (checkpoint / 'tokenizer').iterdir() if p.is_file()}
        if actual_tokenizer != reference['checkpoint_tokenizer_sha256']:
            raise ValueError('Checkpoint tokenizers differ')
    return inputs, settings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    source = Path(args.source_run).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() or output == source or source in output.parents:
        p.error('Use a new output directory outside the source experiment')
    if any(part.startswith('checkpoint-') for part in output.parts):
        p.error('Do not write inside a checkpoint')
    reference = json.loads(Path(args.reference).read_text())
    inputs, settings = verify_source(source, reference)
    from transformers import AutoTokenizer
    # Check saving did not change the tokenizer behavior for frozen input IDs.
    package_tok = AutoTokenizer.from_pretrained(source / 'package/tokenizer', local_files_only=True)
    for entry in reference['checkpoints'].values():
        saved_tok = AutoTokenizer.from_pretrained(source / entry['path'] / 'tokenizer', local_files_only=True)
        if (saved_tok.get_vocab() != package_tok.get_vocab()
                or saved_tok.chat_template != package_tok.chat_template
                or saved_tok.eos_token_id != package_tok.eos_token_id
                or saved_tok.pad_token_id != package_tok.pad_token_id):
            raise ValueError('Saved tokenizer behavior differs from compiled source')
    total = sum(len(selected_inputs(inputs, entry)) for entry in reference['checkpoints'].values())
    print(f'CHECKPOINT PROBE VERIFIED: {total} generations; training 0; search calls 0', flush=True)
    if args.preflight_only:
        return
    with socket.socket() as sock:
        sock.settimeout(.3)
        if sock.connect_ex(('127.0.0.1', 50051)) == 0:
            p.error('Serving port occupied; no process stopped')
    from policy_r1.inference import load_actor
    from scripts.compare_policy_summary import generate_one
    first = next(iter(reference['checkpoints'].values()))
    tokenizer, actor, cfg = load_actor(source / first['path'])
    actor.gradient_checkpointing_disable()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'manifest.json').write_text(json.dumps(dict(
        source_run=str(source), reference=reference, script_sha256=sha(__file__),
        generator_sha256=sha(ROOT/'scripts/compare_policy_summary.py'),
        created_at=datetime.now(timezone.utc).isoformat(), training_started=False,
        search_requests=0, generations=total,
        note='Same teacher-forced validation prefixes and greedy settings as prior run; no autonomous retrieval or automatic factuality score'), indent=2))
    (output / 'probe-inputs.json').write_text(json.dumps(inputs))
    try:
        for index, (label, entry) in enumerate(reference['checkpoints'].items()):
            if index:
                actor.load_adapter(source / entry['path'] / 'actor', adapter_name=label, is_trainable=False)
                actor.set_adapter(label)
            actor.requires_grad_(False)
            rows = []
            selected = selected_inputs(inputs, entry)
            with (output / (label+'-probe.jsonl')).open('x') as stream:
                for item in selected:
                    result = generate_one(actor, tokenizer, item['input_ids'], 'trained', 'tagged',
                                          cfg.max_new_tokens, cfg.seed, include_generation_details=True)
                    action, content = parse_action(result['output'])
                    result.update(variant=label, case_id=item['case_id'], stage=item['stage'],
                                  expected_action=item['expected_action'], action=action, action_content=content,
                                  format_valid=action in ('search','answer'), completed=action in ('search','answer'),
                                  input_ids_sha256=fingerprint(item['input_ids']), input_tokens=len(item['input_ids']),
                                  semantic_grade=None)
                    if 'previous_outputs' in entry:
                        previous = entry['previous_outputs'][item['case_id']+'/'+item['stage']]
                        result['reproduces_previous_output'] = result['output'] == previous
                    rows.append(result)
                    stream.write(json.dumps(result, ensure_ascii=False)+'\n'); stream.flush()
                    print(f'PROBE {label} {len(rows)}/{len(selected)} {item["stage"]}: {action}', flush=True)
            (output / (label+'-probe-summary.json')).write_text(json.dumps(probe_summary(rows), indent=2))
        verify_source(source, reference)
        (output / 'COMPLETE.json').write_text(json.dumps(dict(training_started=False, search_requests=0, generations=total)))
    except BaseException:
        (output / 'INCOMPLETE.txt').write_text('Probe failed/interrupted. Existing checkpoints were not trained or replaced.\n')
        raise
    print('CHECKPOINT COMPARISON COMPLETE:', output, flush=True)


if __name__ == '__main__':
    main()
