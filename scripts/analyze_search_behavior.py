"""Read-only behavioral audit of completed comparisons; never treat hit counts as correctness."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def query_key(text):
    return ' '.join(text.casefold().split())


def behavior(rows):
    output = {}
    for variant in dict.fromkeys(r['variant'] for r in rows):
        data = [r for r in rows if r['variant'] == variant]
        valid = [r for r in data if not r.get('environment_error')]
        searched = [r for r in valid if r['search_actions']]
        empty = [r for r in searched if not r['search_actions'][0]['returned_hits']]
        second = [r for r in searched if len(r['search_actions']) >= 2]
        changed = [r for r in second if query_key(r['search_actions'][0]['query']) != query_key(r['search_actions'][1]['query'])]
        gold = [r for r in valid if r.get('exact_match') is not None]
        calls = [a for r in valid for a in r['search_actions']]
        output[variant] = dict(questions=len(data), environment_errors=len(data)-len(valid),
            format_complete=sum(r['format_complete'] for r in valid),
            searched_questions=len(searched), second_search_questions=len(second),
            changed_second_queries=len(changed), repeated_second_queries=len(second)-len(changed),
            first_empty_questions=len(empty),
            second_search_after_first_empty=sum(len(r['search_actions']) >= 2 for r in empty),
            second_nonempty_after_first_empty_PROXY=sum(len(r['search_actions']) >= 2 and bool(r['search_actions'][1]['returned_hits']) for r in empty),
            answer_after_first_empty_without_retry=sum(len(r['search_actions']) == 1 and r['format_complete'] for r in empty),
            exact_question_copy_queries=sum(query_key(a['query']) == query_key(r['question']) for r in valid for a in r['search_actions']),
            total_model_searches=len(calls), logical_provider_requests=sum(r.get('provider_requests',0) for r in data),
            exact_match_correct=sum(r['exact_match'] == 1 for r in gold), exact_match_evaluated=len(gold),
            finish_reasons=dict(Counter(r['finish_reason'] for r in data)))
    return output


def audit(directory):
    directory = Path(directory)
    rows = json.loads((directory/'results.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    traces = []
    for row in rows:
        acts = row['search_actions']
        if len(acts)>=2 or (acts and not acts[0]['returned_hits']):
            traces.append(dict(case_id=row['case_id'],variant=row['variant'],
                finish_reason=row['finish_reason'], exact_match=row.get('exact_match'),
                searches=[dict(query=a['query'],returned_hits=len(a['returned_hits']),
                    visible_ids=a.get('visible_ids',[]),provider_queries=[q['query'] for q in a['requests']]) for a in acts]))
    prompt_pairs = {}
    for row in rows:prompt_pairs.setdefault(row['case_id'],set()).add(row['input_tokens'].__str__()+':'+row['prompt_ids_sha256'])
    return dict(source=str(directory), source_sha256={n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in ('results.json','manifest.json')},
        environment=manifest['environment'], pipeline=manifest['pipeline'],
        web_evidence=manifest.get('web_evidence','pages') if manifest['environment']=='web' else None,
        intervention=manifest.get('first_empty',False), same_prompt_for_all_variants=all(len(v)==1 for v in prompt_pairs.values()),
        by_variant=behavior(rows), selected_behavior_traces=traces,
        limitations=['Empty hits and insufficient evidence are different; the latter requires semantic review.',
                    'Changed query is an observable action, not proof of better query quality or multi-hop reasoning.',
                    'Logical provider requests include shared-cache hits. Environment failures are excluded from behavioral denominators.',
                    'Base means the same HyperCLOVA model with adapter disabled, not an author-released Search-R1 checkpoint.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    result=[audit(d) for d in args.runs]
    with Path(args.output).open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    for run in result:
        print(run['source'])
        print(json.dumps(run['by_variant'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
