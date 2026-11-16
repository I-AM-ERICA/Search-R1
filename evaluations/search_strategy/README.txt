Search strategy evaluation — inference only, no further training

Purpose
Compare frozen-base Instruct, protocol-warmup checkpoint-00000 (128 SFT updates),
and PPO checkpoint-00150. Base vs 150 alone confounds warmup and PPO.
No sources, expected answers, or reviewer rubrics are placed in the model prompt.
The model must choose the first query, whether to search again, and its answer.

Two environments, two distinct questions
1. wiki / raw: same saved English prompt, 192 generated tokens per action, 768
   observation tokens, SQLite Wiki18 BM25, top-3, max two searches. Same seeded
   200-example validation subset as the historical pre/post-warmup evaluation.
   Start with --max-cases 20; use 200 for a full comparison if time permits.
   Exact match can be scored automatically. This is general English QA, NOT
   Korean policy effectiveness. It is a validation set, not an untouched test.
2. web / raw: Korean policy questions, identical production policy instructions
   for all variants, 512 generated tokens per action, 1200 observation tokens,
   Naver SerpAPI, top-3, max two model searches. Raw query enters the retriever
   unchanged; no clean_search_query, broadening, empty-query fallback, or rewrite.
   Existing retriever still prioritizes official domains and fetches HTML bodies.
3. web / assisted: repeat with --pipeline assisted. ObservedSearch performs the
   SAME cleaning, official filtering, fallback, and body focusing for each model.
   This isolates wrapper effects. It is NOT the complete frontend/gRPC pipeline:
   no current-date header, chat history, previous URL reading, or final rewrite.
4. Optional separate robustness run: web/raw --first-result-empty. First model
   search is deliberately returned empty without spending an API call. Measures
   recovery opportunity; never mix these rates with natural retrieval results.

Fairness / preservation
- One base backbone, greedy decoding, fixed seed. Adapter disabled for base,
  warmup adapter for warmup, current adapter for trained. No optimizer, save_pretrained,
  checkpoint mutation, or training-state restoration.
- Identical prompt tokens within each question pair; variant order rotates.
- Exact query/top-k results AND failures are frozen and shared. Page text for
  a URL is fetched once. Timestamp, full retrieved excerpts, actual visible
  observation, provider diagnostics and source/code/adapter hashes are saved.
- The whole web index cannot be frozen: distinct queries hit live search. Rerun
  with --replay PREVIOUS_RUN to forbid network access and replay recorded queries.
  Missing queries fail explicitly; they never become fake empty results.
- Equal search count/token/top-k limits. Assisted API retries are logged under
  the originating model search action, not counted as model-initiated searches.
- Infrastructure errors are reported separately and counted as end-to-end
  failures. Do not infer poor query quality from a provider timeout alone.
- New output directory required. Existing logs/checkpoints are not overwritten.
- API calls may use SerpAPI quota. Upper bound default 128 unique search requests
  per run; twelve cases x three models x two calls <=72. HTML page fetches are
  additional requests, not extra SerpAPI searches. Cached queries spend no new call.
- An active server on GPU localhost:50051 causes refusal. Stop it yourself before
  evaluation. No process is automatically killed. Restart serving after evaluation.
- These 12 Korean cases are exploratory, partly reused debugging questions.
  Build a larger held-out set spanning agencies, policies, regions, dates, and
  multi-source questions before making claims of statistical generalization.

Run from GPU /root/Search-R1 after unpacking archive into /tmp/search-r1-search-audit:
  bash /tmp/search-r1-search-audit/run_web_compare.sh

For training-environment evaluation, from GPU /root/Search-R1:
  /root/anaconda3/envs/naver/bin/python -u /tmp/search-r1-search-audit/compare_search_strategy.py \
    --repo /root/Search-R1 \
    --checkpoint outputs/search_r1_until_00150_20261114_125545_672828/checkpoint-00150 \
    --warmup-checkpoint outputs/search_r1_20261113_134427_797919/checkpoint-00000 \
    --environment wiki --pipeline raw --max-cases 20 \
    --output-dir outputs/search-compare-wiki-pilot

For assisted web comparison change --pipeline raw to assisted, using a NEW output.
To replay a previous run add --replay outputs/PREVIOUS_RUN and choose a NEW output.

Files / review
- results.json: full trajectories with variant names and blind review ID mapping.
- review-blind.json: same outputs shuffled, without variant names. Reviewer fills
  only the following nullable fields and notes, consulting the saved visible text:
    query_quality_0_to_2: 0=unusable/off-topic, 1=topic relevant but important entity,
      region/date/target missing or noisy, 2=appropriate focused query retaining
      necessary constraints. Do not reward query length or a particular wording.
    relevant_visible_evidence: true only if actually visible returned text helps
      answer THIS question from an appropriate source; domain or hit count alone
      is insufficient. Use null when infrastructure makes judgement impossible.
    sufficient_visible_evidence: evidence so far supports all essential requested
      facts. Review chronologically; information seen only later cannot count.
    new_useful_evidence: on second model search, newly obtained useful facts;
      a different URL or repeated fact alone is insufficient.
    accuracy_0_to_2: 0=major error/no usable answer, 1=partly correct with an error
      or misleading claim, 2=all stated factual claims supported and appropriate
      uncertainty. It is not a measure of covering every requested item.
    coverage_0_to_2: 0=generic/nonanswer, 1=some requested items, 2=all key requested
      items (including justified limitations when the facts cannot be determined).
    unsupported_claims: true/false/null. A source URL is not proof of each claim.
  Two reviewers should independently grade and adjudicate disagreements if this
  is used for a project performance claim. Save a copy; preserve the blank file.
- summary.json: automatic proxies ONLY until reviewed. Nonempty retrieval is not
  search success; valid </answer> is not answer completeness. Missing grades=null.
- manifest.json: model/code/config/seed/environment, actual unique provider calls.
- snapshots/: immutable per-query results for replay. pages.json: fetched bodies.

Aggregate filled grades (choose a new result filename):
  /root/anaconda3/envs/naver/bin/python /tmp/search-r1-search-audit/grade_search_strategy.py \
    --run outputs/RUN --reviews outputs/RUN/review-filled.json \
    --output outputs/RUN/graded-summary.json

Metrics / denominators
Search success: queries with relevant visible evidence / reviewed questions.
Query quality: mean 0..2 / reviewed executed model queries; normalized to 0..1.
Additional success: second search adds useful facts / reviewed cases that made
  a second search after insufficient first evidence. If none, n=0, rate=null.
Recovery: same numerator / ALL reviewed first-insufficient cases, including
  cases where the model failed to try again. Report together to avoid selection bias.
Final accuracy: accuracy=2 / reviewed questions; wiki also normalized exact match.
Content completeness: coverage=2 / reviewed questions, separate from tag closure.
Paired comparison reports wins/losses/ties, paired bootstrap interval and exact
McNemar test for binary accuracy. Small exploratory samples, repeated/multiple
comparisons, and web drift limit significance claims; p>0.05 is not equivalence.

Training audit
  python scripts/audit_ppo_run.py --runs RUN1 RUN2 RUN3 \
    --checkpoint-config CHECKPOINT/config.json \
    --checkpoint-metadata CHECKPOINT/metadata.json --output NEW_FILE.json
Missing/duplicate-conflicting steps are explicit; no interpolation or mixing runs.
