한국어 정책 검색 개선 v1 — 수집/검토 단계, 추가 학습 없음

현재 상태
18개 새 질문(학습 후보12/검증 후보3/테스트 후보3), 최대36회 실제 Naver SerpAPI 검색을 수집한다.
이는 데이터 흐름 검증용 파일럿이며 학습에 충분한 양이 아니다. 정답과 본문은 만들어 넣지 않는다.
기존 150스텝 체크포인트, optimizer, Wiki 데이터, 평가 출력은 변경하지 않는다.
run_capture.sh는 CPU 검색 수집만 한다. 서빙/GPU 학습 프로세스를 종료하거나 시작하지 않는다.

환경 변경
1. observation_strategy=concat: 기존 Wiki와 과거 결과 재현용 기본값 유지.
2. balanced_v1: 각 문서의 완전한 ID, 본문 토큰을 균등 배분하고 짧은 문서의 잔여 예산을 재분배.
   실제 남은 context와 다음 action의 예약 공간을 기준으로 제한한다.
   제목/본문 출처 종류는 보이며 긴 URL은 trace/source 목록에 보존한다.
   문서 속 action 태그와 가짜 [ID] 경계를 escape한다. 전체 observation은 loss에서 제외한다.
3. 공식 호스트에 서울신용보증재단, 경기신용보증재단, 소상공인24를 정확한 호스트 단위로 추가.
   검증 링크는 official_pages.OFFICIAL_REGISTRY에 있다. 도메인 소속만으로 사실성/최신성을 보장하지 않는다.
4. official_only=True: 페이지 fetch 예산을 쓰기 전에 민간 문서를 제외한다.
   새로운 환경의 raw 비교에서는 검색어 정리/자동 재검색을 하지 않는다. 모델의 추가 검색과 혼동하지 않는다.
5. PDF/HWP/첨부파일/로그인/JavaScript 전용 페이지는 여전히 읽지 못할 수 있다.
   HTML 실패를 snippet이나 추측으로 대체하지 않는다. 빈 검색과 API 오류를 구별한다.

기존 서버에 대한 영향
로컬 파일을 수정했지만 GPU 서비스에는 아직 설치하지 않는다.
배포 패키지 runtime/은 독립 복사본으로 /tmp에서 실행한다. 기존 서버 소스/포트를 변경하지 않는다.
기존 serving과 PPO의 기본 observation 설정은 자동으로 balanced_v1로 바뀌지 않는다.
검증 뒤 서비스에도 같은 환경을 적용하고 SFT 모델을 비교해야 한다. 지금 성능 개선을 주장하지 않는다.

수집 데이터
collection_plan.json의 두 query는 교사 후보가 미리 작성한 것이다. 학습된 검색 전략이 아니다.
Collector는 두 후보를 모두 기록하므로 두 번째 검색이 필요했다는 증거로 사용하면 안 된다.
검토자가 실제 첫 검색의 visible observation을 읽고 한 단계 또는 두 단계만 선택한다.
잘못된 검색어/관련 없는 결과를 성공 예제로 승인하지 않는다. 수정 질의는 재수집한 snapshot이 필요하다.
두 번째 검색을 선택할 때 reason_for_next_search에 부족했던 사실/기관/지역/시점과 새 근거를 기록한다.
사유는 모델 target에 넣지 않고 검토 정보로만 보관한다. action 형식은 기존 <search>/<answer>를 유지한다.
first-empty 사례를 인위적으로 만들 경우 자연 발생 검색과 섞지 말고 별도의 intervention 데이터로 관리한다.
최대 요청 수를 초과하면 실행 전에 거부한다. 웹 본문 fetch는 SerpAPI 호출과 별개다.
응답/본문/시간/diagnostics/코드/파일 hash를 보관한다. API 키나 인증 환경변수는 저장하지 않는다.

검토 절차
1. capture/drafts.json을 새 reviews.json으로 복사한다. 원본 snapshot은 수정하지 않는다.
2. 공식 출처의 실제 공고 제목/기간/기관을 확인한다. 단순 홈 화면/목차는 근거가 아니다.
3. 불필요한 두 번째 step은 제거한다. provider error가 난 step은 학습 대상에서 제외한다.
4. answer를 작성하고 모든 중요한 사실을 claims에 연결한다:
   {"text":"답변에 실제 있는 주장", "evidence":[{"id":"web_...", "quote":"본문의 정확한 근거 발췌"}]}
   인원 '미만/이하', 사업장/거주지, 연도/접수기간, 융자/보증, 지원한도/확정액을 검토한다.
   각 근거는 최종 토큰화 후에도 그 문서의 실제 visible observation에 있어야 한다.
5. outcome=supported 또는 insufficient. 부족하면 limitations에 미확인 요구사항을 명시한다.
   답변 가능함에도 무조건 '모름'을 쓰는 것은 좋은 예제가 아니다.
6. review에 reviewer, notes, evidence_checked=true, coverage_checked=true,
   status=approved, content_sha256=policy_sft.content_digest(row)를 기록한다.
   AI 검토라면 reviewer에 AI라고 명시한다. hash는 무결성 검사이지 사람 검증의 증명은 아니다.
   텍스트 포함 검사만으로 의미적 사실성을 자동 판정하지 않는다.
7. prepare_policy_sft.py를 실행해 dataset leakage/출처/마스킹/길이/visible evidence를 검사한다.
   정답을 본문에 넣거나 학습 label을 조용히 자르지 않는다. 실패한 예제는 수정/제외 후 새 산출물로 만든다.

데이터 분리
정책 family group을 기준으로 split을 먼저 고정한다. 같은 URL, 본문 hash, 질문이 split을 넘으면 거부한다.
기존 18개 평가 질문과 과거 관측 37개 URL도 exclusions.json으로 보존/제외한다.
현재 pilot의 분리 축은 policy family다. 모든 기관/지역/날짜가 독립이라는 뜻은 아니다.
URL 파라미터나 표현이 다른 동일 공고, 같은 정책의 다른 연도는 의미 검토로 추가 차단해야 한다.
test는 최종평가에만 사용하고 검색 질의나 reward 튜닝에 재사용하지 않는다.
부족한 데이터 수를 같은 문장 복제로 채우지 않는다.

SFT 준비 (GPU에서 local tokenizer만 읽음; GPU 모델을 로드하거나 학습하지 않음)
  python scripts/prepare_policy_sft.py --capture /absolute/CAPTURE \
    --reviews /absolute/reviews.json --output-dir /absolute/NEW_PREPARED \
    --tokenizer /root/Search-R1/outputs/search_r1_until_00150_20261114_125545_672828/checkpoint-00150/tokenizer
출력 train/val/test.jsonl의 labels는 prompt/observation=-100, search/answer=token ID다.
보고서 ready_for_training은 tokenizer 검증과 최소 데이터 구성을 모두 통과해야 true다.
120 train/30 val/성공적인 2회검색30개/1회검색/근거부족 사례는 초기 engineering gate이며 성능 보장은 아니다.
실제 학습 실행기는 이 패키지에 없다. 데이터를 검토한 뒤 별도 새 실험으로 준비한다.
권장 시작점은 원본 기반 새 LoRA, LR2e-5/1 epoch/누적8의 소규모 SFT pilot이다.
기존 PPO150을 덮어쓰거나 해당 optimizer로 재개하지 않는다.

동일 검색 환경 비교 (모델을 로드하므로 기존 서빙과 메모리 충돌 여부 확인)
  python scripts/compare_search_strategy.py --repo . \
    --checkpoint /absolute/checkpoint-00150 --warmup-checkpoint /absolute/checkpoint-00000 \
    --environment web --pipeline raw --official-only --observation-strategy balanced_v1 \
    --output-dir /absolute/NEW_EVALUATION
기존12문항은 개발용 진단으로만 쓴다. 새 test 파일럿으로 성능을 미리 최적화하지 않는다.
새 SFT 모델은 생성 뒤 별도 adapter comparison에 연결하고 동일한 prompt/token/search budget으로 비교한다.
위 기본 비교는 여전히 기존 원본/예열/PPO150이며 새 SFT 성능 결과가 아니다.

PPO 보상 설계 — 아직 trainer에 연결하지 않음
policy_reward.reviewed_policy_reward는 검토한 결과에 대한 offline calibration이다.
accuracy/coverage/grounding 각각0..2, 가중치 .5/.25/.25. 중대한 오류나 근거 없는 주장 -> 0.
근거 없음은 정상 완료 답변에 점수를 주지 않는다. 정당한 판단 보류는 최대.25로 별도 처리.
길이/태그/검색 횟수/URL 수/기관 도메인 자체에는 점수를 주지 않는다.
candidate hash와 reviewer identity를 요구하며 미채점 null을0 또는정답으로 바꾸지 않는다.
자동 의미 판정기는 아직 없다. 이 수식만으로 online PPO 학습을 시작하면 안 된다.
정답 예제/틀린 경계값/근거 없는 금액/무조건 거절/중복 검색으로 calibration을 먼저 진행한다.
공식 문서 속 근거를 실제로 찾은 완전한 답변 > 정당한 보류 > 근거 없는 단정 순서가 유지되는지 점검한다.
그 후 SFT-only vs SFT+PPO를 같은 held-out 질문으로 비교한다.

산출물 해석
테스트 통과는 수집/마스킹/누수방지 구현 검증이지 정책 답변 성능 개선 증명이 아니다.
추가 학습은 수행하지 않았다. 기존150 로그/체크포인트도 그대로 보존한다.
