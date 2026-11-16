<!-- 역할: 프로젝트 구조, 원본 Search-R1과의 관계, GPU 서버 연결 및 노트북 학습 절차를 설명합니다. -->
<!-- Current product direction and serving entry point are described immediately below. -->

## 현재 목표: 한국어 소상공인 정책 비서 MVP

Search-R1은 방법론 참고이며 논문 수치 재현이 최종 목표가 아닙니다. 아래의 과거 QA/PPO 학습 안내는 보존한 실험 기록입니다. 현재는 추가 PPO/SFT 학습과 데이터 확장 대신 SerpAPI 검색 → 공식 HTML 확인 → 근거 답변 → Spring gRPC 연결을 우선합니다.

서버 루트 `.env`에 `SERPAPI_API_KEY`, `AI_INTERNAL_KEY`를 설정합니다. 실행 스크립트는 셸 실행 없이 이 두 값만 읽습니다. Spring에는 같은 `AI_INTERNAL_KEY`를 환경변수로 설정합니다. 키를 소스나 요청 URL 로그에 넣지 않습니다.

```bash
cd /root/Search-R1
bash scripts/run_policy_mvp.sh
```

이미 실행 중이면 중복 실행하지 않습니다. 2026-11-16 실행 세션은 `policy-mvp-20261116`, 로그는 `outputs/policy-mvp-e2e-20261116/server.log`입니다. 서버는 `127.0.0.1:50051`에 바인딩합니다. 다른 머신의 Spring은 SSH 터널 또는 기존 사설 네트워크 설정을 사용합니다. 기존 프로세스를 자동 종료하지 않습니다.

실제 gRPC 호출:

```bash
/root/anaconda3/envs/naver/bin/python scripts/ask_policy.py \
  --question '2026년 서울시 중소기업육성자금 융자지원 계획의 대상과 신청 기간, 신청 방법을 공식 공고로 확인해줘.' \
  --region 서울 --category 음식점
```

기본 `--pipeline mvp`는 서버가 첫 검색과 최대 두 번의 검색 예산을 관리합니다. Naver 엔진의 SerpAPI를 호출하고 공식 도메인의 읽을 수 있는 HTML만 사용합니다. 부족하면 검색어를 수정합니다. 답변은 지원 대상·제외 조건·신청 기간·절차·지원 내용을 공고별로 구분한 본문 인용입니다. 모델 인용 검증에 실패하면 서버가 원문 섹션을 발췌하며, 이 대체 경로는 모델의 정답으로 집계하지 않습니다. 현재 서비스는 자유로운 생성 답변보다 확인 가능한 원문을 우선하는 MVP입니다. 첨부 PDF/HWP/OCR, 최종 자격 판정, 접수 시스템 상태 확인은 구현 범위에 포함하지 않습니다.

`serving/model-selection.json`은 SFT 1 epoch를 임시 후보로 지정합니다. 원본/PPO150/SFT 1·2·3 epoch의 4개 새 정책 질문 비교에서 자금 공고의 가장 명확한 조건 설명을 기준으로 골랐으며, 일반적인 최적 성능을 입증한 것은 아닙니다. 비교 원출력은 `outputs/policy-mvp-live-20261116/results.jsonl`, 최종 보호 경로 결과는 `outputs/policy-mvp-final-20261116/results.jsonl`에 있습니다. 후자의 두 질문은 SerpAPI 검색 한도 소진 때문에 검증하지 못했습니다. 2026-11-16 확인 당시 월 사용량 250회, 잔여 0회였으므로 새로운 질의의 실제 검색에는 한도 복구가 필요합니다. 검색 장애는 정책 부재로 처리하지 않고 gRPC `UNAVAILABLE`을 반환합니다.

키와 검색 한도가 준비된 뒤 새 출력 디렉터리에서 비교를 실행할 수 있습니다. `--preflight-only`는 GPU나 API를 사용하지 않습니다.

```bash
/root/anaconda3/envs/naver/bin/python scripts/evaluate_policy_mvp.py \
  --output-dir outputs/policy-mvp-next-comparison --search-engine naver
```

Spring Boot 호출 모듈은 `serving/spring-client/`입니다. 기존 `serving/proto/`에서 Java 메시지와 gRPC stub을 생성하며 `POST /api/policy/questions`를 제공합니다. Java 17과 Maven이 있는 환경에서:

```bash
mvn -f serving/spring-client/pom.xml spring-boot:run
```

```json
{"question":"2026년 서울시 중소기업육성자금의 대상과 신청 방법은?","store":{"region":"서울","category":"음식점"}}
```

응답은 `answer`, `sources[{title,url,snippet}]`입니다. 위 standalone Spring 모듈은 초기 예시입니다. 이후 전달받은 실제 저장소는 `integrations/am-erica-backend`, `integrations/am-erica-frontend`에서 수정했습니다. 실제 백엔드는 Java 21/Gradle, 프론트는 Node 22로 빌드·테스트했고 기존 gRPC/SSE 구현을 재사용합니다. `local,policy-mvp` 프로필로 정책만 Python에 연결하며, 실행과 SSH 터널 안내는 실제 백엔드 README를 참고하세요. 전체 프론트 서비스 함수 → Vite 프록시 → Spring 인증/매장 확인 → Python → SerpAPI 경로의 검증 결과는 `outputs/policy-app-integration-20261116/live-smoke.json`에 있습니다. AskStream은 검색/생성 완료 후 청크를 보내며 실시간 토큰 스트리밍은 아닙니다.

# Search-R1 실제 QA 데이터로 HyperCLOVA PPO 학습

`train.ipynb`는 **NQ + HotpotQA 질문 → HyperCLOVA Actor → Wikipedia 검색 → 답변 → 보상 + Critic → PPO 업데이트**를 실행합니다. 기존 가상 정책 실험은 `train_demo.ipynb`와 `configs/first_experiment.yaml`에 보존했습니다.

기본 모델은 [naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B)입니다. 모델 페이지에서 이용 조건에 동의한 Hugging Face 계정이 필요합니다.

## 구현 범위

- Actor: HyperCLOVA 1.5B + 학습 가능한 LoRA adapter.
- Critic: **별도로 로드한** 동일 HyperCLOVA backbone + 독립 LoRA + token별 scalar Value Head. Actor와 가중치를 공유하지 않습니다.
- Reference: Actor의 고정 base에서 LoRA를 일시 비활성화하여 초기 정책 log probability를 계산합니다. base·embedding·LM head는 학습하지 않습니다.
- PPO: rollout 시점 old log probability/value 고정, reference KL penalty, GAE, clipped policy/value loss, gradient clipping, rollout batch 단위 advantage 정규화.
- 본 학습 검색: Wikipedia를 디스크에 저장하는 SQLite FTS5 BM25. 전체 문서를 Python 리스트에 올리지 않습니다. 작은 문서용 메모리 BM25/CPU Dense도 유지합니다. 최대 2회 검색 후 답변 생성 기회 1회.
- 관측 마스킹: 질문과 `<information>` 관측은 문맥으로 유지하되 Actor/Critic loss 및 KL·GAE action 시간축에서 제외합니다. 검색 질의는 Actor가 생성한 action이므로 학습합니다.
- 보상: 영어 QA는 원본 Search-R1의 소문자·ASCII 문장부호·관사·공백 정규화 후 EM을 사용합니다. 한국어 데모의 compact 정규화는 유지합니다. 정답 1, 그 외 0이며 형식/검색 보상은 기본 0입니다.
- 4-bit NF4 + LoRA, GPU 1대, trajectory microbatch 1, 본 학습 rollout batch 8, PPO epoch 2입니다. 노트북/CLI `--epochs 1`은 전체 학습 질문을 최소 한 번 순회하는 update 수를 계산합니다. PPO epoch는 각 수집 배치를 재사용하는 횟수입니다.

정확한 GPU 종류/VRAM은 스크린샷만으로 확정하지 않습니다. 노트북에서 실제 GPU와 메모리를 확인합니다. 메모리 부족 시 `max_seq_length=1024`, `max_new_tokens=128`, `max_observation_tokens=128`부터 줄이세요. rollout batch를 줄이면 누적량은 줄지만 한 trajectory의 최대 GPU 메모리는 길이에 주로 좌우됩니다. SQLite 검색은 CPU/디스크를 사용하며 E5 dense 검색과 품질·속도가 다릅니다. 분산 학습이나 논문 수치 동일 재현은 이 구현의 범위가 아닙니다.

## loss·reward가 계속 0일 때: 이번 수정

실제 확인된 trace는 `<search>...</search>` 대신 `assistant -> tool/python`, Python `search(...)`, 모델이 꾸며 낸 검색 결과를 생성했습니다. 192 action 토큰을 모두 썼지만 parser는 이를 검색/답변으로 인정하지 않아 검색 횟수와 정답 보상이 0이었습니다.

이 코드에서는 초기 LoRA가 base와 같은 출력이고 Critic Value Head가 0입니다. 정답 보상도 0이면 reference KL → token reward → GAE advantage → Actor/Critic gradient가 전부 0이 되어 **optimizer를 호출해도 가중치가 그대로인 상태**가 재현됩니다. 일부 `valid_answer`만 올라가도 오답이고 `format_reward=0`이면 보상은 0입니다. `actor_loss` 평균만 0인 것은 advantage 정규화 때문일 수 있지만, 모든 loss·KL·보상이 장기간 0인 이 경우에는 실제 gradient 확인이 필요합니다.

수정 내용:

- prompt를 짧은 검색/답변 태그 지시로 정리하고 Python 도구 호출과 역할/검색 결과 흉내를 금지했습니다.
- HyperCLOVA [공식 생성 예제](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B#huggingface-usage-example)의 `<|endofturn|>`, `<|stop|>` 종료 조건도 반영했습니다. 기본 EOS만 사용하는 대신 턴 종료 시 멈춰 다른 역할·도구 출력을 계속 흉내 내는 경로를 막습니다. 턴 종료 전 유효한 action이 없으면 `model_turn_end`로 기록합니다.
- greedy 평가에는 sampling 전용 `top_k=0`을 전달하지 않아 경고를 없앴습니다. 학습 sampling은 temperature=1, top_k=0, top_p=1 그대로입니다.
- 생성 길이 초과를 `generation_limit`로 구분하고 event에 `generated_tokens`, `hit_token_limit`를 남깁니다.
- `actor_grad_norm`, `critic_grad_norm`, `advantage_std`, `advantage_abs_mean`, `finish_reasons`를 기록합니다. 두 gradient가 0인 상태가 `max_zero_signal_updates`(기본 3)회 연속이면 중단합니다. 종료 전 모든 batch rollout을 저장합니다.
- `experiment.warmup_protocol(steps=128, learning_rate=1e-4)`로 **학습 데이터만 사용한 선택적 형식 SFT**를 먼저 할 수 있습니다. 실제 retriever가 반환한 관측에 정답이 있어야 합니다. 데모는 `evidence_ids`를 강제하고, 공식 QA는 해당 필드가 없으므로 검색 관측에서 정답 문자열이 확인되는 후보만 사용합니다. 이 필터는 의미적 근거 충분성을 보장하지 않습니다. 모델 action에만 loss를 적용하고 관측·질문은 마스킹합니다. 정답을 PPO rollout이나 평가 결과에 대신 넣지 않습니다.
- 기본 보상을 임의로 바꾸지 않았습니다. 형식 보상은 올바른 answer를 생성할 때만 적용되므로, 모든 생성이 invalid인 문제는 `format_reward`만 올려도 해결되지 않습니다.

수정된 `train.ipynb`는 초기 평가 → 별도 형식 SFT 128 update → 실제 sampled 생성 진단 → PPO 전 평가 → PPO → 최종 평가 순서입니다. SFT와 PPO의 개선을 구분해 보고합니다. 128 update가 형식 습득이나 정답률 상승을 보장하지는 않습니다. 사전 점검에서 유효한 답변이 없으면 raw trace를 먼저 확인하고 형식 학습을 보완하세요.

`warmup.jsonl`에는 별도 SFT loss/gradient/학습률이 저장됩니다. `diagnostics.jsonl`에는 파라미터를 갱신하지 않은 학습 질문 rollout 점검이 저장됩니다. Reference는 warmup 후에도 adapter를 끈 초기 base입니다. 따라서 warmup 이후 KL과 gradient가 생겼다는 사실만으로 정답률이 개선됐다고 판단하면 안 됩니다.

### 서버에 반영

이 수정은 로컬 `Desktop/Search-R1` 기준입니다. 서버 `/root/Search-R1`에 `policy_r1/`, `configs/`, `scripts/`, `requirements.txt`, `train.ipynb`를 반영하고 **서버 노트북 커널을 재시작**하세요. Remote SSH 창에서 실제 열린 경로와 `policy_r1.trainer.__file__` 출력이 일치해야 합니다. `root@/` 복사본은 수정 대상이 아닙니다.

기존 `outputs/first_experiment`와 checkpoint는 보존합니다. 노트북은 새 `outputs/search_r1_<시각>` 폴더를 사용합니다. 모두 0이었던 실험은 우선 새 실험으로 시작하세요. 이미 PPO를 실행한 객체에는 warmup을 추가하지 않습니다. 저장된 설정이 바뀌지 않는 정상 checkpoint 재개는 계속 지원합니다.

로컬 검증은 CPU의 작은 모델로 0-gradient 정체, 실제 SFT 가중치 갱신, reference/critic 동결, 마스킹과 진단을 검사합니다. 실제 HyperCLOVA CUDA/4-bit 재학습과 정답률 확인은 GPU 서버에서 수행해야 합니다.

## 파일 구조

| 파일 | 역할 |
|---|---|
| `train.ipynb` | 설치 확인 → 로그인 → 설정/데이터 → 모델 로딩 → 평가 → 학습 → 저장/재개 |
| `configs/search_r1.yaml` | 본 학습 데이터·검색·평가 설정 |
| `configs/first_experiment.yaml`, `train_demo.ipynb` | 기존 데모 실험 |
| `scripts/prepare_search_r1.py` | QA 다운로드·변환, Wikipedia 다운로드·인덱스 구축 |
| `policy_r1/prepare.py` | Parquet 스트리밍 읽기, 중복 정리, 검증 분리, manifest |
| `policy_r1/sqlite_search.py` | 스트리밍 인덱스 구축, 디스크 BM25 검색 |
| `scripts/evaluate.py` | 저장된 모델의 전체 val/test 평가 |
| `policy_r1/config.py` | 설정 읽기와 유효성 검사 |
| `policy_r1/data.py` | 문서/QA JSONL 검증, train/val/test 중복 확인 |
| `policy_r1/retrieval.py` | CPU BM25 및 선택 Dense 검색 |
| `policy_r1/protocol.py` | 한국어/영어 prompt, 검색·답변 tag 및 보상 |
| `policy_r1/models.py` | Actor, 독립 Critic, Value Head, action log probability |
| `policy_r1/rollout.py` | 검색과 생성을 반복하고 action mask 작성 |
| `policy_r1/objectives.py` | GAE/PPO/Value 수식 |
| `policy_r1/trainer.py` | 형식 SFT·진단·PPO 학습·평가·체크포인트·재개 |
| `policy_r1/warmup.py` | 학습 split의 실제 검색 기반 형식 SFT 예제와 action 마스크 |
| `scripts/check_environment.py` | GPU와 데이터 사전 검사 |
| `scripts/train.py` | 선택적 CLI 학습 |
| `tests/` | 수식·마스킹·검색·보상·작은 모델 업데이트 검증 |
| `data/demo/` | **실제 제도가 아닌 가상 정책** 동작 확인 데이터 |

Python/YAML/requirements에는 맨 위 역할 주석이 있습니다. 노트북 첫 Markdown 셀에도 역할을 적었습니다. JSONL은 주석을 허용하지 않으므로 각 행의 `_description` 필드로 설명합니다.

## conda `naver` 환경과 Cursor Remote SSH

로컬 Mac과 GPU 서버의 conda 환경은 서로 독립적입니다. 로컬 `naver` 환경은 코드 편집·CPU 테스트용이며, GPU 학습은 서버의 환경에서 실행합니다. `requirements.txt`의 `bitsandbytes==0.45.5`는 macOS용 배포가 없어 Linux에서만 설치하도록 조건을 지정했습니다.

로컬 환경을 사용하는 터미널:

```bash
conda activate naver
python -m pip check
```

Cursor에서는 Anysphere의 **Remote - SSH** 확장을 설치하고, `Cmd+Shift+P` → `Remote-SSH: Connect to Host...`에서 서버에 연결합니다. SSH 접속에 필요한 서버 IP, SSH 포트, 로그인 계정, 키 또는 비밀번호는 실제 서버 설정을 사용하세요. 접속 후 **서버의** `Search-R1` 폴더를 엽니다. Remote SSH가 로컬 프로젝트 파일이나 conda 환경을 자동 복사해 주지는 않습니다. 최초 코드는 아래 rsync 예시 또는 Git으로 옮깁니다.

서버에서 conda를 사용한다면 아래 명령을 실행합니다. 기존 `venv` 절차 대신 선택하는 방식입니다.

```bash
conda create -n naver python=3.11 pip -y
conda activate naver
cd ~/Search-R1
nvidia-smi
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m ipykernel install --user --name naver --display-name 'Python (naver, GPU server)'
```

서버의 NVIDIA 드라이버가 CUDA 12.4 wheel을 지원해야 합니다. Cursor의 원격 창에 Python/Jupyter 확장을 준비하고 `train.ipynb`의 **Select Kernel**에서 서버의 `naver` Python 환경을 선택합니다. 서버 터미널 또는 해당 노트북에서 `torch.cuda.is_available()`이 `True`인지 확인합니다. 현재 프로젝트의 학습 코드는 CUDA가 없으면 중단하도록 되어 있습니다.

구조는 `로컬 Cursor → SSH → GPU 서버의 Python/Jupyter → 서버 GPU`입니다. [Cursor 공식 Remote SSH 설명](https://prod.cursor.com/help/troubleshooting/network)을 참고하세요.

## Search-R1과 같은 데이터 받기

원본의 초기 quick start는 NQ 단독, `scripts/nq_hotpotqa/data_process.sh`의 혼합 학습은 **NQ + HotpotQA**입니다. 혼합 실험의 평가 대상은 NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle입니다.

| 용도 | 다운로드 위치 |
|---|---|
| 저자가 공개한 전처리 QA (`train.parquet`, `test.parquet`) | [PeterJinGo/nq_hotpotqa_train](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train/tree/main) |
| 원본 전처리 스크립트가 사용하는 QA 소스 | [RUC-NLPIR/FlashRAG_datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets) |
| 검색 대상 Wikipedia 2018 문서 (`wiki-18.jsonl.gz`, 약 5.12 GB 압축) | [PeterJinGo/wiki-18-corpus](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus/tree/main) |
| 원본 E5 검색 인덱스 (`part_aa`, `part_ab`, 합계 약 64.6 GB) | [PeterJinGo/wiki-18-e5-index](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index/tree/main) |

본 학습 준비는 다음 명령으로 한 번에 실행합니다. 기존 QA Parquet가 있으면 재사용하고 없는 파일만 다운로드합니다.

```bash
conda activate naver
python -m pip install -r requirements.txt
python scripts/prepare_search_r1.py --stage all --download
python scripts/check_environment.py --config configs/search_r1.yaml --data-only
python scripts/train.py --config configs/search_r1.yaml --epochs 1
```

노트북을 쓰면 마지막 명령 대신 `train.ipynb`를 위에서부터 실행하세요. 데이터 준비 셀도 포함되어 있습니다.
Wikipedia 압축 파일만 약 5.12 GB이고 SQLite 인덱스에는 추가 디스크가 필요합니다. 전체 구축은 수 시간 걸릴 수 있습니다.
압축을 별도 파일로 풀지 않고 순차적으로 읽으며 10,000문서마다 진행 상황을 출력합니다. 완성된 인덱스만 최종 경로로 옮깁니다.
중간 실패 시 다시 실행해 구축하며, 완료된 동일 입력은 재사용합니다. QA/인덱스 입력이 바뀌면 기존 결과를 덮어쓰지 않고 새 출력 경로를 요구합니다.

단계별 실행도 가능합니다.

```bash
python scripts/prepare_search_r1.py --stage qa
python scripts/prepare_search_r1.py --stage index --download
```

현재 로컬 Parquet 전체를 실제 변환한 결과:

| 분할 | 원본 행 수 | 변환 후 문항 수 | 용도 |
|---|---:|---:|---|
| train | 169,615 | 167,391 | NQ + HotpotQA PPO/SFT |
| val | train에서 분리 | 2,000 | 개발·튜닝·학습 중 검증 |
| test | 51,713 | 50,480 | 설정 확정 후 최종 평가 |

train 안에서 203개, test 안에서 1,233개의 중복 질문을 합쳐 정답 별칭을 보존했습니다.
train–test에 겹친 질문 21개는 train에서 제거한 뒤 seed 42로 train에서 검증 2,000문항을 분리했습니다.
질문 비교는 Unicode 정규화·대소문자·공백·문장부호를 정규화합니다. 의미가 유사한 질문이나 문서 단위 누수까지 검출하지는 않습니다.
원본 Parquet 파일은 수정하지 않습니다. 변환본 test는 중복 제거된 별도 평가 프로토콜이므로 공식 test의 원래 가중 평균과 같지 않습니다.

`data/search_r1/prepared/manifest.json`에는 원본/산출물 SHA-256, seed, 중복·겹침 수와 데이터셋별 개수가 기록됩니다.
`question`/`golden_answers`를 `question`/`answers`로 변환하고 원본 `prompt`를 재사용하지 않습니다.
`reward_model.ground_truth.target` 형태의 정답도 지원합니다. ID는 source/split/행 번호로 만들어 충돌을 피합니다.
잘못된 질문/정답은 조용히 버리지 않고 변환을 중단합니다.

본 학습은 `allow_demo: false`, 영어 prompt, 영어 QA EM, 디스크 BM25입니다. QA 파일만으로는 검색할 수 없으며
`data/search_r1/index/wiki18.sqlite`를 먼저 만들어야 합니다. 준비된 QA 경로를 바꾸면 config도 함께 수정하세요.
SQLite FTS5의 [BM25 검색](https://www.sqlite.org/fts5.html#the_bm25_function)을 사용하므로 원 논문의 E5 인덱스를 내려받을 필요는 없습니다.
E5와 동일한 검색 결과/성능은 아닙니다. 근거: [공식 QA](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train),
[Wikipedia corpus](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus),
[Search-R1 EM](https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py).

## Naver Cloud GPU 서버에서 실행

GPU는 로컬 노트북이 IP 주소로 직접 가져다 쓰는 장치가 아닙니다. **Jupyter의 Python 커널을 GPU 서버에서 실행하고 로컬 브라우저로 접속**합니다. 이미지의 서버를 사용할 수 있도록 아래 `GPU_HOST`, 계정, 키 경로를 본인 설정으로 바꾸세요. 서버 생성이나 과금 자원 변경은 이 코드가 수행하지 않습니다.

로컬 Mac 터미널에서 프로젝트 전송:

```bash
export GPU_HOST='서버의_공인_IP'
export GPU_USER='서버의_SSH_계정'
export GPU_KEY='/절대경로/서버키.pem'
rsync -av --exclude .git --exclude .venv --exclude outputs --exclude __pycache__ --exclude root@ --exclude data/search_r1 \
  -e "ssh -i $GPU_KEY" ./ "$GPU_USER@$GPU_HOST:~/Search-R1/"
ssh -i "$GPU_KEY" "$GPU_USER@$GPU_HOST"
```

Naver Cloud의 Linux 로그인 방식에 따라 키 대신 초기 비밀번호를 사용할 수도 있습니다. 비밀번호 방식이면 `-i` 옵션을 제외합니다. ACG/서버 방화벽의 SSH 포트는 접속 가능한 상태여야 합니다. 키와 비밀번호를 노트북 셀에 작성하지 마세요.

**서버 터미널**에서 Python 3.10/3.11 환경을 만듭니다. 아래 CUDA 12.4 wheel은 서버 NVIDIA 드라이버가 지원해야 합니다. 먼저 `nvidia-smi`가 정상 작동하는지 확인하세요.

```bash
cd ~/Search-R1
nvidia-smi
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m ipykernel install --user --name policy-r1 --display-name 'Policy-R1 (GPU)'
python scripts/check_environment.py
jupyter lab --ip=127.0.0.1 --port=8888 --no-browser
```

`venv` 모듈이 없는 Ubuntu 이미지에서는 관리 권한으로 `python3-venv`를 설치한 후 진행하세요. SSH 세션 종료 후에도 장시간 실행하려면 서버에서 `tmux` 세션 안에 Jupyter를 실행하세요.

다른 **로컬 터미널**에서 SSH 터널:

```bash
ssh -i "$GPU_KEY" -N -L 8888:127.0.0.1:8888 "$GPU_USER@$GPU_HOST"
```

브라우저에서 서버 Jupyter 로그에 나온 `http://127.0.0.1:8888/lab?token=...` URL을 열고 `train.ipynb` → **Policy-R1 (GPU)** 커널을 선택합니다. Jupyter는 loopback에만 바인딩하므로 8888 포트를 외부에 개방할 필요가 없습니다. 이미 서버 접속 방식이 있다면 VS Code Remote SSH로 연결해 같은 `.venv` 커널을 선택해도 됩니다.

## 노트북 실행 순서

1. 서버 GPU/패키지 확인, Hugging Face 모델 접근 인증.
2. `configs/search_r1.yaml` 로드, 새 output 폴더 생성. 재개 시 체크포인트 설정 사용.
3. QA 변환과 Wikipedia 인덱스 준비. 실제 train/val/test 개수와 검색 결과 확인.
4. Actor/Critic 로드, 질문 길이 확인. 전체 학습 질문을 최소 한 번 사용하는 update 목표 계산.
5. 초기 고정 validation subset 평가 → 선택적 형식 SFT 128 update → sampled rollout 진단 → PPO 전 평가.
6. PPO 실행. 500 update마다 고정 validation 200문항 평가·저장. 마지막 update도 저장.
7. 동일 subset으로 전후 비교, 전체 validation 2,000문항 평가. test는 최종 모델을 고른 뒤 별도 실행.

현재 변환본에서 1회 순회는 `ceil(167391 / 8) = 20924` update입니다. 마지막 배치에 다음 순회의 1문항이 포함됩니다.
각 순회 안에서는 중복 없이 질문을 섞습니다. `train_examples_seen`, `train_passes`에 실제 사용량이 기록됩니다.
학습 중 EM은 배치의 8응답에 대한 점수입니다. `eval_count=200`인 주기 검증과 전체 검증/test 점수를 구분하세요.
CLI에 `--epochs`를 생략하면 YAML의 `total_updates`를 사용하므로 원하는 학습량을 명시하는 편이 좋습니다.

SFT는 train에서 최대 2,048후보를 seed로 추출하고 실제 검색 관측에 정답이 보이는 최대 128예제를 사용합니다.
검색에 실패한 질문은 SFT에서만 제외하며 PPO의 train pool에서는 제거하지 않습니다.
학습 신호가 없는 경우 계속 실행하지 않고 trace를 남기고 중단합니다. 처음부터 높은 성능을 보장하는 설정은 아닙니다.

CLI 실행/재개/최종 평가:

```bash
python scripts/train.py --config configs/search_r1.yaml --epochs 1
python scripts/train.py --resume-from outputs/실험명/checkpoint-00500 --epochs 2
python scripts/evaluate.py --checkpoint outputs/실험명/checkpoint-20924 --split test
```

`--epochs 2`는 완료분을 포함한 총 목표 2회 순회입니다. 재개는 저장된 설정과 모델 revision, optimizer, RNG, 질문 순서를 복구합니다.
새 출력 폴더를 쓰며 `--output-dir`로 지정할 수도 있습니다. 데이터 내용/순서가 바뀌면 새 체크포인트의 재개를 거부합니다.
데모 체크포인트는 데이터·언어·길이 설정이 달라 본 학습에 그대로 재개하지 않습니다.

기존 작은 데모를 다시 실행하려면 `train_demo.ipynb` 또는 아래 명령을 사용하세요.

```bash
python scripts/train.py --config configs/first_experiment.yaml --warmup-steps 40
```

## 실제 정책 데이터로 교체

문서는 먼저 문단 단위로 나누어 JSONL로 저장하세요. 첫 버전의 BM25는 한국어 단어+2음절 조각 기반이며 형태소 분석기는 사용하지 않습니다. Dense는 작은 corpus의 모든 embedding을 CPU 메모리에 보관하는 exact search입니다.

`corpus.jsonl` (한 줄 = 검색 단위 문서):

```json
{"id":"document-001","contents":"정책명, 시행일, 대상, 금액 등 출처가 확인된 본문","source":"원문 URL","effective_date":"YYYY-MM-DD"}
```

`train.jsonl`, `val.jsonl` (한 줄 = 질문과 허용 정답):

```json
{"id":"train-001","question":"해당 제도의 지원 금액은?","answers":["20만 원","200000원"],"evidence_ids":["document-001"]}
```

`answers`와 `evidence_ids`는 Actor prompt에 넣지 않습니다. 정답 목록은 보상/평가에만 사용하고 evidence IDs는 데이터 검증에 사용합니다. EM은 자유로운 장문 설명을 평가하기 어려우므로 **첫 실험은 단답형 QA와 정답 별칭 목록**으로 준비하세요. 의미상 같은 답의 누락, 근거가 없는 답의 보상, 정책 버전 혼합 여부를 직접 점검하세요. 정답 보상만으로 근거 충실성을 보장하지 않습니다.

설정에서 세 경로를 바꾸고 `allow_demo: false`로 설정합니다. train/val id와 동일 질문 겹침은 자동 검사하며, 유사 질문이나 정책별 누수는 별도 검토해야 합니다. 실제 성능 평가에서는 문서/정책/시점 단위 분할과 충분한 평가셋을 권장합니다. 가상 데이터의 4개 train/2개 val 결과를 성능 수치로 보고하지 마세요.

Dense 전환:

```bash
python -m pip install -r requirements-dense.txt
```

그 다음 `retriever: dense`로 변경합니다. 기본 E5에 맞춰 query/passage prefix를 넣으므로 다른 embedding 계열을 사용할 때는 해당 prefix도 확인하세요.

## 출력과 검증

`outputs/<실험명>/`에는 설정, `metrics.jsonl`, `evaluation.jsonl`, `rollouts.jsonl`, `checkpoint-<update>/` 등이 생성됩니다. 형식 SFT 후에는 `checkpoint-00000/`도 저장합니다. checkpoint에는 Actor/Critic adapter, Value Head, optimizer, random state, 남은 질문 순서·누적 사용량, 데이터 fingerprint, tokenizer, resolved model revision, 라이브러리/GPU 정보가 있습니다. base 모델 전체를 중복 저장하지 않으므로 재로딩에는 모델 접근 권한 또는 로컬 캐시가 필요합니다.

```bash
python scripts/check_environment.py --config configs/search_r1.yaml --data-only
python -m pytest -q
```

`evaluation.jsonl`에는 평가 split·문항 수·데이터셋별 점수·evaluation_id가 기록되고,
`evaluation_examples.jsonl`에는 같은 evaluation_id로 개별 응답/검색 trace를 순차 저장합니다. test 전체를 메모리에 누적하지 않습니다.
최종 test는 NQ/TriviaQA/PopQA/HotpotQA/2WikiMultiHopQA/MuSiQue/Bamboogle의 데이터셋별 점수를 함께 확인하세요.

로컬에서 실제 QA 전체 변환을 완료했고, 자동 테스트는 Parquet schema/중복·누수 방지, gzip 인덱스/검색,
영어 EM, 실제 SFT/작은 모델 PPO, 고정 subset/전체 평가, 질문 순회와 저장·재개를 확인합니다.
전체 Wikipedia 다운로드/인덱스 성능과 HyperCLOVA CUDA 본 학습은 이 로컬 검증에 포함되지 않습니다.

테스트는 인터넷·모델 다운로드·GPU 없이 작은 무작위 Llama 모델을 생성합니다. 실제 HyperCLOVA CUDA/4-bit 학습 검증과 구분됩니다. 실제 학습에서는 loss뿐 아니라 EM, valid-answer 비율, 검색 횟수, reference KL, clip fraction, rollout trace를 확인하세요. valid-answer 비율이 계속 0이면 실제 모델의 tag 생성부터 점검하고, 필요하면 별도의 검색 형식 SFT를 먼저 준비하세요.

## 원본 Search-R1과의 관계

검토한 로컬 원본: `/Users/parksung-cheol/Desktop/car/Search-R1`, commit `598e61bd1d36895726d28a8d06b3a15bed19f5d3`.

참조한 흐름은 `train_ppo.sh`, `search_r1/llm_agent/generation.py`, `verl/trainer/ppo/core_algos.py`, `verl/utils/reward_score/qa_em.py`입니다. 현재 작업 폴더 `/Users/parksung-cheol/Desktop/Search-R1`에 독립 실행 가능한 구현을 작성했으며 원본은 수정하지 않았습니다.

이는 원본 veRL 실행기를 호출하거나 논문 실험을 그대로 재현하는 설정이 아닙니다. 검색 태그/관측 마스킹/정답 보상/PPO 원리를 유지하면서 단일 GPU 노트북용으로 구현했습니다. Ray/FSDP/vLLM 대신 Transformers + PEFT를 사용하고, 생성 action token으로 압축한 시간축에서 GAE를 계산하며, 길이/검색 제한 종료도 finite-horizon terminal로 취급합니다. 본 학습 EM은 영어 QA 정규화를 사용하고, 원본의 prompt 포함 문자열에서 두 번째 answer를 찾는 처리 대신 생성 action만 엄격히 파싱합니다.

참고: [Search-R1 코드](https://github.com/PeterGriffinJin/Search-R1), [논문](https://arxiv.org/abs/2503.09516), [모델 카드](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B), [PEFT 양자화 설명](https://huggingface.co/docs/peft/main/en/developer_guides/quantization).
