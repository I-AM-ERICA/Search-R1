<!-- 역할: 프로젝트 구조, 원본 Search-R1과의 관계, GPU 서버 연결 및 노트북 학습 절차를 설명합니다. -->
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
공개 `wiki-18.jsonl.gz`의 실제 내용은 GNU tar.gz이며 내부에 `wiki_dump.jsonl`이 들어 있습니다.
로더는 확장자 대신 파일 헤더로 JSONL/gzip JSONL/tar/tar.gz를 구분하고, tar 내부 JSONL을 추출 없이 읽습니다.
이전 로더의 `UnicodeDecodeError ... byte 0x80 in position 124`는 tar 크기 필드를 텍스트로 읽은 오류입니다.
이 경우 수정된 `policy_r1/sqlite_search.py`를 서버에 반영하고 같은 준비 셀 또는 `--stage index`를 다시 실행하세요.
다운로드된 corpus와 준비된 QA를 삭제하거나 다시 받을 필요는 없습니다.
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

### 2026-11-13: 실행 중 노트북 복구와 검색 성능 수정

2026-11-14에 전달받은 서버 원본을 비교한 결과, `trainer.py`와 CLI는 기존 버전과 같고
`sqlite_search.py`는 마지막 줄바꿈만 달랐습니다. 서버 원본의 정확한 SHA-256을 기준으로
`outputs/delivery/search-r1-search-fix.tar.gz`를 준비했습니다. 이 패키지는 검색·학습·CLI·벤치마크
4개 파일만 설치하며, 설치 전 원본 확인·백업과 설치 실패 시 복원을 수행합니다.
이미 저장한 `checkpoint-00111`이나 노트북 파일은 수정하지 않습니다.
실제 서버 검증은 아래처럼 질의별 실행 순서를 번갈아 두 번씩 측정합니다.

```bash
python scripts/benchmark_search.py --run-dir outputs/search_r1_20261113_134427_797919 \
  --queries 5 --repeats 2 --output outputs/search-benchmark-20261114.json
```

보고서의 `complete`, `summary.all_identical`, `summary.search_improved`를 확인합니다.
비교에는 결과 캐시를 사용하지 않으며 문서 순서·본문·점수를 모두 비교합니다. 서버의 OS 페이지
캐시를 강제로 비우지는 않습니다. 측정된 질의에 대한 결과이므로 전체 검색 성능을 보장하지 않습니다.
실제 서버에서 결과와 속도를 확인하기 전에는 수정된 코드로 장시간 학습을 재개하지 마세요.

`SQLiteRetriever.search`에서 오래 머무르는 현상을 진단했습니다. 기본 BM25 검색은
명시적인 점수 계산과 top-k 정렬을 사용하고, 최종 문서만 읽습니다. OR 질의, 토큰화,
BM25 점수 정의는 유지합니다. 읽기 캐시 256 MiB, mmap 상한 1 GiB, 최대 128개 질의의
LRU 캐시를 사용합니다. 실제 속도는 인덱스·SQLite 버전·디스크에 따라 달라집니다.
검색을 GPU 연산과 겹쳐 실행하는 변경은 아니므로 GPU 유휴 시간이 모두 없어지지는 않습니다.

이전 코드가 이미 실행 중인 커널에는 새 파일이 자동 반영되지 않습니다. 아래 복구 도구는
Linux에서 **rollout 수집 중 SQLite 검색**임을 확인하고 프로세스를 잠깐 정지시킨 뒤
같은 위치인지 다시 확인해 SIGINT를 보냅니다. 다른 학습 단계에서는 중단하지 않습니다.
예외 시 재개 처리와 별도 10초 재개 watchdog이 있습니다. 기존 커널을 종료하거나
수동 Interrupt 버튼을 누르기 전에 사용하세요. 로컬 테스트에서는 신호 순서를 모의 검증했으며,
실제 GPU 서버에서의 저장 완료 여부는 아래 출력과 체크포인트 파일로 확인해야 합니다.

서버 터미널 (`--pid`는 현재 GPU Python PID):

```bash
python scripts/pause_for_checkpoint.py --pid 202314
```

노트북의 학습 셀이 `KeyboardInterrupt`로 끝나면 **같은 커널의 새 셀**에서:

```python
%run -i scripts/recover_notebook.py
```

복구 코드는 실제 traceback·마지막 metrics·배치의 질문 수를 다시 검증합니다. 검증 실패 시
저장하지 않습니다. 완료한 PPO update의 Actor/Critic/optimizer를 보존하고, 미완료 배치의
질문은 다시 처리하도록 질문 순서와 사용량을 되돌립니다. 미완료 rollout에서 소비한 난수는
되돌릴 수 없으므로 이후 생성이 중단 없는 실행과 비트 단위로 같지는 않습니다.
`RECOVERY COMPLETE`와 `outputs/<실험명>/RECOVERED_CHECKPOINT`가 생긴 것을 확인한 뒤에만
기존 커널을 종료하세요. 저장 주기는 복구 체크포인트의 설정에도 10 update로 기록됩니다.

저장 후 서버 터미널에서 실제 로그의 질의로 검색 결과와 시간을 비교합니다. 이 검사는 GPU를 사용하지 않습니다.

```bash
conda activate naver
python scripts/benchmark_search.py --run-dir outputs/search_r1_20261113_134427_797919 \
  --output /tmp/search-r1-search-benchmark.json
```

`all_identical: true`인지, 실제 `speedup`이 개선되는지 확인한 후 기존 커널을 종료하고 tmux에서 재개합니다.
결과가 다르거나 더 느리면 재개 전에 원인을 확인하세요. `--skip-resume-checks`는 별도 검색 검증이
끝난 재개 작업에서만 사용하며, 긴 초기 평가와 중복 rollout 진단을 생략합니다.

```bash
tmux new -s search-r1
conda activate naver
cd ~/Search-R1
python scripts/train.py \
  --resume-from "$(cat outputs/search_r1_20261113_134427_797919/RECOVERED_CHECKPOINT)" \
  --skip-resume-checks
```

수정된 학습의 `metrics.jsonl`에는 `collect_seconds`, `search_seconds` (collection의 일부),
`update_seconds`, `step_seconds`, `search_cache_hits`가 기록됩니다. 새 실행을 안전하게 멈추려면
**새 실행 출력 폴더**에 다음 파일을 만드세요. 현재 update를 끝내고 저장한 뒤 종료합니다.
이 파일은 요청 증거로 남겨두므로 같은 객체에서 계속하려면 직접 삭제해야 합니다.

```bash
touch outputs/새_실험_폴더/STOP_AFTER_STEP
```

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
6. PPO 실행. 10 update마다 저장, 500 update마다 고정 validation 200문항 평가. 평가 전에 체크포인트를 저장하며 마지막 update도 저장합니다.
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

## SerpAPI 웹 검색 추론 서버와 am-erica 연결

완료한 `checkpoint-00150`의 Actor adapter를 base 모델과 함께 GPU에 올립니다. 프론트에
모델을 복사하지 않습니다. 요청 경로는 **프론트 → Spring 백엔드 → GPU의 PolicyAssistant
gRPC → 모델의 검색 action → SerpAPI Google 검색 → 모델 답변 → 백엔드 SSE → 프론트**입니다.
학습용 SQLite/Wikipedia 인덱스, Critic, optimizer, 학습 데이터는 이 추론 서버에서 읽지 않습니다.

서빙·진단 스크립트는 검색 결과의 **링크를 열어 HTML 본문을 읽고** 모델에 전달합니다. 검색 요약이
있어도 본문을 읽으며, 실패한 페이지를 검색 요약으로 대체하지 않습니다. 확인된 공식 호스트는
`official_page_excerpt`, 그 밖의 공개 사이트는 `page_excerpt`로 구분합니다. PDF·첨부파일·JavaScript 렌더링은 지원하지 않고,
공고의 신청 가능 여부를 자동 검증하지 않습니다. 모델에는 공식 공고를 우선하고 확인되지 않은
조건·금액·기간을 단정하지 않도록 지시합니다. 반환하는 sources는 모델에 전달된 검색 결과이며,
답변의 모든 문장이 출처와 일치함을 보장하는 인용 검증은 아닙니다. 한국어 정책 품질은 실제
체크포인트로 별도 확인해야 합니다. 학습 150단계 완료 자체는 그 검증을 대신하지 않습니다.

`serving/proto`와 `america/ai/v1`은 기존 am-erica-backend의 PolicyAssistant 규격과 일치합니다.
`Ask`, `AskStream`, 표준 health check, `x-internal-key` 인증을 제공합니다. `AskStream`은
검색과 답변 생성이 끝난 후 답변을 나눠 전송하며, 실시간 token streaming은 아닙니다.
한 번에 GPU 요청 하나를 처리하고 다른 요청은 RESOURCE_EXHAUSTED로 반환합니다.
요청 제한은 기본 85초, 검색은 최대 2회, 검색 결과는 기본 3개입니다. 같은 검색은 5분간
메모리에 캐시합니다. API 실패나 불완전한 action, 근거 없는 답변은 성공으로 위장하지 않습니다.

### 서버 설치와 검색 확인

로컬에서 생성된 `outputs/delivery/search-r1-serpapi-serving.tar.gz`를 Mac 터미널에서 복사합니다.

```bash
scp /Users/parksung-cheol/Desktop/Search-R1/outputs/delivery/search-r1-serpapi-serving.tar.gz root@223.130.161.182:/tmp/
```

GPU 서버에서 실행합니다. installer는 기존 rollout 코드가 예상 버전과 일치하는지 먼저 검사하고
코드를 백업합니다. 체크포인트와 학습 설정은 변경하지 않습니다. 버전 불일치가 나오면 강제로
덮어쓰지 말고 해당 파일을 비교해야 합니다. 이미 학습이 끝난 상태에서 새 Python 프로세스로 실행하세요.

```bash
cd /root/Search-R1
mkdir -p /tmp/search-r1-serpapi-serving
tar -xzf /tmp/search-r1-serpapi-serving.tar.gz -C /tmp/search-r1-serpapi-serving
/root/anaconda3/envs/naver/bin/python /tmp/search-r1-serpapi-serving/install.py /root/Search-R1
/root/anaconda3/envs/naver/bin/python -m pip install -r requirements-serving.txt
tmux new -s policy-api
```

새 tmux 창 **안에서** 아래를 실행합니다(서버의 bash 기준). 키는 화면과 명령 기록에 남지 않게
입력하며, 채팅·Git·프론트 환경변수에 넣지 않습니다. `AI_INTERNAL_KEY`는 SerpAPI 키와 별개로
백엔드와 추론 서버가 공유하는 임의의 긴 ASCII 비밀 문자열입니다.

```bash
cd /root/Search-R1
read -rsp 'SerpAPI key: ' SERPAPI_API_KEY; export SERPAPI_API_KEY; printf '\n'
read -rsp 'Internal key: ' AI_INTERNAL_KEY; export AI_INTERNAL_KEY; printf '\n'
/root/anaconda3/envs/naver/bin/python scripts/check_web_search.py
```

검색 확인은 실제 API 사용량을 소비합니다. `count`와 공식 출처 URL이 나오면 모델 서버를 실행합니다.
기존 학습 환경의 Torch/CUDA를 재설치하지 않습니다. base 모델 캐시 또는 모델 다운로드 권한이 필요합니다.

`count`는 빈 설명과 알려진 “이 페이지에 관한 정보가 없습니다” 안내문, Google `/goto`·`/url`
중계 링크를 제외한 결과 수입니다. 이것만으로 정책 근거의 정확성이 검증되지는 않으므로 제목·요약·URL도
확인하세요. `diagnostics`에는 전체 결과 수, 제외 사유, 응답 검색어가 요청 검색어와 일치하는지가
나옵니다. 0개라면 추론 서버 실행 전에 다음처럼 다른 검색어로 확인합니다. API 키나 원본 응답 전체는
공유하지 않아도 됩니다.

```bash
/root/anaconda3/envs/naver/bin/python scripts/check_web_search.py --query '소상공인 지원사업 모집공고 site:mss.go.kr'
```

검색 확인 명령은 기본 socket timeout 45초를 사용하며 `--timeout 60`처럼 바꿀 수 있습니다.
이는 진단 명령에만 적용되고 추론 서버의 검색 timeout 기본값은 15초입니다. 실패 시 `TIMEOUT`,
`DNS_ERROR`, `TLS_CERTIFICATE_ERROR`, `CONNECTION_ERROR`, `HTTP_PROTOCOL_ERROR`,
`HTTP_<상태코드>`, `INVALID_JSON` 등과 경과 시간을 표시합니다. URL·키·원본 오류 응답은
출력하지 않습니다. 통신/API 오류는 자동 재시도하지 않으며, 실제 오류에 따라 원인을 확인한 뒤 재실행하세요.

gRPC 검색 실패에도 동일한 오류 코드가 전달됩니다(예: `UNAVAILABLE: Web search unavailable [TIMEOUT]`).
`PROVIDER_ERROR`와 `search_id`가 나오면 SerpAPI 대시보드에서 해당 검색 응답을 확인하세요.
서버 로그에도 코드와 경과 시간만 남기며 API 키·원본 오류 문구·검색어는 출력하지 않습니다.
SerpAPI의 `status=Success`와 명시적인 결과 없음 메시지는 빈 관측으로 전달하여 모델이 제한 내에서
검색어를 바꿀 수 있게 합니다. 계정/통신 오류를 빈 결과로 바꾸지는 않습니다.
[SerpAPI 오류 응답 규격](https://serpapi.com/api-status-and-error-codes).

Google 응답에 문제가 있으면 같은 SerpAPI 계정으로 네이버 웹 검색을 선택해 별도로 확인할 수
있습니다. 검색엔진을 자동 전환하지 않습니다. 아래 진단 명령은 실제 검색 1회를 요청합니다.

```bash
/root/anaconda3/envs/naver/bin/python scripts/check_web_search.py \
  --engine naver --timeout 45 --query '소상공인시장진흥공단 정책자금 지원'
```

네이버는 실제 서버 응답이 확인된 `engine=naver`, `query`, `where=nexearch`(통합검색)로 요청하고
`web_results`를 사용합니다. 광고·이미지·지도·쇼핑·AI 요약은 섞지 않습니다. `diagnostics`의 엔진, `result_count`,
`elapsed_seconds`와 실제 출처 내용을 확인하세요. 네이버도 서비스 상태와 검색어에 따라 실패하거나
부적절한 결과를 줄 수 있습니다. [SerpAPI 네이버 공식 문서](https://serpapi.com/naver-search-api).

본문 읽기는 공개 HTTPS 사이트에 적용하고, 확인된 공식 호스트(`www.semas.or.kr`,
`semas.or.kr`, `ols.semas.or.kr`, `www.mss.go.kr`, `www.bizinfo.go.kr`)를 우선합니다.
검색당 최대 3페이지, 페이지당 최대 8초·1MiB, 본문 최대 5,000자로 제한합니다.
공개 IP를 확인해 고정하고 HTTPS 인증서를 검증하며 리다이렉트는 따라가지 않습니다.
`curl` 명령이 필요하지만 Python 추가 패키지는 필요하지 않습니다. 메뉴·스크립트 등을 제거하고
본문 영역이 확인될 때만 사용합니다. 접속/본문 추출 실패는 `page_fetch_failures`에 표시하고
빈 본문을 근거로 만들지 않습니다. 검색 요약이 이미 있는 결과도 원문을 가져옵니다.
모델의 관측 토큰 제한으로 본문은 더 잘릴 수 있으며, 원문 전체나 첨부파일을 읽은 것으로 취급하지 않습니다.
여러 페이지에 관측 예산을 나눠 주어 첫 번째 긴 문서가 다른 출처를 모두 밀어내지 않게 합니다.
라이브러리 `SerpAPIRetriever`의 기본값은 기존 진단 호환용 요약 모드이며, 서빙·진단 스크립트는
`require_page_evidence=True`를 명시합니다.

프론트는 마지막 완료 답변의 출처 URL 최대 2개를 이전 대화에 `이전 답변 출처 URL:`로 전달합니다.
요약·내용 설명 후속 질문은 이 URL의 본문을 다시 읽거나 5분 이내 페이지 캐시를 사용합니다.
사용자가 보낸 이전 답변 자체는 근거로 삼지 않으며, 자료를 읽지 못하면 요약을 거절합니다.
본문을 받은 후에도 모델이 링크 안내만 출력하면 같은 본문으로 한 번만 다시 요약을 요청합니다.
체크포인트나 학습 설정은 바꾸지 않으며, 실제 GPU의 다양한 질문에 대한 품질 확인은 별도입니다.

공식 HTML 본문은 제목/소제목의 단락 경계를 보존합니다. 모델이 긴 표를 복사·반복하는 문제를
줄이기 위해, 질문의 단어·구문과 맞는 짧은 단락이 확인되면 그 단락을 **원문 그대로** 골라 관측에
넣습니다. 바로 이어지는 제외·주의 단락도 유지합니다. 숫자·자격 조건·답변을 새로 만들어 붙이지
않으며, 검색 요약에는 이 처리를 적용하지 않습니다. 분명한 단락을 고르지 못하면 기존 발췌를
유지합니다. 다른 단락의 조건이 생략될 수 있으므로 이를 전체 공고 검토로 간주해서는 안 됩니다.
진단의 `evidence_focus`에 원래 길이·선택한 길이·소제목이 표시됩니다. 생성 지시는 질문한 항목에
2~5개의 짧은 문장이나 항목으로 답하고 종료하도록 했으나, 준수 여부와 사실 정확성은 실제 모델로 검증해야 합니다.

추론에서는 모델이 검색어에 복사한 **해당 요청의 정확한 기준일·지역·업종 접두부**와 문장 끝의
“공식 자료로 확인해줘” 같은 요청 표현을 정리합니다. 접두부 필드 사이와 끝의 공백·`/`·`|`·`;`·`,`
구분자를 처리하되, 검색어 내부의 URL이나 날짜 구분자는 보존합니다. 질문 본문에 직접 포함된 지역·연도·업종·
제외 조건은 일괄 삭제하지 않습니다. 정리한 검색어의 결과가 비었을 때만 `공통 지원 자격`을
`지원`으로 넓히는 제한된 보완 검색을 한 번 시도합니다. 기관명이나 답변을 하드코딩하지 않으며
전혀 다른 정책으로 바꾸지 않습니다. 한 사용자 요청의 검색기 호출은 합계 최대 2회입니다.
동일 검색어는 요청 내 결과를 재사용하고, API/네트워크 오류에는 이 보완 검색을 적용하지 않습니다.
`searches`는 모델 search action 수, `search_requests`는 보완 검색을 포함한 검색기 호출 수입니다.
이는 추론 환경의 보정이며 PPO 체크포인트 자체의 검색 능력이 향상됐다는 의미는 아닙니다.

최신 사용자 질문에서 “공식 자료/문서/출처/공고/사이트/홈페이지”를 요청하면 지원 중인 공식
호스트 목록에 포함된 HTTPS URL만 모델의 관측과 반환 sources에 넣습니다. 제목에 기관 이름이
있다는 이유로 공식 출처로 인정하지 않습니다. 후보는 최대 5개를 검사하고 요청한 top_k 이내로
반환합니다. 공식 후보가 없으면 빈 결과와 동일하게 제한된 보완 검색을 적용하며, 끝내 없으면
근거 없는 답변을 반환하지 않습니다. 지원 목록 밖의 실제 공식 기관도 제외될 수 있으므로 이는
전국 모든 기관을 포괄하는 공식 출처 판별기가 아닙니다. 도메인이 확인돼도 공고의 최신성이나
질문과의 관련성, 답변 내용 일치 여부까지 검증된 것은 아닙니다.

네이버 검색 확인이 통과하면 아래 서버 실행 명령에 `--search-engine naver`를 추가합니다.
필요하면 `--search-timeout 30`으로 검색 socket timeout을 조정할 수 있으나, 전체 요청 85초
제한에는 모델 생성 시간도 포함됩니다. 모델의 검색 action/관측/답변 흐름과 체크포인트는 그대로입니다.

```bash
/root/anaconda3/envs/naver/bin/python -u scripts/serve_policy.py \
  --checkpoint outputs/search_r1_until_00150_20261114_125545_672828/checkpoint-00150
```

`POLICY SERVER READY`가 나오면 `Ctrl-b` 후 `d`로 tmux에서 나옵니다. 다른 서버 터미널에서
동일한 internal key를 입력하고 실제 모델 요청을 확인합니다. 이 요청도 SerpAPI 사용량을 소비할 수 있습니다.

```bash
cd /root/Search-R1
read -rsp 'Internal key: ' AI_INTERNAL_KEY; export AI_INTERNAL_KEY; printf '\n'
/root/anaconda3/envs/naver/bin/python scripts/ask_policy.py \
  --question '서울 음식점 소상공인이 확인할 수 있는 지원사업의 공식 공고를 찾아줘. 신청 기간이 불확실하면 알려줘.'
```

답변과 sources를 읽고 실제 공고의 지역·대상·기간과 비교하세요. 실패하면 오류 코드와 메시지를
먼저 확인합니다. health의 SERVING은 모델 로딩과 서버 시작을 뜻하며 API 잔여량이나 답변 품질까지
보증하지 않습니다. 로컬 자동 테스트는 가짜 검색 응답/작은 모델과 실제 localhost gRPC를 사용하며,
GPU 체크포인트와 유료 API의 통합 검증은 위 단계로 별도 수행합니다.

`No retrieved evidence` 오류는 검색을 생략한 답변, 검색 결과 없음, 관측에서 출처 식별자가
잘리는 경우 등을 구분해야 합니다. 서버를 `Ctrl-C`로 종료한 후 같은 GPU 셸에서 아래 진단을
실행하면 한 요청의 검색어·검색 결과 개수·관측에 남은 출처·종료 사유를 확인할 수 있습니다.
출력에는 모델이 생성한 검색어가 포함됩니다. 전체 답변·원문·API 키는 출력하지 않습니다.
실제 모델과 검색 API를 사용하며 재학습하거나 체크포인트를 변경하지 않습니다.
`source_selection`은 검색엔진 후보와 공식 출처 필터를 통과해 모델에 전달한 결과를 구분합니다.
`--show-answer`를 추가하면 최종 답변과 채택한 출처 URL도 출력해 내용까지 검토할 수 있습니다.
진단의 `status=success`는 검색-관측-답변 흐름이 끝났다는 뜻이며 정책 답변의 정답 판정이 아닙니다.

공식 근거가 전달됐는데 `generation_limit`으로 끝나면, 생성 한도 부족과 반복/형식 오류를
구분해야 합니다. 진단과 서버 모두 `--max-new-tokens 512`처럼 추론 한도만 바꿀 수 있습니다.
옵션을 생략하면 체크포인트 설정(현재 192)을 유지하며, 저장된 설정이나 학습 가중치를 수정하지
않습니다. `--show-invalid-output`은 진단에서만 실패 출력 앞뒤 각 240자와 답변 태그 여부를
표시합니다. 미완성 출력을 정상 답변으로 반환하거나 닫는 태그를 임의로 붙이지 않습니다.
한도를 늘려도 전체 context/요청 시간 제한은 유지되며, 속도와 메모리 사용량이 늘 수 있습니다.

```bash
/root/anaconda3/envs/naver/bin/python scripts/diagnose_policy.py \
  --checkpoint outputs/search_r1_until_00150_20261114_125545_672828/checkpoint-00150 \
  --max-new-tokens 512 --show-answer --show-invalid-output
```

```bash
/root/anaconda3/envs/naver/bin/python scripts/diagnose_policy.py \
  --checkpoint outputs/search_r1_until_00150_20261114_125545_672828/checkpoint-00150
```

### 로컬 Spring 백엔드 연결

서버는 `127.0.0.1:50051`에만 바인딩합니다. Mac의 별도 터미널에서 SSH 터널을 유지합니다.

```bash
ssh -N -L 50061:127.0.0.1:50051 root@223.130.161.182
```

백엔드 실행 환경에 다음 값을 설정하고 재시작합니다. `AI_INTERNAL_KEY`는 GPU 서버와 같은
값을 IDE의 비공개 실행 환경 또는 로컬 환경변수로 설정합니다. SerpAPI 키는 GPU에만 필요합니다.

```text
AI_MOCK_ENABLED=false
AI_POLICY_HOST=127.0.0.1
AI_POLICY_PORT=50061
AI_GRPC_DEADLINE=90s
AI_GRPC_TLS=false
```

기존 프론트의 policy SSE 요청 경로를 그대로 사용할 수 있습니다. 프론트 timeout 120초 안에
서버 85초/백엔드 90초 제한이 들어갑니다. **현재 백엔드의 mock 스위치는 전체 AI 서비스 공통**이므로
false로 바꾸면 홍보·고객·경영 mock도 꺼집니다. 해당 서버가 없다면 그 기능과 통합 health가
실패할 수 있습니다. 이 변경에는 백엔드/프론트 코드를 수정하거나 다른 AI 서비스를 배포하는 작업은
포함하지 않았습니다. SSH 터널 방식은 로컬 연동용입니다.

SerpAPI 필드와 설정: [공식 Search API](https://serpapi.com/search-api),
[Organic results](https://serpapi.com/organic-results).

## 원본 Search-R1과의 관계

검토한 로컬 원본: `/Users/parksung-cheol/Desktop/car/Search-R1`, commit `598e61bd1d36895726d28a8d06b3a15bed19f5d3`.

참조한 흐름은 `train_ppo.sh`, `search_r1/llm_agent/generation.py`, `verl/trainer/ppo/core_algos.py`, `verl/utils/reward_score/qa_em.py`입니다. 현재 작업 폴더 `/Users/parksung-cheol/Desktop/Search-R1`에 독립 실행 가능한 구현을 작성했으며 원본은 수정하지 않았습니다.

이는 원본 veRL 실행기를 호출하거나 논문 실험을 그대로 재현하는 설정이 아닙니다. 검색 태그/관측 마스킹/정답 보상/PPO 원리를 유지하면서 단일 GPU 노트북용으로 구현했습니다. Ray/FSDP/vLLM 대신 Transformers + PEFT를 사용하고, 생성 action token으로 압축한 시간축에서 GAE를 계산하며, 길이/검색 제한 종료도 finite-horizon terminal로 취급합니다. 본 학습 EM은 영어 QA 정규화를 사용하고, 원본의 prompt 포함 문자열에서 두 번째 answer를 찾는 처리 대신 생성 action만 엄격히 파싱합니다.

참고: [Search-R1 코드](https://github.com/PeterGriffinJin/Search-R1), [논문](https://arxiv.org/abs/2503.09516), [모델 카드](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B), [PEFT 양자화 설명](https://huggingface.co/docs/peft/main/en/developer_guides/quantization).
