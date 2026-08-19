# LeRobot Episode Inspector

Pi0 (π0) VLA를 파인튜닝하기 전에, 수집한 에피소드가 제대로 녹화됐는지 검수하는 GUI.

탑/전면 뷰와 팔목 뷰 영상, INSTRUCTION, 조인트 궤적을 한 화면에서 동기 재생하고,
에피소드마다 자동 품질 검사를 돌린 뒤 keep/drop 판정을 남깁니다.
판정 결과는 JSON으로 뽑아 학습에서 불량 에피소드를 제외하는 데 씁니다.

LeRobot v2.0 / v2.1 / v3.0 데이터셋 레이아웃과, `meta/` 없이 parquet만 있는 경우를 지원합니다.

---

## 왜 필요한가

에피소드가 조용히 실패하는 경우가 많습니다. 팔이 안 움직인 채로 30초가 녹화되거나,
그리퍼가 헛잡거나, 텔레옵이 끊겨 프레임이 드롭되거나, instruction이 비어 있거나.
이런 에피소드는 그냥 쓸모없는 게 아니라 **"이 지시에는 가만히 있어라"를 학습시켜
정책을 망칩니다.** 수백 개를 모은 뒤에 발견하면 늦습니다.

---

## 실행

데이터가 있는 머신(젯슨)에서 서버를 띄우고 PC 브라우저로 접속하는 걸 권장합니다.
영상을 복사할 필요가 없고, HTTP Range 스트리밍이라 스크럽할 때 필요한 구간만 전송됩니다.
녹화 직후 바로 검수할 수 있는 것도 장점입니다.

**젯슨에서:**

```bash
pip install fastapi uvicorn pandas pyarrow numpy
python server.py --data /home/sejong/lerobot_datasets/piper_pick --host 0.0.0.0 --port 8000
```

**PC 브라우저에서:** `http://<젯슨-IP>:8000`

[uv](https://docs.astral.sh/uv/)가 있으면 의존성 설치 없이 바로 실행됩니다
(스크립트에 PEP 723으로 인라인 명시돼 있습니다):

```bash
uv run server.py --data <경로> --host 0.0.0.0
```

**PC에 복사해서 로컬로 보려면:**

```powershell
scp -r sejong@<젯슨-IP>:/home/sejong/lerobot_datasets/piper_pick D:\datasets\
uv run server.py --data D:\datasets\piper_pick
```

### 옵션

| 옵션 | 설명 |
|---|---|
| `--data` | 데이터셋 루트 디렉터리, 또는 단일 `.parquet` 파일 |
| `--host` | 다른 머신에서 접속하려면 `0.0.0.0` (기본 `127.0.0.1`) |
| `--port` | 기본 `8000` |
| `--review` | 판정 저장 위치 (기본 `<root>/episode_review.json`) |

---

## 구성

| 파일 | 역할 |
|---|---|
| [server.py](server.py) | 데이터셋 로더 + 품질 검사 + FastAPI 서버 |
| [static/index.html](static/index.html) | GUI 전체 (의존성 없음, 캔버스로 직접 그림) |
| [mp4probe.py](mp4probe.py) | mp4 컨테이너 파서 — 길이·프레임수·해상도·코덱 |
| [make_local_dataset.py](make_local_dataset.py) | 흩어진 parquet+mp4를 v3.0 레이아웃으로 조립 |

프론트엔드는 CDN을 포함해 외부 의존성이 전혀 없습니다. 로봇이 인터넷에 연결돼 있지
않아도 동작해야 하므로, 차트도 라이브러리 없이 캔버스로 직접 그립니다.

---

## 화면

- **INSTRUCTION 배너** — 이 에피소드의 task 문자열. 비어 있으면 빨갛게 표시됩니다.
- **영상 그리드** — 카메라 스트림 수만큼 자동 배치. 공통 클럭으로 동기되고,
  0.12초 이상 벌어지면 자동 보정합니다.
- **조인트 궤적** — 조인트별 한 행, `observation.state`(실선) vs `action`(점선).
  클릭·드래그로 해당 시점 탐색. 커서 위치의 상태값이 실시간 표시됩니다.
- **Quality checks** — 아래 9가지 자동 검사 결과.
- **Verdict** — Good / Bad / Unsure + 메모. 즉시 저장됩니다.

### 단축키

| 키 | 동작 |
|---|---|
| `Space` | 재생 / 정지 |
| `←` `→` | 1프레임 이동 (`Shift` 조합 시 10프레임) |
| `J` `K` | 이전 / 다음 에피소드 |
| `G` `B` `U` | Good / Bad / Unsure 판정 |

에피소드가 100개를 넘어가면 사이드바의 **`flagged` 필터**로 warn/fail만 걸러
`J`/`K`로 넘기며 `G`/`B`로 판정하는 방식이 빠릅니다.

---

## 지원 레이아웃

| 레이아웃 | 데이터 | 인스트럭션 | 영상 |
|---|---|---|---|
| **v3.0** | `data/chunk-000/file-000.parquet` | `meta/tasks.parquet` + `meta/episodes/` | `videos/{key}/chunk-000/file-000.mp4`, 에피소드별 timestamp 윈도우 |
| **v2.0 / v2.1** | `data/chunk-000/episode_000000.parquet` | `meta/tasks.jsonl` + `meta/episodes.jsonl` | `videos/chunk-000/{key}/episode_000000.mp4` |
| **meta 없음** | parquet 스캔 | 없음 (경고 표시) | 파일명 매칭 시도 |

### v3.0의 파일 하나 = 에피소드 여러 개

v2.x는 에피소드당 파일 하나였지만, 수천 에피소드를 모으면 작은 파일이 수만 개가 되어
파일시스템과 HF 허브 양쪽에서 느려집니다. 그래서 v3.0은 여러 에피소드를 한 파일에 묶습니다.

```
 index  episode_index  frame_index  timestamp
   897              0          897  29.900000   ← 에피소드 0 마지막 행
   898              1            0   0.000000   ← 에피소드 1 시작, frame_index·timestamp 리셋
```

| 컬럼 | 동작 |
|---|---|
| `episode_index` | 이 행이 몇 번 에피소드인지 (데이터셋 전체에서 유일) |
| `frame_index` | 에피소드마다 0으로 리셋 |
| `timestamp` | 에피소드마다 0.0으로 리셋 |
| `index` | 데이터셋 전체 통짜 카운터, 리셋 안 됨 |

영상도 같습니다. mp4 하나에 여러 에피소드가 연달아 들어가고, 각 에피소드의 위치는
`meta/episodes/`의 `videos/{key}/from_timestamp` / `to_timestamp`에 기록됩니다.
GUI는 에피소드를 고를 때 영상을 처음부터 트는 게 아니라 해당 오프셋으로 점프시킵니다.

파일이 일정 크기를 넘으면 `file-001.parquet` / `file-001.mp4`로 롤오버되고,
새 비디오 파일에서는 `from_timestamp`가 0부터 다시 시작합니다. 이것도 처리됩니다.

에피소드 목록과 길이는 **항상 parquet에서 직접** 읽습니다 — `meta/`가 오래됐어도
실제 데이터가 기준입니다. 같은 `episode_index`가 두 파일에 나타나면(부분 재녹화,
잘못된 병합) 시작 시 경고를 출력합니다.

---

## 자동 검사 항목

| 검사 | 잡아내는 문제 |
|---|---|
| Instruction | task 문자열 누락 — Pi0는 언어 조건부라 비면 학습에 못 씀 |
| Duration | 데이터셋 중앙값 대비 절반 미만 / 2배 초과 (녹화 잘림, 끝부분 방치) |
| Frame timing | 타임스탬프 역행, 프레임 드롭 (영상-상태 싱크 깨짐) |
| NaN / Inf | 손상된 값 |
| **Motion** | **죽은 녹화** — 전 조인트 정지, 또는 잠깐 움직이고 나머지 내내 정지. 시작/끝 유휴 구간 |
| Gripper | 그리퍼 미작동 = 대개 실패한 시도. open/close 횟수 |
| Action vs state | 조인트별 추종 오차 — 리더-팔로워 불일치 감지 |
| Discontinuities | 한 프레임에 가동범위 15% 이상 점프 (텔레옵 글리치, 엔코더 튐) |
| Cameras | 스트림 누락, 해상도·코덱, 영상 길이/fps와 상태 불일치 |

### Motion 검사의 임계값

가장 중요한 검사이고, 가장 틀리기 쉬운 검사이기도 합니다.

프레임 간 차이를 `1e-4` 같은 값으로 판정하면 **정지한 팔의 엔코더 노이즈를 움직임으로 세어
죽은 녹화를 통과시킵니다.** 실제로 이 프로젝트에서 처음 그렇게 만들었다가,
30초 내내 멈춰 있던 에피소드가 `ok`로 나오는 걸 보고 고쳤습니다.

지금은 이렇게 판정합니다:

- 조인트 가동범위가 `STATIC_JOINT_RANGE`(1°) 미만이면 그 조인트는 정지로 간주
- 전 조인트가 정지 → **fail** (죽은 녹화)
- 0.25초 창(window) 동안의 변위로 움직임 판정 — 프레임 간 차이가 아니라
  창 단위로 봐야 느린 동작은 살리고 센서 노이즈는 거릅니다
- 움직인 구간이 전체 길이의 20% 미만 → **fail** (잠깐 움직이고 멈춘 것도 죽은 녹화)

같은 이유로 추종 오차와 점프 검사도 **실제로 움직인 조인트에만** 적용합니다.
가동범위 0에 가까운 조인트를 분모로 쓰면 `1066667%` 같은 값이 나옵니다.

임계값은 [server.py](server.py) 상단 상수로 조정합니다 — Piper의 상태 단위(도) 기준입니다.

---

## 영상 코덱

LeRobot v3.0은 기본적으로 **AV1**로 인코딩합니다.
Chrome / Edge / Firefox는 재생되지만 **Safari는 안 됩니다.**
코덱은 QC의 Cameras 항목에 표시되고, AV1이면 경고가 붙습니다.

mp4의 길이·프레임 수·해상도·코덱은 [mp4probe.py](mp4probe.py)가 컨테이너 박스를
직접 읽어 가져옵니다. ffprobe 설치가 필요 없고, 헤더만 읽으므로 수백 MB 파일도 즉시
처리합니다. `moov`가 파일 끝에 있는 경우(faststart 아님)도 처리합니다.

---

## 흩어진 파일로 미리보기 만들기

parquet과 mp4만 손에 있고 `meta/`가 없을 때, v3.0 레이아웃으로 조립합니다:

```bash
uv run make_local_dataset.py \
  --parquet file-000.parquet \
  --video observation.images.front=front.mp4 \
  --video observation.images.wrist=wrist.mp4 \
  --task "pick up the object and place it in the box" \
  --out ./piper_pick
```

`--probe-only`를 주면 아무것도 만들지 않고 영상 스펙과 parquet 대조 결과만 출력합니다:

```
observation.images.front
    640x480, AV1, 68.87s, 2066 frames, 30fps
3 episode(s), 2066 frames @ 30 fps
    observation.images.front: video 68.87s vs state 68.87s  (OK, drift 0.00s)
    observation.images.front: video 2066 frames vs state 2066 frames  (OK)
```

에피소드 경계·길이·fps는 parquet에서 읽고, 영상 timestamp 윈도우는 각 에피소드
길이를 누적해 계산합니다.
**`--task`는 수동 입력이므로, 실제 instruction은 로봇의 `meta/tasks.parquet`이 정본입니다.**

---

## 학습에서 불량 에피소드 제외

판정은 `<dataset>/episode_review.json`에 즉시 저장됩니다.
헤더의 **Export keep/drop** 버튼 또는 `GET /api/export`로 받습니다:

```json
{
  "keep": [0, 2, 3],
  "drop": [1],
  "reviews": { "1": { "status": "bad", "note": "그리퍼 헛잡음" } }
}
```

```python
import json
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

review = json.load(open("piper_pick/episode_review.json"))
drop = {int(k) for k, v in review.items() if v["status"] == "bad"}
ds = LeRobotDataset("piper_pick", episodes=[i for i in range(n_episodes) if i not in drop])
```

`drop`에는 `bad`로 판정한 것만 들어갑니다. 미검토와 `unsure`는 `keep`에 남으므로,
전부 훑기 전에 뽑으면 검수 안 한 에피소드가 그대로 포함됩니다.

---

## API

| 엔드포인트 | 용도 |
|---|---|
| `GET /api/dataset` | 레이아웃, fps, 로봇 타입, 카메라 키, 에피소드 수 |
| `GET /api/episodes` | 에피소드 목록 + QC 등급 + 판정 상태 |
| `GET /api/episodes/{i}` | 상태/액션 시계열, 영상 윈도우, QC 상세 |
| `GET /api/video/{i}/{key}` | mp4 스트리밍 (Range 지원) |
| `POST /api/review/{i}` | `{"status": "good\|bad\|unsure", "note": "..."}` |
| `GET /api/export` | keep/drop 목록 |

---

## 검증 결과 (2026-08-19)

Piper 7-DoF(6축 + 그리퍼), 30fps 실제 수집 데이터로 확인했습니다.

**단일 에피소드 parquet** (`file-000.parquet`, 896프레임) — `meta/`와 영상 없이
parquet만으로 로드, fps 추론, 조인트 이름 생성 동작 확인.

**3 에피소드 + 영상** (`file-000-pick.parquet` + front/wrist mp4) —
2066프레임 @ 30fps = 68.867초로 영상 길이·프레임 수와 정확히 일치. 640x480 AV1.
윈도우 분할(0–29.93 / 29.93–59.83 / 59.83–68.87) 정확.

| ep | 프레임 | 판정 | 내용 |
|---|---|---|---|
| 0 | 898 | warn | 정상. 7/7 조인트 동작, 그리퍼 0→98.4 (2회 개폐) |
| 1 | 897 | **fail** | 0.2초만 움직이고 29.6초 정지. 그리퍼 미작동 |
| 2 | 271 | **fail** | 전 조인트 가동범위 0.17° 미만. 9초간 완전 정지 |

에피소드 1·2는 팔이 사실상 움직이지 않은 죽은 녹화입니다. 두 에피소드의 정지 자세가
서로 동일한 것으로 보아 텔레옵이 연결되지 않은 상태로 녹화가 돈 것으로 보입니다.

**다중 파일** (합성 픽스처, 40 에피소드 / 데이터 파일 2개 / 카메라당 비디오 파일 2개) —
파일 롤오버 시 `t0` 리셋 확인. 전 에피소드 QC 계산 0.1초.
단, 이는 작은 파일 기준입니다. 실제 100MB급 parquet은 메모리에 2개까지만 올리므로
파일이 많으면 첫 로딩에 파일당 1~2초씩 들 수 있습니다 (실측 아님).

**HTTP Range** — 일반 범위, suffix 범위(`bytes=-500`), 열린 범위(`bytes=N-`),
범위 초과 시 416 응답 확인.

---

## 알려진 제약

- 에피소드가 여러 데이터 파일에 걸쳐 나뉘어 있는 경우는 지원하지 않습니다
  (v3.0은 그렇게 저장하지 않습니다).
- 100MB급 실제 파일에서의 첫 로딩 시간은 실측하지 않았습니다.
- 프레임 단위 이미지 확인은 영상 디코딩에 의존하므로, 브라우저가 코덱을 지원해야 합니다.
