# 실험 실행과 확장

## 1. 실행 환경

기본 비교는 Python 3.13과 `Research/requirements.txt`의 버전을 기준으로 합니다. 저장소 루트에서 가상환경을 활성화한 후 설치합니다.

```bash
python -m pip install -r Research/requirements.txt
python Research/run.py check
```

27개 입력 파일은 SHA-256으로 고정했습니다. 입력을 바꾼 경우 `check`가 변경을 보고하는 것이 정상입니다. 자기 실험의 입력과 해시는 별도로 관리하세요. 비교 실행은 스냅샷 해시 일치를 강제하지 않으며 구조와 분할을 검증하고 실제 사용한 입력 해시를 기록합니다.

## 2. 첫 실험과 주 비교

```bash
python Research/run.py baseline
python Research/run.py compare
```

기본 설정은 정상 학습 5,000개, 실제 불량 seed 247개, 조건별 추가 불량 750개, RF 300 trees, min_samples_leaf=2, balanced_subsample, RF seed=42, 정상 추출 seed=65입니다. 검증 F1을 최대화하는 임계값을 선택하고 동률이면 재현율, 정밀도 순으로 비교합니다.

원본 행 메타데이터로 학습·참조 자료와 겹치는 검증 윈도우를 제거한 뒤 평가 윈도우에서도 중첩을 제거합니다. 현재 입력에서는 검증 1,408개, 평가 3,080개가 남습니다. 평가 집합을 임계값 선택에 사용하지 않습니다.

결과는 새로운 `Research/runs/<실행ID>/`에 저장합니다.

| 파일 | 내용 |
|---|---|
| main_comparison.csv | 방법별 임계값·분류 지표·혼동행렬 |
| predictions.npz | 검증·평가 정답, 모델별 점수와 제외 마스크 |
| run.json | 환경, RF 설정, 추출 인덱스, 표본 수, 입력과 코드 해시 |

`--output <새 경로>`로 저장 위치를 지정할 수 있습니다. 기존 디렉터리는 덮어쓰지 않습니다.

```bash
python Research/run.py compare --trees 500 --seed 7 --jobs 4 --output Research/runs/my-experiment
```

## 3. 원본 CSV에서 윈도우 재구성

```bash
python Research/run.py prepare
```

`prepare`는 train CSV를 시간 순서로 정렬하고 정상 train만으로 StandardScaler를 적합합니다. 메타데이터의 원본 파일·행 구간·시각과 라벨을 확인해 길이 30 윈도우를 재구성하고, 제공된 NPZ와 정확히 같은지 검사합니다. 원본 CSV와 기존 배열은 변경하지 않습니다. 결과는 실행 폴더의 `augmentation_split/`과 `scaler.json`에 저장됩니다. 이 명령은 기존 분할 소속을 재구성하는 명령이며, 새로운 분할을 자동 설계하지 않습니다.

재구성한 자료를 사용하는 예:

```bash
python Research/run.py baseline --split-dir Research/runs/<prepare-실행ID>/augmentation_split
```

## 4. 전체 조건 분석

```bash
python Research/run.py full
```

기존 통일 조건 분석 코드를 실행하되 출력은 새 실행 폴더로 지정합니다. 주 비교, 클래스 구성, 필터링, 마스킹 비율과 30회 반복 안정성을 포함하므로 첫 실행에는 `baseline` 또는 `compare`를 권합니다. 이 명령도 저장된 생성 후보를 사용합니다.

## 5. 생성기 재학습

```bash
python -m pip install -r Research/requirements-training.txt
python Research/run.py generate
python Research/run.py masking
python Research/run.py compare
```

GPU 환경은 별도로 적합한 PyTorch 배포판을 준비할 수 있습니다. `generate`는 기존 Research02의 GT-GAN·Diffusion·마스킹 생성기를 학습하고 후보 1,000개씩을 저장합니다. `masking`은 Research10의 마스킹 비율별 후보를 준비하며 기존 캐시가 있으면 재사용합니다. 특정 비율을 새로 학습하려면 해당 캐시를 다른 이름으로 보관한 후 실행하세요.

이 두 명령은 기존 연구 스크립트이므로 `Research/data/research02` 또는 `research10`의 후보·집계 결과를 갱신합니다. 생성 후보 변경 후 `check`의 기존 입력 해시는 맞지 않게 됩니다. `compare`는 저장된 품질 JSON의 고정 추출 인덱스를 사용하며 FMD 후보는 현재 입력에서 다시 선별합니다. 품질 통계 자체도 갱신하려면 `Research`에서 `python tools/research02_quality_comparison_table.py`를 실행하세요.

전체 생성기 재학습·30회 반복은 이번 저장소 정리에서 다시 수행하지 않았습니다. 실제 확인 범위는 [검증 기록](validation.md)에 구분했습니다.

## 6. 새 데이터와 새 방법

- 새 데이터의 `normal_train`, `anomaly_train_seed`, `validation`, `final_test`를 NPZ와 CSV로 준비하고 `--split-dir`로 지정할 수 있습니다. 배열은 `X` (N,30,5), `y` (N,), `features`를 포함해야 하며 현재 코드의 변수 순서를 따라야 합니다.
- 메타데이터에는 `source_file`, `start_index`, `end_index`, `label`이 필요합니다. 원시 행 범위의 길이는 30이어야 하며 검증·평가는 두 클래스를 포함해야 합니다.
- 새 자료의 첫 실험은 `baseline`으로 실행하세요. 기존 데이터에서 만든 생성 후보를 새 데이터와 혼합해 얻은 `compare` 결과를 새 생성기의 성능으로 해석하면 안 됩니다. 생성 후보와 필터 참조도 자기 데이터로 다시 준비해야 합니다.
- 다른 윈도우 길이·변수 구성을 사용하려면 입력 검증과 해당 학습·생성 코드를 함께 수정해야 합니다.
- 새 방법은 `tools/reproduce_comparison.py`의 `augmented` 구성에 추가하고 같은 검증·평가 경로를 사용하면 됩니다.

연구 이력 노트북은 당시 탐색 과정입니다. 현재 재실행의 기준은 `run.py`와 tools 코드이며, 노트북에 최신 분할 보정이 모두 반영된 것은 아닙니다.
