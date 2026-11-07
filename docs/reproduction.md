# 연구 코드와 재현 범위

이 공개 저장소는 README의 연구 설명·결과·그림과 관련 코드를 중심으로 구성합니다. 원본 데이터, 생성 표본, 학습 모델, 노트북 출력은 포함하지 않으므로 clone만으로 전체 실험을 재현할 수 없습니다.

## 환경 준비

아래 패키지는 기존 README의 설치 예시이며 전체 학습 환경을 고정한 목록은 아닙니다. PyTorch는 운영체제와 가속기 환경에 맞춰 준비해야 합니다.

```powershell
cd Research
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas matplotlib seaborn scikit-learn torch
```

## 필요한 연구 입력

이용 권한이 있는 연구 자료를 기존 상대 경로에 배치해야 합니다.

- `Research/data/raw_data/`: 원본 센서 CSV
- `Research/data/augmentation_split/`: 학습·검증·평가 배열과 메타데이터
- `Research/data/research02/generated/`: 생성 방법별 후보 배열
- `Research/data/research10/generated/`: 마스킹 비율별 생성 배열
- `Research/output/section_4_4_1/generation_quality_comparison.json`: 최신 통일 조건 비교의 생성 품질 입력

입력 준비 후 `Research` 폴더에서 실행합니다.

```powershell
python tools/research01_randomforest_baseline.py
python tools/research02_generate_data.py
python tools/research03_model_augmented_validation.py
python tools/research05_balanced_aug_experiment.py
python tools/research08_augmentation_saturation.py
python tools/research09_filtering_strength.py
python tools/research10_masking_ratio.py
```

분할 수정 후 비교는 `python tools/research_section43_unified.py`로 별도 실행합니다. 기존 README의 평가 3,081개 결과와 최신 통일 조건의 3,080개 결과를 구분하세요. [최신 결과 안내](latest-results.md)에 조건을 설명했습니다.

공개한 CSV·JSON·그림은 저장된 실험 산출물입니다. 이번 문서 정리에서 모델 재학습이나 성능 재검증은 수행하지 않았습니다.
