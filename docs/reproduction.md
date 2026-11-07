# 실행 환경과 재현 범위

## 환경 준비

저장소 루트에서 다음과 같이 준비합니다. requirements-api.txt는 기존 Python 3.13 API 검증 환경의 버전이며 전체 생성 모델 학습 환경을 고정한 파일은 아닙니다. 생성 모델별 추가 의존성은 해당 스크립트의 import와 실행 환경을 확인해야 합니다.

```powershell
cd Research
python -m venv .venv-api
.\.venv-api\Scripts\python.exe -m pip install -r requirements-api.txt
```

## 연구 결과 재현

`Research/tools/research_section43_unified.py`는 기존 생성 후보 풀과 분할 자료를 이용합니다. 공개 저장소에는 다음 입력이 포함되어 있지 않습니다. 이용 권한을 확보한 원본 연구 자료를 동일한 상대 경로에 배치해야 합니다.

- `Research/data/raw_data/`: 원본 용접 센서 CSV
- `Research/data/augmentation_split/`: 학습·검증·평가 배열과 메타데이터
- `Research/data/research02/generated/`: 생성 방법별 후보 배열
- `Research/data/research10/generated/`: 마스킹 비율별 생성 배열
- `Research/output/section_4_4_1/generation_quality_comparison.json`: 생성 품질 집계

```powershell
# Research 폴더에서, 필요한 입력을 준비한 후 실행
.\.venv-api\Scripts\python.exe tools/research_section43_unified.py
```

공개 결과의 주 비교는 길이 30 윈도우, 정상 학습 5,000개, 불량 seed 247개, 증강 750개, RF 300 trees, min_samples_leaf=2, balanced_subsample 조건입니다. 검증 집합에서 임계값을 선택하고 별도 평가 집합에서 측정했습니다. 최신 집계는 평가 3,080개를 사용합니다. 기존 Research11 API는 이전 3,081개 평가 실험을 기반으로 하므로 서로 다른 실험 수치를 직접 혼합하지 마세요.

## API와 대시보드

API는 저장된 Random Forest 모델을 추론합니다. Diffusion 생성기를 실시간으로 실행하는 API가 아닙니다.

1. 권한이 있는 연구 데이터를 복원합니다.
2. `tools/research11_export_api_models.py`를 실행해 `artifacts/research11/`에 모델과 manifest를 생성합니다.
3. 아래 명령으로 서버를 실행합니다.

```powershell
.\.venv-api\Scripts\python.exe tools/research11_export_api_models.py
.\.venv-api\Scripts\python.exe -m uvicorn fmd_api.app:app --host 127.0.0.1 --port 8000
```

브라우저에서 http://127.0.0.1:8000/ 을 엽니다. 모델 파일이 없으면 서버 시작에 실패합니다. 대시보드의 샘플·검증 결과 기능에는 원본 CSV, 분할 메타데이터와 Research11 검증 산출물도 필요합니다.

## 한계

공개 CSV와 이미지는 기존 실험 산출물이며, 포트폴리오 정리 과정에서 모델 재학습이나 성능 재검증을 수행한 결과가 아닙니다. 원본 데이터와 모델을 배포하지 않으므로 공개 파일만으로 수치의 완전한 재현은 지원하지 않습니다.
