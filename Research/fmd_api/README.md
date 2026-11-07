# 이상 탐지 API와 대시보드

Baseline, Magnitude warping, FMD 증강으로 학습한 Random Forest 모델의 추론 결과를 비교합니다. FMD 생성기는 오프라인 증강에 사용합니다.

[환경 준비와 필수 데이터·모델](../../docs/reproduction.md)을 먼저 확인하세요. 공개 저장소에는 원본 데이터와 학습 모델을 포함하지 않으므로 clone만으로 API를 실행할 수 없습니다.

필요한 연구 입력을 복원한 뒤 `Research` 폴더에서 실행합니다.

```powershell
python -m venv .venv-api
.\.venv-api\Scripts\python.exe -m pip install -r requirements-api.txt
.\.venv-api\Scripts\python.exe tools/research11_export_api_models.py
.\.venv-api\Scripts\python.exe -m uvicorn fmd_api.app:app --host 127.0.0.1 --port 8000
```

- 대시보드: http://127.0.0.1:8000/
- API 문서: http://127.0.0.1:8000/docs
- 입력: 정규화 전 센서 5개 × 30시점과 feature_names
- 출력: 모델별 이상 점수, 임계값, 판정, 추론 시간

대시보드의 샘플과 전체 검증 결과 기능에는 원본 CSV·분할 메타데이터·Research11 결과 파일도 필요합니다. 로컬 연구 환경이 준비되어 있으면 Start_API.cmd와 Stop_API.cmd로 서버를 관리할 수 있습니다.

Research11은 이전 평가 3,081개 실험을 기반으로 합니다. 포트폴리오 대표 표는 최신 통일 조건 평가 3,080개 결과이므로 별개의 실험으로 해석하세요.
