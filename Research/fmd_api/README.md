# 이상 탐지 API와 대시보드

Baseline, Magnitude warping, FMD 증강으로 학습한 Random Forest 모델의 추론 결과를 비교합니다. FMD 생성기는 오프라인 증강에 사용합니다.

[연구 실행 안내](../../docs/reproduction.md)를 먼저 확인하세요. 연구 데이터는 포함하며 학습된 분류기는 아래 내보내기 명령으로 생성합니다. API 환경은 기본 비교 환경과 별도로 설치합니다.

`Research` 폴더에서 실행합니다.

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

대시보드의 샘플과 검증 결과 기능에는 함께 제공한 원본 CSV·분할 메타데이터·Research11 집계 파일을 사용합니다. 환경과 모델이 준비되면 Start_API.cmd와 Stop_API.cmd로 서버를 관리할 수 있습니다.

Research11은 이전 평가 3,081개 실험을 기반으로 합니다. 루트 README의 기존 연구 설명과 분할 수정 후 3,080개 비교 결과를 구분해서 확인하세요. 이 API는 연구의 부가 구현 사례입니다.
