# 실행 가능한 연구 폴더

공통 진입점은 `run.py`입니다. 저장소 루트에서 실행할 수 있습니다.

```bash
python -m pip install -r Research/requirements.txt
python Research/run.py check
python Research/run.py baseline
python Research/run.py compare
```

- `data/`: 원본 CSV·분할 배열·메타데이터·생성 후보와 기존 단계별 결과
- `notebooks/`: 연구 과정 노트북 (실행 출력 제외)
- `tools/`: 전처리·생성·필터링·평가·시각화
- `pictures/`: 연구 설명용 그림
- `output/`: 고정 비교에 필요한 입력
- `runs/`: 새 실험별 출력 (Git 제외)
- `fmd_api/`: 부가 구현인 API와 대시보드

[연구 포트폴리오](../README.md) · [실행·확장 안내](../docs/reproduction.md) · [검증 기록](../docs/validation.md)
