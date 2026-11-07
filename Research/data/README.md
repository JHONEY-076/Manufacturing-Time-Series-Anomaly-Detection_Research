# 재현 입력과 연구 단계별 결과

`raw_data`의 센서 CSV, `preprocessed`의 기존 라벨, `augmentation_split`의 배열·원시 행 메타데이터, Research02·10의 저장된 생성 후보를 포함합니다. `input_manifest.json`은 실행 입력 27개의 SHA-256입니다.

기존 분할 소속은 그대로 보존했으며, 새 비교 실행은 원시 행이 겹치는 검증·평가 윈도우를 제외합니다. `run.py prepare`로 원본 CSV에서 배열을 다시 구성하고, `run.py compare`로 저장된 생성 후보의 후속 탐지 성능을 평가할 수 있습니다.

기존 단계별 집계는 연구 이력입니다. 새 실행 결과는 `../runs/`에 저장하며 과거 지표와 섞어 쓰지 마세요. 외부 출처·재배포 라이선스의 별도 근거는 현재 자료에 명시되지 않았으므로, 재사용 시 원출처 이용 조건을 확인해야 합니다.

[실행·확장 안내](../../docs/reproduction.md) · [기존 연구 이력](../../docs/research-history.md)
