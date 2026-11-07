# 연구 과정과 코드 대응표

README의 연구 설명을 읽고 아래 구현과 공개 결과를 함께 확인할 수 있습니다. 기존 실험과 분할 수정 후 비교는 조건이 다르므로 구분해서 검토하세요.

| 연구 단계 | 확인할 구현 | 공개 결과 또는 설명 |
|---|---|---|
| 기준 모델과 검증 임계값 선택 | [Research01](../Research/tools/research01_randomforest_baseline.py), `select_threshold_by_validation_f1` | [기준 모델 집계](results/research/research01_randomforest_baseline_summary.json) |
| 적응형 마스킹과 생성 모델 | [Research02](../Research/tools/research02_generate_data.py), `build_adaptive_mask_config`, `make_structured_mask`, `train_temporal_masked_diffusion_model` | [생성 품질 집계](results/research/research02_summary.json) |
| 증강 후 분류 성능과 임계값 전략 | [Research03](../Research/tools/research03_model_augmented_validation.py), F1·F2·목표 재현율별 선택 함수 | [지도학습 집계](results/research/research03_model_improvement_summary.json) |
| 클래스 구성과 전통 증강 비교 | [Research05](../Research/tools/research05_balanced_aug_experiment.py) | [비교 집계](results/research/research05_summary.json) |
| 증강량과 포화 분석 | [Research08](../Research/tools/research08_augmentation_saturation.py) | [증강량 집계](results/research/research08_summary.json) |
| 생성 후보의 품질 필터링 | [Research09](../Research/tools/research09_filtering_strength.py), `quality_filter_generated_anomalies_with_strength` | [필터링 강도 집계](results/research/research09_filtering_strength_summary.json) |
| 마스킹 비율 민감도 | [Research10](../Research/tools/research10_masking_ratio.py) | [마스킹 비율 집계](results/research/research10_masking_ratio_summary.json) |
| 원시 행 중첩 제외 후 통일 조건 비교 | [비교 실행](../Research/tools/research_section43_unified.py), [분할 처리 도우미](../Research/tools/research_section441_matched_rf.py) | [최신 조건과 결과](latest-results.md) |
| 연구 결과의 응용 | [FastAPI](../Research/fmd_api/app.py), [화면과 데이터 연결](../Research/fmd_api/dashboard.py) | [부가 구현 사례](dashboard.md) |

## 독자가 확인할 수 있는 범위

- 코드 검토: 학습에 사용하는 자료, 마스킹·선별 규칙, 임계값 선택과 지표 계산을 확인할 수 있습니다.
- 집계 검산: 공개 CSV의 TP·FP·TN·FN으로 Precision·Recall·F1·F2를 다시 계산할 수 있습니다. [검산 코드](../Research/tools/verify_public_metrics.py)를 데이터 없이 실행할 수 있습니다.
- 실험 재실행: 원본 센서 CSV·분할 메타데이터·전처리 배열·저장된 생성 후보를 포함했습니다. `python Research/run.py compare`로 후속 탐지 실험을 실행하고 개별 예측 점수도 확인할 수 있습니다. [실행과 확장](reproduction.md)을 확인하세요.

지표 검산은 예측이 올바르게 생성되었는지, 학습 과정이 적절했는지까지 입증하지 않습니다. AP·AUROC 검증에는 정답과 개별 예측 점수가 필요합니다. 기존 실험에서 발견한 분할 문제와 최신 평가의 제한점도 함께 검토해야 합니다.
