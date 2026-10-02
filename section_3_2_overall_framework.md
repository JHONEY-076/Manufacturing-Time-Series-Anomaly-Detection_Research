# 3.2. 제안 방법의 전체 프레임워크 (Overall framework of the proposed method)

그림 X는 제한된 불량 데이터 조건에서 용접 공정 시계열을 증강하기 위한 Filtered Masking Diffusion(FMD)의 전체 프레임워크를 나타낸다. FMD는 (1) 실제 정상·불량 학습 데이터에 기반한 적응형 구조적 마스킹, (2) 마스킹된 실제 불량 윈도우를 조건으로 사용하는 시계열 확산 생성, (3) 실제 불량과의 유사성 및 정상 데이터와의 분리성을 이용한 생성 후보 필터링의 세 단계로 구성된다. 선별된 생성 윈도우는 실제 학습 데이터와 결합되어 후속 불량 탐지 분류기의 학습에 사용된다.

시간순으로 구성된 $i$번째 다변량 시계열 윈도우를 $\mathbf{X}_i\in\mathbb{R}^{T\times d}$, 윈도우 레이블을 $y_i\in\{0,1\}$로 나타낸다. $y_i=0$은 정상 윈도우, $y_i=1$은 비정상 시점이 하나 이상 포함된 공정 이상 윈도우를 의미한다. 본 장에서는 코드 및 실험 명칭과의 일관성을 위해 이를 불량 윈도우라고 부르지만, 그 수는 독립적인 불량 제품의 개수가 아니라 비정상 라벨을 포함한 시계열 구간의 수를 뜻한다. 실제 정상 학습 윈도우 집합과 실제 불량 학습 윈도우 집합을 각각 $\mathcal{D}_0$와 $\mathcal{D}_1$로 정의한다. 제한된 $\mathcal{D}_1$은 FMD의 시드 윈도우로 사용되며, $\mathcal{D}_0$는 정상과 불량의 분포 차이를 계산하고 생성 후보의 정상 데이터 근접을 억제하는 기준으로 사용된다. 입력 변수, 윈도우 구성 및 데이터 분할은 4.1절에서 설명한다.

첫 번째 단계에서는 정상과 불량 윈도우의 위치별·변수별 분포 차이를 계산하여 마스킹 확률을 구성한다. 정상과 불량의 차이가 큰 연속 시간 구간은 더 높은 확률로 선택하며, 변수별 차이와 실제 불량 데이터의 변수 간 상관관계를 함께 고려해 변수 그룹을 구성한다. 선택된 시간 구간과 변수 그룹은 0, 나머지 관측 영역은 1인 이진 마스크를 생성한다. 이러한 적응형 구조적 마스킹은 불량 판별에 관련된 영역의 변동성을 집중적으로 확장하면서 실제 시드 윈도우의 나머지 구조를 유지한다.

두 번째 단계에서는 마스킹된 불량 윈도우와 이진 마스크를 조건으로 받는 확산 모델을 학습한다. 잡음 예측기는 GRU를 이용하여 윈도우 내부의 시간적 의존성을 처리하고, 확산 단계 임베딩과 조건 정보를 이용해 각 시점의 잡음을 추정한다. 생성 단계에서는 실제 불량 시드에 중간 수준의 잡음을 추가한 뒤 역확산을 수행한다. 역확산 결과 가운데 마스킹된 영역만 새로운 값으로 사용하고, 관측 영역은 원래 시드 값으로 복원한다. 이에 따라 생성 윈도우는 실제 불량의 관측 문맥을 유지하면서 선택된 영역에서 새로운 변동을 갖는다. 적응형 마스크와 조건부 확산 생성기의 세부 구조는 3.3절에서 설명한다.

세 번째 단계에서는 Masking Diffusion이 생성한 모든 후보를 학습에 사용하지 않고 품질 기준에 따라 선별한다. 각 생성 후보를 실제 불량 중심과의 거리, 정상 중심과의 거리, 실제 불량의 변수별 범위 안에 포함되는 비율로 평가한다. 실제 불량 분포에 가깝고 정상 분포와 충분히 분리되며 실제 불량의 값 범위를 유지하는 후보를 우선한다. 세 기준을 결합한 품질 점수로 후보의 순위를 정하고, 목표 증강 수만큼 선택한다. 품질 기준, 점수 및 선별 절차는 3.4절에서 제시한다.

선별된 불량 증강 윈도우 집합을 $\widetilde{\mathcal{D}}_1^{\mathrm{sel}}$이라 하면 후속 분류기의 학습 데이터는 다음과 같이 구성된다.

$$
\mathcal{D}_{\mathrm{aug}}
=\mathcal{D}_0\cup\mathcal{D}_1\cup
\widetilde{\mathcal{D}}_1^{\mathrm{sel}}.
\tag{6}
$$

본 연구에서는 모든 비교 조건에서 동일한 RandomForest 분류기와 데이터 분할을 사용하여 증강 방법에 따른 차이를 평가한다. 판정 임계값은 검증 데이터에서 선택하고 최종 성능은 학습과 임계값 선택에 사용하지 않은 최종 평가 데이터에서 측정한다. FMD의 마스크 구성, 생성 모델 학습 및 품질 필터에는 학습 데이터만 사용하며, 검증 데이터와 최종 평가 데이터는 생성 또는 선별 기준 계산에 사용하지 않는다.

## 그림 캡션

**국문**

그림 X. 제한된 불량 데이터 조건에서 용접 시계열을 증강하는 Filtered Masking Diffusion의 전체 프레임워크. 정상·불량 학습 데이터의 분포 차이를 기반으로 적응형 구조적 마스크를 생성하고, 마스킹된 실제 불량 윈도우를 조건으로 사용하는 시계열 확산 모델이 생성 후보를 복원한다. 생성 후보는 실제 불량과의 근접성, 정상 데이터와의 분리성 및 불량 범위 일관성에 따라 선별되며, 선택된 윈도우는 실제 학습 데이터와 결합되어 불량 탐지 분류기의 학습에 사용된다.

**영문**

Fig. X. Overall framework of Filtered Masking Diffusion for augmenting welding time-series data under limited anomalous training windows. An adaptive structured mask is constructed from distributional differences between normal and anomalous training windows. A temporal diffusion model conditioned on a masked real anomaly window restores the selected regions. Generated candidates are filtered according to proximity to real anomalies, separation from normal data, and consistency with the anomaly range, and the selected windows are combined with real data to train the downstream defect classifier.

---

## 그림 및 작성 메모 - 투고 본문에 포함하지 않음

- 그림의 중심 경로는 `실제 학습 데이터 -> 적응형 마스킹 -> 조건부 확산 복원 -> 품질 필터링 -> 증강 학습 데이터 -> 분류기`로 구성한다.
- 검증 데이터는 임계값 선택으로만, 최종 평가 데이터는 최종 성능 측정으로만 연결하여 데이터 누수를 방지한 구조를 보여준다.
- GT-GAN, 일반 Diffusion 및 전통적 시계열 변환은 4장의 비교 실험에 배치한다. 3.2절의 중심 그림에 모든 비교 방법을 병렬로 넣으면 제안 방법의 구조가 흐려질 수 있다.
- 문제 정의 초안의 구체적인 표본 수와 불균형 비율은 4.1절 데이터 구성 표로 이동한다.
