# 문제 정의 초안 (3장 구조 개편 전 보관본)

> 3.1절을 사전 지식으로 개편하면서 기존 문제 정의를 보존한 문서이다. 시계열 윈도우, 클래스 불균형 및 증강 학습 데이터셋에 관한 핵심 정의는 개편된 3.2절의 연구 흐름과 4.1절의 데이터 구성 설명에 반영한다.

본 연구에서는 불량 학습 데이터가 제한적이고 정상과 불량의 표본 수에 불균형이 존재하는 조건에서, 용접 공정의 다변량 시계열을 이용하여 불량 여부를 판단하는 문제를 다룬다. 정상 및 불량 레이블이 부여된 학습 자료를 활용하는 지도학습 기반 이상 탐지를 대상으로 하며, 시계열 데이터 증강을 통해 불량 학습 자료를 확장하고 새로운 실제 데이터에 대한 탐지 성능을 향상시키는 것을 목적으로 한다.

시간순으로 구성된 다변량 시계열의 $i$번째 윈도우를 $\mathbf{X}_i\in\mathbb{R}^{T\times d}$로 나타낸다. 여기서 $T$는 윈도우에 포함되는 관측 시점의 수이고, $d$는 모델 입력으로 선택된 공정 변수의 수이다. 각 윈도우에는 정상 상태를 의미하는 $y_i=0$ 또는 불량 상태를 의미하는 $y_i=1$의 레이블을 부여한다. 윈도우 내 모든 시점이 정상으로 표시된 경우 정상 윈도우로 정의하고, 불량으로 표시된 시점이 하나 이상 포함된 경우 불량 윈도우로 정의한다. 따라서 본 연구의 예측 단위는 시계열 윈도우이며, 출력은 해당 윈도우의 불량 여부를 나타내는 이진 레이블이다. 입력 변수의 선택, 윈도우 구성 및 레이블 부여의 구체적인 절차는 4.1절에서 설명한다.

학습 데이터셋은 정상 윈도우 집합 $\mathcal{D}_{0}$과 불량 윈도우 집합 $\mathcal{D}_{1}$로 구성되며, 다음과 같이 나타낼 수 있다.

$$
\mathcal{D}_{\mathrm{tr}}
=\mathcal{D}_{0}\cup\mathcal{D}_{1},
\qquad
\mathcal{D}_{c}
=\{(\mathbf{X}_{i}^{(c)},c)\}_{i=1}^{n_c},
\quad c\in\{0,1\}.
\tag{1}
$$

정상 및 불량 학습 윈도우의 수를 각각 $n_0=|\mathcal{D}_0|$, $n_1=|\mathcal{D}_1|$로 정의하면, 클래스 불균형 비율은 다음과 같다.

$$
\rho=\frac{n_0}{n_1},\qquad n_0\gg n_1>0.
\tag{2}
$$

본 연구에서 불량 데이터 부족은 관측 가능한 불량 학습 윈도우의 수가 제한되어 불량 패턴의 변동성을 충분히 학습하기 어려운 조건을 의미한다. 클래스 불균형은 정상과 불량의 상대적인 표본 수 차이가 큰 조건을 의미하며, 식 (2)의 $\rho$로 나타낸다. 두 조건은 각각 불량 클래스 내부의 정보 부족과 클래스 간 학습 자료의 편중에 해당한다. 본 연구는 이 두 조건이 함께 존재하는 학습 환경을 대상으로 하며, 실제 표본 수와 데이터 분할별 구성은 4.1절에서 제시한다.

이러한 조건에서 데이터 증강의 목적은 관측된 불량 시계열을 바탕으로 분류 학습에 유효한 추가 표본을 구성하는 것이다. 증강 표본에는 불량 레이블을 부여하므로, 실제 불량의 시간적 특성과 변수 간 관계를 고려하여 신호 특성과 레이블 사이의 일관성을 유지하도록 요구한다. 또한 관측된 불량 표본의 변동성을 확장하면서 학습에 활용할 수 있는 품질의 표본을 확보하는 것을 증강의 요구 조건으로 설정한다.

학습에 사용할 불량 증강 윈도우를 $\widetilde{\mathbf{X}}_j\in\mathbb{R}^{T\times d}$, 그 수를 $m$으로 정의하면, 증강 데이터셋과 이를 결합한 학습 데이터셋은 다음과 같이 표현된다.

$$
\widetilde{\mathcal{D}}_{1}
=\{(\widetilde{\mathbf{X}}_j,1)\}_{j=1}^{m},
\qquad
\mathcal{D}_{\mathrm{aug}}
=\mathcal{D}_{0}\cup\mathcal{D}_{1}\cup\widetilde{\mathcal{D}}_{1}.
\tag{3}
$$

여기서 $m$은 최종적으로 분류 학습에 투입되는 증강 윈도우의 수를 의미하며, 품질 필터링을 적용하는 경우 선별된 표본 수에 해당한다. 증강된 학습 데이터셋은 실제 정상·불량 데이터와 불량 증강 데이터로 구성되며, 이를 이용하여 새로운 실제 윈도우의 불량 여부를 판단하는 분류기를 학습한다.

증강 데이터셋 $\mathcal{D}_{\mathrm{aug}}$으로 학습한 분류기의 불량 점수를 $s_{\theta}(\mathbf{X})\in[0,1]$로 나타내면, 새로운 윈도우의 예측 레이블은 판정 임계값 $\tau$를 이용하여 다음과 같이 결정한다.

$$
\widehat{y}(\mathbf{X})
=\mathbb{I}\!\left[s_{\theta}(\mathbf{X})\geq\tau\right],
\qquad \tau\in[0,1],
\tag{4}
$$

여기서 $\theta$는 분류기의 학습된 매개변수를 나타내며, $\mathbb{I}[\cdot]$는 조건이 참이면 1, 그렇지 않으면 0을 반환하는 지시함수이다. 판정 임계값은 실제 검증 데이터에서 선택하고, 탐지 성능은 별도로 구성한 실제 최종 평가 데이터에서 측정한다. 최종적으로 본 연구는 제한된 불량 데이터와 클래스 불균형 조건에서 증강을 적용한 분류기가 원본 데이터만으로 학습한 기준선보다 불량 탐지 성능을 향상시킬 수 있는지 검증한다. 이때 불량의 미검출 감소와 정상 데이터의 오탐 억제를 함께 고려하며, 생성 표본의 품질과 실제 데이터에 대한 분류 성능을 연결하여 증강의 유효성을 평가한다.

---

## 4.1 작성에 반영할 실험 설정 — 3.1 본문에 포함하지 않음

| 항목 | 확인된 설정 및 작성 내용 |
| --- | --- |
| 시간 기준 | `WorkingTime`을 시간순 정렬 기준으로 사용하며 모델 입력 변수에서 제외 |
| 최종 입력 변수 | `RealPower`, `GateOnTime`, `SetPower`, `Speed`, `Length`의 5개 변수 |
| 제외 변수 | `SetFrequency`, `SetDuty`는 현재 연구에서 입력 변수로 사용하지 않음. 원본 변수 목록과 최종 입력 목록의 연결을 설명 |
| 윈도우 설정 | 관측 시점 수 $T=30$, 이동 간격 1, 입력 차원 $d=5$ |
| 윈도우 레이블 | 구간 내 시점 레이블의 최댓값. 모두 정상인 경우 0, 불량 시점이 하나 이상인 경우 1 |
| 학습 후보 | 정상 윈도우 135,730개, 불량 윈도우 247개 |
| 주요 비교 실험의 실제 학습 자료 | 정상 5,000개, 불량 247개. 증강 전 불균형 비율 약 20.24:1 |
| 검증 데이터 | 정상 1,283개, 불량 164개, 총 1,447개 |
| 최종 평가 데이터 | 정상 2,785개, 불량 296개, 총 3,081개 |

그림에는 입력 변수 선택과 윈도우 설정을 표시하고, 윈도우 생성 이후의 학습·검증·최종 평가 분할은 본문 또는 표로 설명한다. 원본 제공 파일의 train/test 구분과 본 연구의 학습·검증·최종 평가 구분을 연결하여 서술한다.

## 2장 작성에 활용할 선행연구 메모 — 3.1 본문에 포함하지 않음

다음 문헌의 연구별 소개와 비교는 2장 관련 연구에서 활용한다. 3.1은 본 연구의 입력, 레이블, 데이터 제약 및 목적을 정의하는 내용으로 구성하였다. 정의·수식·기법을 특정 문헌에서 가져오는 경우에는 해당 위치에 필요한 출처를 인용할 수 있다.

- **Kim et al. (2023), Pu et al. (2026):** 제조 시계열 및 산업 결함 영상에서 불량 데이터 부족을 완화하기 위한 생성형 증강과 후속 탐지 성능 평가. 2.1–2.2절에서 연구 대상과 접근 방식을 구분해 소개한다.
- **Wen et al. (2021):** 시간적 의존성, 변수 간 관계 및 레이블 보존을 고려하는 시계열 증강의 요구 조건. 2.2절에서 활용한다.
- **Ekwaro-Osire et al. (2025):** 제조 수치 데이터에서 증강 효과가 데이터 및 적용 조건에 따라 달라질 수 있다는 실험적 근거. 2.2절에서 증강 기법의 적용 범위와 한계를 설명할 때 활용한다.

### 참고문헌

1. Kim, Y., et al. (2023). Self-supervised representation learning anomaly detection methodology based on boosting algorithms enhanced by data augmentation using StyleGAN for manufacturing imbalanced data. *Computers in Industry, 153*, 104024. https://doi.org/10.1016/j.compind.2023.104024
2. Pu, C., et al. (2026). Towards data-constrained defect inspection in advanced manufacturing: A lightweight diffusion model adapter for high-fidelity data generation. *Expert Systems with Applications, 325*, 132452. https://doi.org/10.1016/j.eswa.2026.132452
3. Wen, Q., Sun, L., Yang, F., Song, X., Gao, J., Wang, X., & Xu, H. (2021). Time Series Data Augmentation for Deep Learning: A Survey. *Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence*, 4653–4660. https://doi.org/10.24963/ijcai.2021/631
4. Ekwaro-Osire, H., Ponugupati, S. L., Al Noman, A., Bode, D., & Thoben, K.-D. (2025). Data augmentation for numerical data from manufacturing processes: an overview of techniques and assessment of when which techniques work. *Industrial Artificial Intelligence, 3*, Article 1. https://doi.org/10.1007/s44244-024-00021-x

## 작성 근거 메모 — 투고 본문에 포함하지 않음

- 데이터 분할과 표본 수: [split_summary.csv](data/augmentation_split/split_summary.csv).
- 윈도우 크기와 변수 수: `data/augmentation_split/anomaly_train_seed_windows.npz`의 `X.shape=(247, 30, 5)`. 변수는 `RealPower`, `GateOnTime`, `SetPower`, `Speed`, `Length`이다.
- 윈도우 레이블 규칙: [Research.ipynb](data/raw_data/Research.ipynb)의 `make_windows` 함수는 구간 내 시점 레이블의 최댓값을 윈도우 레이블로 사용한다.
- 주요 학습 조건 및 검증 데이터에서의 임계값 선택: [research01_randomforest_baseline.py](tools/research01_randomforest_baseline.py), [연구 개요](README.md).
- 정상 학습 후보 전체에 대한 비율은 약 549.51:1이며, 4.1 작성 메모의 20.24:1은 정상 5,000개를 사용하는 주요 비교 실험의 조건이다. 3.1 본문에는 기호로 정의한 불균형 조건을 제시하고, 구체적인 실험값은 4.1절에 배치한다.
- 불량 윈도우 수는 독립적인 불량 사건 수를 뜻하지 않는다. [윈도우 메타데이터](data/augmentation_split/anomaly_train_seed_metadata.csv)에 겹치는 구간이 존재하므로, 독립 사건 수나 공정 전체의 불량 발생률로 해석하지 않는다.
- 제시한 수식은 현재 연구의 입력·레이블·증강 학습 자료·판정 규칙을 정리한 표기이다. 별도의 공동 최적화 손실이나 확인되지 않은 실험 결과를 도입하지 않았다. 수식 번호는 전체 원고의 순서에 맞춰 조정할 수 있다.
- 검토한 선행연구 네 편은 모두 현재 `참고논문/` 폴더에 있는 논문이다. 연구별 소개는 2장 작성용 메모로 분리하였다. 영상 결함 검출 결과를 용접 시계열의 실증 결과로 사용하지 않았다.
