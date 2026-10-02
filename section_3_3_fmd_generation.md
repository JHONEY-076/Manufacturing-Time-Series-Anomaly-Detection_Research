# 3.3. 마스킹 기반 조건부 확산 생성 (Masking-based conditional diffusion generation)

FMD의 생성 단계는 실제 불량 시드의 일부를 구조적으로 마스킹하고, 관측된 문맥을 조건으로 마스킹된 영역을 확산 모델이 복원하도록 구성된다. 전체 윈도우를 순수 잡음에서 생성하는 방식과 달리, 관측 영역을 그대로 보존하여 제한된 불량 표본의 시간적 형태와 변수 간 관계가 생성 결과에 유지되도록 한다.

## 3.3.1. 데이터 적응형 구조적 마스크

정상 윈도우와 실제 불량 윈도우의 위치별 차이를 이용해 마스킹 우선순위를 결정한다. 정상 학습 집합에서 추출한 윈도우의 위치별 평균과 실제 불량 윈도우의 위치별 평균을 각각 $\boldsymbol{\mu}_{0,t}$와 $\boldsymbol{\mu}_{1,t}$로 나타내고, 두 집합을 결합하여 계산한 변수 $k$의 표준편차를 $\sigma_k$라고 한다. 시간 위치 $t$와 변수 $k$의 표준화 차이는 다음과 같다.

$$
A_{t,k}
=\frac{\left|\mu_{1,t,k}-\mu_{0,t,k}\right|}
{\sigma_k+\varepsilon}.
\tag{7}
$$

시간 위치별 점수는 $A_{t,k}$를 변수 방향으로 평균하여 계산한다. 점수의 국소적인 변동을 완화한 후 길이 $L$의 연속 구간별 평균을 구하고, 이를 정규화하여 시간 블록 시작 위치의 선택 확률로 사용한다. 따라서 정상과 불량의 차이가 지속적으로 큰 시간 구간이 마스킹될 가능성이 높다.

변수별 점수는 $A_{t,k}$를 시간 방향으로 평균하여 계산한다. 변수 점수가 큰 변수를 기준 변수로 선택하고, 실제 불량 윈도우에서 해당 변수와 절댓값 상관계수가 큰 변수들을 결합하여 크기 $G$의 변수 그룹을 구성한다. 각 변수 그룹의 평균 점수를 정규화하여 그룹 선택 확률로 사용한다. 이 과정은 불량과 관련된 변수뿐 아니라 함께 변화하는 변수 관계를 복원 대상으로 포함한다.

선택된 시간 구간을 $\mathcal{I}_i$, 변수 그룹을 $\mathcal{G}_i$라 하면 식 (2)에 따라 이진 마스크 $\mathbf{M}_i$를 구성한다. 현재 구현에서 입력 윈도우는 $T=30$, 변수 수는 $d=5$이며 기본 설정은 시간 블록 비율 0.25와 변수 그룹 비율 0.40이다. 이에 따라 연속된 8개 시점과 2개 변수로 구성된 영역을 마스킹한다. 실제 적용 설정과 민감도 분석은 4장에서 제시한다.

## 3.3.2. 조건부 시계열 잡음 예측기

실제 불량 윈도우 $\mathbf{X}_0$와 마스크 $\mathbf{M}$를 이용해 다음 조건 정보를 구성한다.

$$
\mathbf{C}
=\operatorname{Concat}
\left(\mathbf{M}\odot\mathbf{X}_0,\mathbf{M}\right).
\tag{8}
$$

$\mathbf{C}$는 각 시점에서 보존된 관측값과 해당 값의 관측 여부를 함께 제공한다. 잡음이 추가된 시계열 $\mathbf{X}_s$, 확산 단계 $s$의 임베딩 및 조건 $\mathbf{C}$를 시간 방향으로 결합하여 GRU에 입력한다. GRU의 각 시점 출력은 선형 계층을 거쳐 입력 변수와 동일한 차원의 예측 잡음으로 변환된다.

$$
\widehat{\boldsymbol{\epsilon}}_s
=\boldsymbol{\epsilon}_{\theta}
(\mathbf{X}_s,s,\mathbf{C}).
\tag{9}
$$

모델은 실제 잡음 $\boldsymbol{\epsilon}$과 예측 잡음 $\widehat{\boldsymbol{\epsilon}}_s$의 평균제곱오차를 최소화하도록 학습한다.

$$
\mathcal{L}_{\mathrm{FMD}}
=\mathbb{E}
\left[
\left\|
\boldsymbol{\epsilon}
-\boldsymbol{\epsilon}_{\theta}
(\mathbf{X}_s,s,\mathbf{C})
\right\|_2^2
\right].
\tag{10}
$$

현재 구현은 128차원 은닉 상태의 GRU와 32차원 확산 단계 임베딩을 사용한다. 잡음 분산 $\beta_s$는 총 80단계에 걸쳐 $10^{-4}$에서 0.02까지 선형적으로 증가한다. 이러한 수치는 구현 세부 설정이며, 모델의 핵심은 마스킹된 실제 불량 문맥과 이진 마스크를 동시에 조건으로 사용하는 시간적 잡음 예측 구조에 있다.

## 3.3.3. 시드 기반 역확산과 부분 복원

생성 단계에서는 실제 불량 윈도우를 복원 문맥으로 사용한다. 임의로 선택한 실제 불량 시드 $\mathbf{X}^{\mathrm{seed}}$에 확산 시작 단계 $s_0$에 해당하는 잡음을 추가하고, 조건 $\mathbf{C}$를 유지한 채 $s_0$에서 0까지 역확산을 수행한다. 현재 구현은 전체 80단계 중 $s_0=20$에서 역확산을 시작한다. 이 방식은 순수 잡음에서 전체 윈도우를 생성하는 것보다 실제 시드 주변에서 제한된 변동을 생성하는 역할을 한다.

역확산의 최종 출력 $\widehat{\mathbf{X}}_0$을 얻은 후 관측 영역은 시드 값으로 되돌리고 마스킹 영역만 생성값으로 대체한다.

$$
\widetilde{\mathbf{X}}
=\mathbf{M}\odot\mathbf{X}^{\mathrm{seed}}
+(\mathbf{1}-\mathbf{M})\odot\widehat{\mathbf{X}}_0.
\tag{11}
$$

마지막으로 변수별 생성값을 실제 불량 학습 데이터의 관측 범위에 표준편차 여유분을 더한 구간으로 제한한다. 변수 $k$의 실제 불량 최솟값, 최댓값 및 표준편차를 각각 $x_k^{\min}$, $x_k^{\max}$, $\sigma_{1,k}$라고 하면 현재 구현의 제한 범위는 $[x_k^{\min}-0.25\sigma_{1,k},\ x_k^{\max}+0.25\sigma_{1,k}]$이다. 이 처리는 역확산에서 발생할 수 있는 극단적인 값을 억제한다. 생성된 전체 후보는 다음 단계에서 품질 기준에 따라 선별하며, 그 과정은 3.4절에서 설명한다.

---

## 구현 근거 메모 - 투고 본문에 포함하지 않음

- 구조적 마스크: `make_structured_mask`, `build_adaptive_mask_config`
- 조건부 GRU 잡음 예측기: `TemporalDiffusionDenoiser`
- 학습과 생성: `train_temporal_masked_diffusion_model`, `generate_diffusion_masked`
- 역확산 후 관측값 복원: `restored = mask * seed + (1 - mask) * denoised`
- 범위 제한: `clamp_to_real_range`
- 주요 구현 파일: `tools/research02_generate_data.py`
