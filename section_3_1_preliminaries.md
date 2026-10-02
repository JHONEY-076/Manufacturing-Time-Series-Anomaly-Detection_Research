# 3.1. 사전 지식 (Preliminaries)

본 절에서는 제안하는 Filtered Masking Diffusion(FMD)을 설명하기 위해 필요한 다변량 시계열 마스킹과 확산 확률 모델의 기본 개념을 정리한다. 마스킹은 실제 불량 윈도우에서 관측된 정보를 조건으로 유지하면서 일부 시간 구간과 변수 그룹을 복원 대상으로 지정한다. 확산 모델은 잡음이 추가된 시계열에서 잡음을 추정하고 제거하는 과정을 학습하여 마스킹된 영역의 새로운 값을 생성한다.

## 3.1.1. 다변량 시계열의 구조적 마스킹

길이가 $T$이고 변수의 수가 $d$인 $i$번째 다변량 시계열 윈도우를 $\mathbf{X}_i\in\mathbb{R}^{T\times d}$로 나타낸다. 마스킹 행렬 $\mathbf{M}_i\in\{0,1\}^{T\times d}$에서 1은 보존할 관측값, 0은 생성 모델이 복원할 값을 의미한다. 모델에 조건으로 제공되는 관측 영역은 다음과 같다.

$$
\mathbf{X}_i^{\mathrm{obs}}
=\mathbf{M}_i\odot\mathbf{X}_i,
\tag{1}
$$

여기서 $\odot$는 원소별 곱을 나타낸다. 마스킹된 입력만 사용하면 실제 값이 0인 경우와 마스킹 때문에 0이 된 경우를 구분하기 어렵다. 따라서 생성 모델에는 $\mathbf{X}_i^{\mathrm{obs}}$와 $\mathbf{M}_i$를 함께 제공한다.

시계열 증강에서는 시간적 의존성과 변수 간 관계를 유지하는 것이 중요하다 [2]. 독립적인 원소를 무작위로 제거하는 방식은 이러한 구조를 충분히 반영하기 어렵다. 본 연구에서는 연속된 시간 구간 $\mathcal{I}_i$와 서로 관련된 변수 그룹 $\mathcal{G}_i$를 마스킹 대상으로 사용한다. 이에 따라 마스크의 각 원소는 다음과 같이 정의할 수 있다.

$$
M_{i,t,k}=
\begin{cases}
0, & t\in\mathcal{I}_i\ \text{or}\ k\in\mathcal{G}_i,\\
1, & \text{otherwise},
\end{cases}
\tag{2}
$$

여기서 $t$는 윈도우 내부의 시간 위치이고 $k$는 공정 변수를 나타낸다. 구조적 마스킹은 선택된 연속 시간 구간의 모든 변수와 선택된 변수 그룹의 전체 시간 구간을 복원 대상으로 설정한다. 이 방식은 실제 불량 윈도우의 나머지 관측값을 유지하면서 시간적·변수적 일부만 변화시키므로, 제한된 불량 표본의 구조를 보존한 조건부 증강에 적합하다. 본 연구에서 마스킹 위치와 변수 그룹을 선택하는 구체적인 방법은 3.3절에서 설명한다.

## 3.1.2. 확산 확률 모델

확산 모델은 원본 데이터에 점진적으로 가우스 잡음을 추가하는 순방향 과정과, 잡음이 포함된 데이터에서 원본을 복원하는 역방향 과정으로 구성된다 [1]. 확산 단계의 수를 $S$라 하고, $s\in\{1,\ldots,S\}$를 확산 단계로 나타낸다. 잡음 분산 스케줄을 $\beta_s\in(0,1)$, $\alpha_s=1-\beta_s$, $\bar{\alpha}_s=\prod_{r=1}^{s}\alpha_r$로 정의하면 순방향 과정은 다음과 같다.

$$
q(\mathbf{X}_s\mid\mathbf{X}_{s-1})
=\mathcal{N}\!\left(
\sqrt{\alpha_s}\mathbf{X}_{s-1},
\beta_s\mathbf{I}
\right).
\tag{3}
$$

임의의 확산 단계 $s$에서 잡음이 추가된 표본은 다음과 같이 직접 계산할 수 있다.

$$
\mathbf{X}_s
=\sqrt{\bar{\alpha}_s}\mathbf{X}_0
+\sqrt{1-\bar{\alpha}_s}\boldsymbol{\epsilon},
\qquad
\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
\tag{4}
$$

역방향 과정에서는 신경망 $\boldsymbol{\epsilon}_{\theta}$가 $\mathbf{X}_s$에 포함된 잡음을 예측한다. 일반적인 DDPM의 학습 목적은 실제 잡음과 예측 잡음 사이의 평균제곱오차를 최소화하는 것이다.

$$
\mathcal{L}_{\mathrm{diff}}
=\mathbb{E}_{\mathbf{X}_0,s,\boldsymbol{\epsilon}}
\left[
\left\|
\boldsymbol{\epsilon}
-\boldsymbol{\epsilon}_{\theta}(\mathbf{X}_s,s)
\right\|_2^2
\right].
\tag{5}
$$

FMD에서는 잡음 예측기가 마스킹된 실제 불량 윈도우와 마스크를 조건으로 받도록 식 (5)를 확장한다. 조건 정보를 $\mathbf{C}_i=[\mathbf{M}_i\odot\mathbf{X}_i,\mathbf{M}_i]$로 나타내면 조건부 잡음 예측기는 $\boldsymbol{\epsilon}_{\theta}(\mathbf{X}_s,s,\mathbf{C}_i)$가 된다. 이 구성은 관측 영역과 마스킹 위치를 명시적으로 전달하여, 역확산 과정이 실제 불량 윈도우의 문맥에 맞는 값을 복원하도록 한다. 구체적인 조건부 시계열 복원 구조와 생성 절차는 3.3절에서 제시한다.

### 참고문헌

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems, 33*, 6840-6851.
2. Wen, Q., Sun, L., Yang, F., Song, X., Gao, J., Wang, X., & Xu, H. (2021). Time series data augmentation for deep learning: A survey. *Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence*, 4653-4660. https://doi.org/10.24963/ijcai.2021/631

---

## 작성 메모 - 투고 본문에 포함하지 않음

- 참고 논문의 3.1절처럼 제안 모델을 이해하는 데 필요한 기본 원리만 배치하였다. 영상용 U-Net 설명은 현재 FMD 구현과 다르므로 가져오지 않았다.
- 마스킹의 데이터 적응형 확률, GRU 잡음 예측기 및 품질 필터는 제안 방법의 고유 구성에 해당하므로 각각 3.3절과 3.4절에서 설명한다.
- 기존 문제 정의의 데이터 수, 윈도우 레이블 및 클래스 불균형에 관한 실험 설정은 4.1절에 배치한다.
