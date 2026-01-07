# ML From Scratch

PyTorch의 autograd만 활용하여 머신러닝 알고리즘을 밑바닥부터 직접 구현한 프로젝트입니다.

> 이 프로젝트는 **University of Chicago CS-35400 (Machine Learning)** 수업의 학습 내용을 현대적인 구조로 재구성한 것입니다.

## 특징

- **직접 구현**: `torch.nn.Linear`, `torch.optim.SGD` 등을 사용하지 않고 직접 구현
- **PyTorch autograd 활용**: 자동 미분만 활용하여 역전파 구현
- **학습 친화적**: 번호순 노트북으로 단계별 학습 가능
- **sklearn 스타일 API**: `fit()`, `predict()`, `transform()` 인터페이스

## Quick Start

### 1. 환경 설정

```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 클론 및 설정
git clone https://github.com/username/ml-from-scratch.git
cd ml-from-scratch
uv sync
```

### 2. 노트북 실행

```bash
uv run jupyter notebook notebooks/
```

### 3. 데이터

- MNIST, CIFAR-10 등의 데이터는 **첫 실행 시 자동 다운로드**됩니다
- 별도 설정이 필요 없습니다!

## 프로젝트 구조

```
ml-from-scratch/
├── mlfs/                          # 핵심 패키지
│   ├── nn/                        # 신경망 모듈
│   │   ├── layers.py              # Linear, Conv2d, MaxPool2d
│   │   ├── activations.py         # ReLU, Sigmoid, Softmax
│   │   ├── losses.py              # CrossEntropyLoss, MSELoss
│   │   ├── optim.py               # SGD, Adam
│   │   └── models.py              # MLP, CNN
│   ├── classical/                 # 전통 ML
│   │   ├── clustering.py          # KMeans, SpectralClustering
│   │   ├── reduction.py           # PCA, LLE, Isomap
│   │   └── ensemble.py            # AdaBoost
│   ├── probabilistic/             # 확률 모델
│   │   ├── em.py                  # EM Algorithm, GMM
│   │   └── sparse.py              # Sparse Coding
│   └── utils/                     # 유틸리티
│       ├── data.py                # 데이터 로딩
│       └── viz.py                 # 시각화
└── notebooks/                     # 튜토리얼 노트북
```

## 학습 경로

| # | 노트북 | 주제 | 핵심 개념 |
|---|--------|------|----------|
| 01 | `perceptron` | 퍼셉트론 | 단일 뉴런, 경사하강법 |
| 02 | `mlp` | 다층 퍼셉트론 | 역전파, 은닉층 |
| 03 | `cnn_mnist` | CNN (MNIST) | 합성곱, 풀링 |
| 04 | `cnn_cifar` | CNN (CIFAR) | 컬러 이미지, 깊은 네트워크 |
| 05 | `kmeans` | K-Means | 클러스터링, K-Means++ |
| 06 | `spectral_clustering` | 스펙트럴 클러스터링 | 그래프 라플라시안 |
| 07 | `pca` | PCA | 차원축소, 고유값 분해 |
| 08 | `lle` | LLE | 지역 선형 임베딩 |
| 09 | `isomap` | Isomap | 측지선 거리, 매니폴드 |
| 10 | `ensemble` | 앙상블 | AdaBoost, 약한 분류기 |
| 11 | `em_algorithm` | EM 알고리즘 | GMM, 기댓값 최대화 |
| 12 | `face_detection` | 얼굴 인식 | Haar-like, Viola-Jones |
| 13 | `sparse_coding` | 희소 코딩 | 희소 표현, 딕셔너리 |

## PyTorch 사용 규칙

### 허용 (사용 가능)
```python
import torch
import torch.nn.functional as F

# 텐서 연산
x = torch.randn(32, 784)
y = x @ weight + bias

# 자동 미분
loss.backward()
param.grad

# 기본 함수
F.relu(x), F.softmax(x, dim=1)
```

### 금지 (직접 구현)
```python
# 레이어 - 직접 구현해야 함
torch.nn.Linear      # → mlfs.nn.layers.Linear
torch.nn.Conv2d      # → mlfs.nn.layers.Conv2d

# 옵티마이저 - 직접 구현해야 함
torch.optim.SGD      # → mlfs.nn.optim.SGD
torch.optim.Adam     # → mlfs.nn.optim.Adam
```

## 사용 예시

```python
from mlfs.nn.layers import Linear
from mlfs.nn.optim import SGD
from mlfs.utils.data import load_mnist

# 데이터 로드
X_train, y_train = load_mnist(train=True)

# 모델 생성 (직접 구현한 레이어)
layer = Linear(784, 10)

# 순전파
output = layer.forward(X_train[:32])

# 옵티마이저 (직접 구현)
optimizer = SGD(layer.parameters(), lr=0.01)
```

## 라이선스

MIT License

## 참고

- 이 프로젝트는 교육 목적으로 제작되었습니다.
- University of Chicago CS-35400 (Machine Learning) 수업 내용을 기반으로 합니다.
