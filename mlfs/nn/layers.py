"""
레이어 모듈 (mlfs.nn.layers)
============================

torch.nn.Linear, torch.nn.Conv2d 등을 직접 구현한 모듈입니다.
PyTorch의 autograd만 활용하고, 레이어 연산은 직접 구현합니다.
"""

from typing import Optional, Tuple, List
import math

import torch
import torch.nn.functional as F


class Linear:
    """
    완전연결층 (Fully Connected Layer)

    y = x @ W + b

    Args:
        in_features: 입력 특성 수
        out_features: 출력 특성 수
        bias: 편향 사용 여부

    Example:
        >>> layer = Linear(784, 256)
        >>> x = torch.randn(32, 784)
        >>> output = layer.forward(x)
        >>> output.shape
        torch.Size([32, 256])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features

        # He 초기화 (ReLU에 적합)
        std = math.sqrt(2.0 / in_features)
        self.weight = torch.randn(in_features, out_features) * std
        self.weight.requires_grad_(True)

        if bias:
            self.bias = torch.zeros(out_features)
            self.bias.requires_grad_(True)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            x: (batch_size, in_features) 형태의 입력

        Returns:
            (batch_size, out_features) 형태의 출력
        """
        output = x @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return output

    def parameters(self) -> List[torch.Tensor]:
        """학습 가능한 파라미터 반환"""
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"


class Conv2d:
    """
    2D 합성곱층 (Convolutional Layer)

    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수 (필터 수)
        kernel_size: 커널 크기 (int 또는 tuple)
        stride: 스트라이드
        padding: 패딩

    Example:
        >>> conv = Conv2d(1, 32, kernel_size=3, padding=1)
        >>> x = torch.randn(16, 1, 28, 28)
        >>> output = conv.forward(x)
        >>> output.shape
        torch.Size([16, 32, 28, 28])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He 초기화
        fan_in = in_channels * kernel_size * kernel_size
        std = math.sqrt(2.0 / fan_in)

        self.weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * std
        self.weight.requires_grad_(True)

        self.bias = torch.zeros(out_channels)
        self.bias.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 - F.conv2d 사용 (연산 자체는 PyTorch에 위임)

        Note:
            합성곱 연산의 수학적 구현은 복잡하므로,
            F.conv2d를 사용하되 가중치는 직접 관리합니다.

        Args:
            x: (batch, in_channels, H, W) 형태의 입력

        Returns:
            (batch, out_channels, H', W') 형태의 출력
        """
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )

    def parameters(self) -> List[torch.Tensor]:
        return [self.weight, self.bias]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return (
            f"Conv2d(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size})"
        )


class MaxPool2d:
    """
    2D 최대 풀링층

    Args:
        kernel_size: 풀링 윈도우 크기
        stride: 스트라이드 (기본값: kernel_size)

    Example:
        >>> pool = MaxPool2d(kernel_size=2)
        >>> x = torch.randn(16, 32, 28, 28)
        >>> output = pool.forward(x)
        >>> output.shape
        torch.Size([16, 32, 14, 14])
    """

    def __init__(self, kernel_size: int = 2, stride: Optional[int] = None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        return F.max_pool2d(x, self.kernel_size, self.stride)

    def parameters(self) -> List[torch.Tensor]:
        return []  # 학습 가능한 파라미터 없음

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride})"


class Flatten:
    """
    텐서를 1D로 펼칩니다.

    Example:
        >>> flatten = Flatten()
        >>> x = torch.randn(16, 32, 7, 7)
        >>> output = flatten.forward(x)
        >>> output.shape
        torch.Size([16, 1568])
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, C, H, W) -> (batch, C*H*W)"""
        return x.view(x.size(0), -1)

    def parameters(self) -> List[torch.Tensor]:
        return []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return "Flatten()"


class Dropout:
    """
    드롭아웃 정규화

    학습 시 무작위로 뉴런을 비활성화합니다.

    Args:
        p: 드롭아웃 확률 (0~1)

    Example:
        >>> dropout = Dropout(p=0.5)
        >>> dropout.train()
        >>> x = torch.ones(10, 100)
        >>> output = dropout.forward(x)
        >>> # 약 절반의 값이 0이 됨
    """

    def __init__(self, p: float = 0.5):
        self.p = p
        self.training = True

    def train(self):
        """훈련 모드로 설정"""
        self.training = True

    def eval(self):
        """평가 모드로 설정"""
        self.training = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        if not self.training or self.p == 0:
            return x

        # 마스크 생성 및 스케일링
        mask = (torch.rand_like(x) > self.p).float()
        return x * mask / (1 - self.p)

    def parameters(self) -> List[torch.Tensor]:
        return []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


class BatchNorm1d:
    """
    1D 배치 정규화

    Args:
        num_features: 특성 수
        eps: 분모에 더할 작은 값 (수치 안정성)
        momentum: 이동 평균 업데이트 비율

    Example:
        >>> bn = BatchNorm1d(256)
        >>> x = torch.randn(32, 256)
        >>> output = bn.forward(x)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # 학습 가능한 파라미터
        self.gamma = torch.ones(num_features)
        self.gamma.requires_grad_(True)
        self.beta = torch.zeros(num_features)
        self.beta.requires_grad_(True)

        # 이동 평균 (학습 중 업데이트)
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        if self.training:
            # 배치 통계 계산
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

            # 이동 평균 업데이트
            with torch.no_grad():
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean + self.momentum * mean
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var + self.momentum * var
                )
        else:
            mean = self.running_mean
            var = self.running_var

        # 정규화
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 스케일 및 이동
        return self.gamma * x_norm + self.beta

    def parameters(self) -> List[torch.Tensor]:
        return [self.gamma, self.beta]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"BatchNorm1d(num_features={self.num_features})"
