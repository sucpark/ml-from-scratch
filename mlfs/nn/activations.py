"""
활성화 함수 모듈 (mlfs.nn.activations)
======================================

ReLU, Sigmoid 등의 활성화 함수를 구현합니다.
torch.nn.functional을 사용할 수 있지만, 이해를 위해 직접 구현도 포함합니다.
"""

import torch
import torch.nn.functional as F
from typing import List


def relu(x: torch.Tensor) -> torch.Tensor:
    """
    ReLU 활성화 함수

    f(x) = max(0, x)

    Args:
        x: 입력 텐서

    Returns:
        ReLU가 적용된 텐서

    Example:
        >>> x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        >>> relu(x)
        tensor([0., 0., 1., 2.])
    """
    return torch.maximum(x, torch.zeros_like(x))


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Sigmoid 활성화 함수

    f(x) = 1 / (1 + exp(-x))

    Args:
        x: 입력 텐서

    Returns:
        Sigmoid가 적용된 텐서 (0~1 범위)

    Example:
        >>> x = torch.tensor([0.0])
        >>> sigmoid(x)
        tensor([0.5000])
    """
    return 1 / (1 + torch.exp(-x))


def tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Tanh 활성화 함수

    f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Args:
        x: 입력 텐서

    Returns:
        Tanh가 적용된 텐서 (-1~1 범위)

    Example:
        >>> x = torch.tensor([0.0])
        >>> tanh(x)
        tensor([0.])
    """
    return torch.tanh(x)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax 활성화 함수

    f(x_i) = exp(x_i) / sum(exp(x_j))

    Args:
        x: 입력 텐서
        dim: softmax를 적용할 차원

    Returns:
        확률 분포 텐서 (합이 1)

    Example:
        >>> x = torch.tensor([[1.0, 2.0, 3.0]])
        >>> softmax(x, dim=1)
        tensor([[0.0900, 0.2447, 0.6652]])
    """
    # 수치 안정성을 위해 최댓값을 빼줌
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """
    Leaky ReLU 활성화 함수

    f(x) = x if x > 0 else negative_slope * x

    Args:
        x: 입력 텐서
        negative_slope: 음수 영역의 기울기

    Returns:
        Leaky ReLU가 적용된 텐서

    Example:
        >>> x = torch.tensor([-1.0, 0.0, 1.0])
        >>> leaky_relu(x, negative_slope=0.1)
        tensor([-0.1000,  0.0000,  1.0000])
    """
    return torch.where(x > 0, x, negative_slope * x)


# 레이어 형태의 활성화 함수 클래스들

class ReLU:
    """ReLU 활성화 레이어"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return relu(x)

    def parameters(self) -> List[torch.Tensor]:
        return []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid:
    """Sigmoid 활성화 레이어"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sigmoid(x)

    def parameters(self) -> List[torch.Tensor]:
        return []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh:
    """Tanh 활성화 레이어"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return tanh(x)

    def parameters(self) -> List[torch.Tensor]:
        return []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return "Tanh()"


class Softmax:
    """Softmax 활성화 레이어"""

    def __init__(self, dim: int = -1):
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return softmax(x, dim=self.dim)

    def parameters(self) -> List[torch.Tensor]:
        return []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"Softmax(dim={self.dim})"


class LeakyReLU:
    """Leaky ReLU 활성화 레이어"""

    def __init__(self, negative_slope: float = 0.01):
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return leaky_relu(x, self.negative_slope)

    def parameters(self) -> List[torch.Tensor]:
        return []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"LeakyReLU(negative_slope={self.negative_slope})"
