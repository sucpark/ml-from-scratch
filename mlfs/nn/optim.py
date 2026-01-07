"""
옵티마이저 모듈 (mlfs.nn.optim)
===============================

SGD, Adam 등의 옵티마이저를 직접 구현합니다.
torch.optim을 사용하지 않고 직접 파라미터를 업데이트합니다.
"""

import torch
from typing import List, Optional
import math


class SGD:
    """
    확률적 경사 하강법 (Stochastic Gradient Descent)

    v = momentum * v - lr * grad
    param = param + v

    Args:
        params: 학습할 파라미터 리스트
        lr: 학습률
        momentum: 모멘텀 계수 (0이면 일반 SGD)
        weight_decay: L2 정규화 계수

    Example:
        >>> layer = Linear(784, 10)
        >>> optimizer = SGD(layer.parameters(), lr=0.01, momentum=0.9)
        >>>
        >>> # 학습 루프
        >>> for epoch in range(epochs):
        ...     output = layer(x)
        ...     loss = criterion(output, y)
        ...     optimizer.zero_grad()
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(
        self,
        params: List[torch.Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # 모멘텀을 위한 속도 벡터 초기화
        self.velocities = [torch.zeros_like(p) for p in params]

    def zero_grad(self) -> None:
        """
        모든 파라미터의 그래디언트를 0으로 초기화합니다.
        """
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self) -> None:
        """
        파라미터를 업데이트합니다.
        """
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue

                grad = param.grad

                # Weight decay (L2 정규화)
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * param

                # 모멘텀 업데이트
                if self.momentum != 0:
                    self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
                    param.add_(self.velocities[i])
                else:
                    param.add_(-self.lr * grad)

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.momentum}, weight_decay={self.weight_decay})"


class Adam:
    """
    Adam 옵티마이저 (Adaptive Moment Estimation)

    모멘텀과 적응형 학습률을 결합한 옵티마이저입니다.

    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad^2
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    param = param - lr * m_hat / (sqrt(v_hat) + eps)

    Args:
        params: 학습할 파라미터 리스트
        lr: 학습률
        betas: (beta1, beta2) 튜플
        eps: 수치 안정성을 위한 작은 값
        weight_decay: L2 정규화 계수

    Example:
        >>> layer = Linear(784, 10)
        >>> optimizer = Adam(layer.parameters(), lr=0.001)
        >>>
        >>> # 학습 루프
        >>> for epoch in range(epochs):
        ...     output = layer(x)
        ...     loss = criterion(output, y)
        ...     optimizer.zero_grad()
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(
        self,
        params: List[torch.Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # 1차/2차 모멘트 초기화
        self.m = [torch.zeros_like(p) for p in params]  # 1차 모멘트 (평균)
        self.v = [torch.zeros_like(p) for p in params]  # 2차 모멘트 (분산)

        # 시간 스텝
        self.t = 0

    def zero_grad(self) -> None:
        """
        모든 파라미터의 그래디언트를 0으로 초기화합니다.
        """
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self) -> None:
        """
        파라미터를 업데이트합니다.
        """
        self.t += 1

        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue

                grad = param.grad

                # Weight decay
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * param

                # 1차 모멘트 업데이트
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

                # 2차 모멘트 업데이트
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

                # 편향 보정
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # 파라미터 업데이트
                param.add_(-self.lr * m_hat / (torch.sqrt(v_hat) + self.eps))

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, betas=({self.beta1}, {self.beta2}))"


class RMSprop:
    """
    RMSprop 옵티마이저

    학습률을 그래디언트의 이동 평균으로 나누어 적응형 학습률을 구현합니다.

    v = alpha * v + (1 - alpha) * grad^2
    param = param - lr * grad / (sqrt(v) + eps)

    Args:
        params: 학습할 파라미터 리스트
        lr: 학습률
        alpha: 이동 평균 계수
        eps: 수치 안정성을 위한 작은 값

    Example:
        >>> layer = Linear(784, 10)
        >>> optimizer = RMSprop(layer.parameters(), lr=0.001)
    """

    def __init__(
        self,
        params: List[torch.Tensor],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
    ):
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.eps = eps

        # 그래디언트 제곱의 이동 평균
        self.v = [torch.zeros_like(p) for p in params]

    def zero_grad(self) -> None:
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self) -> None:
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue

                grad = param.grad

                # 이동 평균 업데이트
                self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (grad ** 2)

                # 파라미터 업데이트
                param.add_(-self.lr * grad / (torch.sqrt(self.v[i]) + self.eps))

    def __repr__(self) -> str:
        return f"RMSprop(lr={self.lr}, alpha={self.alpha})"


class Adagrad:
    """
    Adagrad 옵티마이저

    각 파라미터에 대해 과거 그래디언트의 제곱합을 축적하여 학습률을 조정합니다.

    v = v + grad^2
    param = param - lr * grad / (sqrt(v) + eps)

    Args:
        params: 학습할 파라미터 리스트
        lr: 학습률
        eps: 수치 안정성을 위한 작은 값

    Note:
        학습이 진행될수록 학습률이 계속 감소하는 특성이 있습니다.
    """

    def __init__(
        self,
        params: List[torch.Tensor],
        lr: float = 0.01,
        eps: float = 1e-10,
    ):
        self.params = params
        self.lr = lr
        self.eps = eps

        # 그래디언트 제곱의 누적합
        self.sum_squares = [torch.zeros_like(p) for p in params]

    def zero_grad(self) -> None:
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self) -> None:
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue

                grad = param.grad

                # 제곱합 누적
                self.sum_squares[i] += grad ** 2

                # 파라미터 업데이트
                param.add_(-self.lr * grad / (torch.sqrt(self.sum_squares[i]) + self.eps))

    def __repr__(self) -> str:
        return f"Adagrad(lr={self.lr})"
