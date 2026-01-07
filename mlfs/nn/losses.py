"""
손실 함수 모듈 (mlfs.nn.losses)
===============================

CrossEntropyLoss, MSELoss 등의 손실 함수를 직접 구현합니다.
"""

import torch
from typing import Optional

from .activations import softmax


class CrossEntropyLoss:
    """
    교차 엔트로피 손실 함수

    분류 문제에 사용됩니다. Softmax + Negative Log Likelihood를 결합한 형태입니다.

    L = -1/N * sum(log(softmax(x)[y]))

    Args:
        reduction: 'mean', 'sum', 'none' 중 하나

    Example:
        >>> criterion = CrossEntropyLoss()
        >>> logits = torch.randn(32, 10)  # (batch_size, num_classes)
        >>> labels = torch.randint(0, 10, (32,))  # (batch_size,)
        >>> loss = criterion(logits, labels)
    """

    def __init__(self, reduction: str = "mean"):
        assert reduction in ["mean", "sum", "none"]
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        손실 계산

        Args:
            logits: (batch_size, num_classes) 형태의 로짓
            targets: (batch_size,) 형태의 정답 레이블 (정수)

        Returns:
            스칼라 손실 값
        """
        batch_size = logits.size(0)

        # Softmax 적용 (수치 안정성 포함)
        probs = softmax(logits, dim=1)

        # 정답 클래스의 확률 추출
        # targets를 인덱스로 사용
        target_probs = probs[torch.arange(batch_size), targets]

        # Negative log likelihood
        # 0에 가까운 확률에 대한 수치 안정성을 위해 clamp
        nll = -torch.log(target_probs.clamp(min=1e-10))

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        else:
            return nll

    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(logits, targets)

    def __repr__(self) -> str:
        return f"CrossEntropyLoss(reduction='{self.reduction}')"


class MSELoss:
    """
    평균 제곱 오차 손실 함수

    회귀 문제에 사용됩니다.

    L = 1/N * sum((y_pred - y_true)^2)

    Args:
        reduction: 'mean', 'sum', 'none' 중 하나

    Example:
        >>> criterion = MSELoss()
        >>> predictions = torch.randn(32, 1)
        >>> targets = torch.randn(32, 1)
        >>> loss = criterion(predictions, targets)
    """

    def __init__(self, reduction: str = "mean"):
        assert reduction in ["mean", "sum", "none"]
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        손실 계산

        Args:
            predictions: 예측값
            targets: 정답값

        Returns:
            손실 값
        """
        squared_errors = (predictions - targets) ** 2

        if self.reduction == "mean":
            return squared_errors.mean()
        elif self.reduction == "sum":
            return squared_errors.sum()
        else:
            return squared_errors

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(predictions, targets)

    def __repr__(self) -> str:
        return f"MSELoss(reduction='{self.reduction}')"


class BCELoss:
    """
    이진 교차 엔트로피 손실 함수

    이진 분류 문제에 사용됩니다.

    L = -1/N * sum(y * log(p) + (1-y) * log(1-p))

    Args:
        reduction: 'mean', 'sum', 'none' 중 하나

    Example:
        >>> criterion = BCELoss()
        >>> probs = torch.sigmoid(torch.randn(32, 1))  # (batch_size, 1)
        >>> targets = torch.randint(0, 2, (32, 1)).float()
        >>> loss = criterion(probs, targets)
    """

    def __init__(self, reduction: str = "mean"):
        assert reduction in ["mean", "sum", "none"]
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        손실 계산

        Args:
            predictions: 예측 확률 (0~1 범위, sigmoid 적용 후)
            targets: 정답 레이블 (0 또는 1)

        Returns:
            손실 값
        """
        # 수치 안정성을 위한 클리핑
        eps = 1e-10
        predictions = predictions.clamp(eps, 1 - eps)

        # Binary cross entropy
        bce = -(targets * torch.log(predictions) + (1 - targets) * torch.log(1 - predictions))

        if self.reduction == "mean":
            return bce.mean()
        elif self.reduction == "sum":
            return bce.sum()
        else:
            return bce

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(predictions, targets)

    def __repr__(self) -> str:
        return f"BCELoss(reduction='{self.reduction}')"


class BCEWithLogitsLoss:
    """
    이진 교차 엔트로피 + Sigmoid

    Sigmoid를 적용하지 않은 로짓을 입력으로 받습니다.
    수치적으로 더 안정적입니다.

    Args:
        reduction: 'mean', 'sum', 'none' 중 하나

    Example:
        >>> criterion = BCEWithLogitsLoss()
        >>> logits = torch.randn(32, 1)  # sigmoid 적용 전
        >>> targets = torch.randint(0, 2, (32, 1)).float()
        >>> loss = criterion(logits, targets)
    """

    def __init__(self, reduction: str = "mean"):
        assert reduction in ["mean", "sum", "none"]
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        손실 계산

        수치 안정성을 위해 다음 공식을 사용:
        max(logits, 0) - logits * targets + log(1 + exp(-|logits|))
        """
        # 수치적으로 안정한 구현
        max_val = torch.clamp(-logits, min=0)
        loss = max_val + (logits - logits * targets) + torch.log(
            torch.exp(-max_val) + torch.exp(-logits - max_val)
        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(logits, targets)

    def __repr__(self) -> str:
        return f"BCEWithLogitsLoss(reduction='{self.reduction}')"
