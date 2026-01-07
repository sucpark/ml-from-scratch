"""
앙상블 모듈 (mlfs.classical.ensemble)
======================================

AdaBoost 등의 앙상블 방법을 구현합니다.
"""

import torch
from typing import Optional, List


class DecisionStump:
    """
    결정 스텀프 (약한 분류기)

    단일 특성에 대한 임계값 기반 분류기입니다.
    """

    def __init__(self):
        self.feature_idx: int = 0
        self.threshold: float = 0.0
        self.polarity: int = 1  # 1 또는 -1

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        weights: torch.Tensor,
    ) -> float:
        """
        가중 오류를 최소화하는 스텀프 학습

        Args:
            X: (n_samples, n_features) 형태의 데이터
            y: (n_samples,) 형태의 레이블 (-1 또는 1)
            weights: (n_samples,) 형태의 샘플 가중치

        Returns:
            가중 오류율
        """
        n_samples, n_features = X.shape
        min_error = float('inf')

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = torch.unique(feature_values)

            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = torch.ones(n_samples)
                    if polarity == 1:
                        predictions[feature_values < threshold] = -1
                    else:
                        predictions[feature_values >= threshold] = -1

                    # 가중 오류 계산
                    error = (weights * (predictions != y).float()).sum()

                    if error < min_error:
                        min_error = error
                        self.feature_idx = feature_idx
                        self.threshold = threshold.item()
                        self.polarity = polarity

        return min_error.item()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """예측"""
        n_samples = X.size(0)
        feature_values = X[:, self.feature_idx]
        predictions = torch.ones(n_samples)

        if self.polarity == 1:
            predictions[feature_values < self.threshold] = -1
        else:
            predictions[feature_values >= self.threshold] = -1

        return predictions


class AdaBoost:
    """
    AdaBoost 분류기

    약한 분류기들을 결합하여 강한 분류기를 만듭니다.

    Args:
        n_estimators: 약한 분류기 수
        learning_rate: 학습률 (alpha 스케일링)

    Example:
        >>> model = AdaBoost(n_estimators=50)
        >>> X = torch.randn(100, 10)
        >>> y = torch.randint(0, 2, (100,)) * 2 - 1  # -1 또는 1
        >>> model.fit(X, y)
        >>> predictions = model.predict(X)
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.stumps: List[DecisionStump] = []
        self.alphas: List[float] = []

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "AdaBoost":
        """
        AdaBoost 학습

        Args:
            X: (n_samples, n_features) 형태의 데이터
            y: (n_samples,) 형태의 레이블 (-1 또는 1)

        Returns:
            self
        """
        n_samples = X.size(0)

        # 가중치 초기화
        weights = torch.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            # 약한 분류기 학습
            stump = DecisionStump()
            error = stump.fit(X, y, weights)

            # 오류가 0.5 이상이면 중단
            error = max(error, 1e-10)
            if error >= 0.5:
                break

            # alpha 계산
            alpha = self.learning_rate * 0.5 * torch.log(
                torch.tensor((1 - error) / error)
            ).item()

            # 예측
            predictions = stump.predict(X)

            # 가중치 업데이트
            weights = weights * torch.exp(-alpha * y * predictions)
            weights = weights / weights.sum()

            self.stumps.append(stump)
            self.alphas.append(alpha)

        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        예측

        Args:
            X: 입력 데이터

        Returns:
            예측 레이블 (-1 또는 1)
        """
        n_samples = X.size(0)
        predictions = torch.zeros(n_samples)

        for alpha, stump in zip(self.alphas, self.stumps):
            predictions += alpha * stump.predict(X)

        return torch.sign(predictions)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """확률 예측 (sigmoid 적용)"""
        n_samples = X.size(0)
        raw_predictions = torch.zeros(n_samples)

        for alpha, stump in zip(self.alphas, self.stumps):
            raw_predictions += alpha * stump.predict(X)

        # Sigmoid로 확률 변환
        proba = torch.sigmoid(raw_predictions)
        return torch.stack([1 - proba, proba], dim=1)
