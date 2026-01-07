"""
EM 알고리즘 모듈 (mlfs.probabilistic.em)
=========================================

EM 알고리즘과 GMM을 구현합니다.
"""

import torch
import math
from typing import Optional, Tuple


class GMM:
    """
    가우시안 혼합 모델 (Gaussian Mixture Model)

    EM 알고리즘을 사용하여 학습합니다.

    Args:
        n_components: 가우시안 성분 수
        max_iters: 최대 반복 횟수
        tol: 수렴 허용 오차
        random_state: 랜덤 시드

    Example:
        >>> gmm = GMM(n_components=3)
        >>> X = torch.randn(100, 2)
        >>> gmm.fit(X)
        >>> labels = gmm.predict(X)
        >>> proba = gmm.predict_proba(X)
    """

    def __init__(
        self,
        n_components: int = 3,
        max_iters: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

        # 파라미터
        self.weights_: Optional[torch.Tensor] = None  # (n_components,)
        self.means_: Optional[torch.Tensor] = None  # (n_components, n_features)
        self.covariances_: Optional[torch.Tensor] = None  # (n_components, n_features, n_features)

    def fit(self, X: torch.Tensor) -> "GMM":
        """
        EM 알고리즘으로 GMM 학습

        Args:
            X: (n_samples, n_features) 형태의 데이터

        Returns:
            self
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        n_samples, n_features = X.shape
        K = self.n_components

        # 초기화
        self.weights_ = torch.ones(K) / K

        # K-Means 스타일 초기화
        indices = torch.randperm(n_samples)[:K]
        self.means_ = X[indices].clone()

        # 공분산 초기화 (단위 행렬)
        self.covariances_ = torch.stack([torch.eye(n_features) for _ in range(K)])

        prev_log_likelihood = float('-inf')

        for iteration in range(self.max_iters):
            # E-step: 책임 계산
            responsibilities = self._e_step(X)

            # M-step: 파라미터 업데이트
            self._m_step(X, responsibilities)

            # 로그 우도 계산
            log_likelihood = self._compute_log_likelihood(X)

            # 수렴 확인
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                break

            prev_log_likelihood = log_likelihood

        return self

    def _e_step(self, X: torch.Tensor) -> torch.Tensor:
        """
        E-step: 각 데이터 포인트에 대한 책임 계산

        Returns:
            (n_samples, n_components) 형태의 책임 행렬
        """
        n_samples = X.size(0)
        K = self.n_components

        # 각 성분에서의 확률 밀도 계산
        weighted_likelihoods = torch.zeros(n_samples, K)

        for k in range(K):
            likelihood = self._multivariate_gaussian(X, self.means_[k], self.covariances_[k])
            weighted_likelihoods[:, k] = self.weights_[k] * likelihood

        # 정규화
        total = weighted_likelihoods.sum(dim=1, keepdim=True)
        responsibilities = weighted_likelihoods / (total + 1e-10)

        return responsibilities

    def _m_step(
        self,
        X: torch.Tensor,
        responsibilities: torch.Tensor,
    ) -> None:
        """
        M-step: 파라미터 업데이트
        """
        n_samples, n_features = X.shape
        K = self.n_components

        # N_k: 각 성분에 할당된 유효 샘플 수
        N_k = responsibilities.sum(dim=0)

        # 가중치 업데이트
        self.weights_ = N_k / n_samples

        # 평균 업데이트
        for k in range(K):
            self.means_[k] = (responsibilities[:, k:k+1] * X).sum(dim=0) / (N_k[k] + 1e-10)

        # 공분산 업데이트
        for k in range(K):
            diff = X - self.means_[k]  # (n_samples, n_features)
            weighted_diff = responsibilities[:, k:k+1] * diff  # (n_samples, n_features)
            self.covariances_[k] = (weighted_diff.T @ diff) / (N_k[k] + 1e-10)

            # 정규화 (수치 안정성)
            self.covariances_[k] += torch.eye(n_features) * 1e-6

    def _multivariate_gaussian(
        self,
        X: torch.Tensor,
        mean: torch.Tensor,
        cov: torch.Tensor,
    ) -> torch.Tensor:
        """다변량 가우시안 확률 밀도 계산"""
        n_features = X.size(1)

        diff = X - mean  # (n_samples, n_features)

        # 공분산 역행렬과 행렬식
        try:
            cov_inv = torch.linalg.inv(cov)
            cov_det = torch.linalg.det(cov)
        except:
            cov_reg = cov + torch.eye(n_features) * 1e-6
            cov_inv = torch.linalg.inv(cov_reg)
            cov_det = torch.linalg.det(cov_reg)

        # 마할라노비스 거리
        mahal = (diff @ cov_inv * diff).sum(dim=1)

        # 정규화 상수
        norm_const = 1.0 / (
            (2 * math.pi) ** (n_features / 2) * torch.sqrt(cov_det.abs() + 1e-10)
        )

        return norm_const * torch.exp(-0.5 * mahal)

    def _compute_log_likelihood(self, X: torch.Tensor) -> float:
        """로그 우도 계산"""
        n_samples = X.size(0)
        K = self.n_components

        total_likelihood = torch.zeros(n_samples)

        for k in range(K):
            likelihood = self._multivariate_gaussian(X, self.means_[k], self.covariances_[k])
            total_likelihood += self.weights_[k] * likelihood

        return torch.log(total_likelihood + 1e-10).sum().item()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """가장 높은 책임을 가진 성분 예측"""
        responsibilities = self._e_step(X)
        return torch.argmax(responsibilities, dim=1)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """각 성분에 대한 책임(확률) 반환"""
        return self._e_step(X)


# EM 클래스는 GMM의 별칭으로 제공
EM = GMM
