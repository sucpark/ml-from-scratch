"""
희소 코딩 모듈 (mlfs.probabilistic.sparse)
==========================================

Sparse Coding과 Dictionary Learning을 구현합니다.
"""

import torch
from typing import Optional, Tuple


class SparseCoding:
    """
    희소 코딩 (Sparse Coding)

    데이터를 희소 계수와 딕셔너리의 조합으로 표현합니다.
    X ≈ D @ A, where A is sparse

    Args:
        n_components: 딕셔너리 원자(atom) 수
        alpha: 희소성 정규화 강도
        max_iters: 최대 반복 횟수
        tol: 수렴 허용 오차

    Example:
        >>> sc = SparseCoding(n_components=100, alpha=0.1)
        >>> X = torch.randn(500, 64)  # 500개의 64차원 데이터
        >>> sc.fit(X)
        >>> codes = sc.transform(X)
    """

    def __init__(
        self,
        n_components: int = 100,
        alpha: float = 0.1,
        max_iters: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

        self.dictionary_: Optional[torch.Tensor] = None  # (n_features, n_components)

    def fit(self, X: torch.Tensor) -> "SparseCoding":
        """
        딕셔너리 학습

        교대 최소화(Alternating Minimization)를 사용합니다:
        1. 딕셔너리 D 고정, 희소 코드 A 업데이트
        2. 희소 코드 A 고정, 딕셔너리 D 업데이트

        Args:
            X: (n_samples, n_features) 형태의 데이터

        Returns:
            self
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        n_samples, n_features = X.shape

        # 딕셔너리 초기화 (랜덤 정규화된 열)
        self.dictionary_ = torch.randn(n_features, self.n_components)
        self.dictionary_ = self.dictionary_ / self.dictionary_.norm(dim=0, keepdim=True)

        prev_loss = float('inf')

        for _ in range(self.max_iters):
            # 희소 코드 업데이트 (ISTA)
            codes = self._sparse_encode(X)

            # 딕셔너리 업데이트
            self._update_dictionary(X, codes)

            # 손실 계산
            reconstruction = codes @ self.dictionary_.T
            recon_loss = ((X - reconstruction) ** 2).sum()
            sparse_loss = self.alpha * codes.abs().sum()
            total_loss = recon_loss + sparse_loss

            # 수렴 확인
            if abs(total_loss - prev_loss) < self.tol:
                break

            prev_loss = total_loss.item()

        return self

    def _sparse_encode(
        self,
        X: torch.Tensor,
        max_iters: int = 100,
    ) -> torch.Tensor:
        """
        ISTA (Iterative Shrinkage-Thresholding Algorithm)를 사용한 희소 코딩

        min_A ||X - D @ A||^2 + alpha * ||A||_1
        """
        n_samples = X.size(0)
        D = self.dictionary_  # (n_features, n_components)

        # 초기화
        codes = torch.zeros(n_samples, self.n_components)

        # 스텝 크기 (Lipschitz 상수의 역수)
        L = torch.linalg.eigvalsh(D.T @ D).max()
        step_size = 1.0 / L

        for _ in range(max_iters):
            # 그래디언트 스텝
            residual = X - codes @ D.T
            gradient = -residual @ D

            codes = codes - step_size * gradient

            # Soft thresholding (Proximal operator for L1)
            threshold = self.alpha * step_size
            codes = torch.sign(codes) * torch.clamp(codes.abs() - threshold, min=0)

        return codes

    def _update_dictionary(
        self,
        X: torch.Tensor,
        codes: torch.Tensor,
    ) -> None:
        """
        딕셔너리 업데이트 (최소제곱)

        min_D ||X - D @ A||^2 s.t. ||D[:, k]||_2 = 1
        """
        # 최소제곱 해
        # D = X.T @ A @ (A.T @ A)^-1
        AtA = codes.T @ codes
        AtA = AtA + torch.eye(self.n_components) * 1e-6  # 정규화

        XtA = X.T @ codes
        self.dictionary_ = XtA @ torch.linalg.inv(AtA)

        # 열 정규화
        self.dictionary_ = self.dictionary_ / (
            self.dictionary_.norm(dim=0, keepdim=True) + 1e-10
        )

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        데이터를 희소 코드로 변환

        Args:
            X: (n_samples, n_features) 형태의 데이터

        Returns:
            (n_samples, n_components) 형태의 희소 코드
        """
        if self.dictionary_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self._sparse_encode(X)

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """학습 후 변환"""
        self.fit(X)
        return self.transform(X)

    def reconstruct(self, X: torch.Tensor) -> torch.Tensor:
        """
        데이터 재구성

        Args:
            X: 입력 데이터

        Returns:
            재구성된 데이터
        """
        codes = self.transform(X)
        return codes @ self.dictionary_.T
