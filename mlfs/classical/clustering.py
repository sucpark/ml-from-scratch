"""
클러스터링 모듈 (mlfs.classical.clustering)
============================================

K-Means, Spectral Clustering 등을 직접 구현합니다.
"""

import torch
from typing import Optional, Tuple


class KMeans:
    """
    K-Means 클러스터링

    Args:
        n_clusters: 클러스터 수
        max_iters: 최대 반복 횟수
        tol: 수렴 허용 오차
        random_state: 랜덤 시드

    Example:
        >>> kmeans = KMeans(n_clusters=3)
        >>> X = torch.randn(100, 2)
        >>> labels = kmeans.fit_predict(X)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        max_iters: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

        self.centers: Optional[torch.Tensor] = None
        self.labels_: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor) -> "KMeans":
        """
        클러스터링 학습

        Args:
            X: (n_samples, n_features) 형태의 데이터

        Returns:
            self
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        n_samples = X.size(0)

        # 무작위 초기화
        indices = torch.randperm(n_samples)[:self.n_clusters]
        self.centers = X[indices].clone()

        for _ in range(self.max_iters):
            # 각 점을 가장 가까운 중심에 할당
            distances = self._compute_distances(X, self.centers)
            self.labels_ = torch.argmin(distances, dim=1)

            # 중심 업데이트
            new_centers = torch.zeros_like(self.centers)
            for k in range(self.n_clusters):
                mask = self.labels_ == k
                if mask.sum() > 0:
                    new_centers[k] = X[mask].mean(dim=0)
                else:
                    new_centers[k] = self.centers[k]

            # 수렴 확인
            shift = (new_centers - self.centers).norm()
            self.centers = new_centers

            if shift < self.tol:
                break

        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        클러스터 레이블 예측

        Args:
            X: (n_samples, n_features) 형태의 데이터

        Returns:
            (n_samples,) 형태의 클러스터 레이블
        """
        if self.centers is None:
            raise ValueError("Model not fitted. Call fit() first.")

        distances = self._compute_distances(X, self.centers)
        return torch.argmin(distances, dim=1)

    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        """학습 후 예측"""
        self.fit(X)
        return self.labels_

    def _compute_distances(
        self,
        X: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        """각 점과 각 중심 간의 거리 계산"""
        # (n_samples, 1, n_features) - (1, n_clusters, n_features)
        # -> (n_samples, n_clusters, n_features)
        diff = X.unsqueeze(1) - centers.unsqueeze(0)
        return (diff ** 2).sum(dim=2)


class KMeansPP(KMeans):
    """
    K-Means++ 클러스터링

    K-Means++는 초기 중심을 더 똑똑하게 선택합니다.
    멀리 떨어진 점을 선택할 확률이 높습니다.

    Example:
        >>> kmeans = KMeansPP(n_clusters=3)
        >>> X = torch.randn(100, 2)
        >>> labels = kmeans.fit_predict(X)
    """

    def fit(self, X: torch.Tensor) -> "KMeansPP":
        """K-Means++ 초기화를 사용한 학습"""
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        n_samples = X.size(0)

        # K-Means++ 초기화
        centers = []

        # 첫 번째 중심: 무작위 선택
        idx = torch.randint(0, n_samples, (1,)).item()
        centers.append(X[idx])

        # 나머지 중심: 거리에 비례한 확률로 선택
        for _ in range(1, self.n_clusters):
            current_centers = torch.stack(centers)
            distances = self._compute_distances(X, current_centers)
            min_distances = distances.min(dim=1).values

            # 확률 계산 (거리의 제곱에 비례)
            probs = min_distances / min_distances.sum()

            # 확률에 따라 다음 중심 선택
            idx = torch.multinomial(probs, 1).item()
            centers.append(X[idx])

        self.centers = torch.stack(centers)

        # 이후 일반 K-Means 반복
        for _ in range(self.max_iters):
            distances = self._compute_distances(X, self.centers)
            self.labels_ = torch.argmin(distances, dim=1)

            new_centers = torch.zeros_like(self.centers)
            for k in range(self.n_clusters):
                mask = self.labels_ == k
                if mask.sum() > 0:
                    new_centers[k] = X[mask].mean(dim=0)
                else:
                    new_centers[k] = self.centers[k]

            shift = (new_centers - self.centers).norm()
            self.centers = new_centers

            if shift < self.tol:
                break

        return self


class SpectralClustering:
    """
    스펙트럴 클러스터링

    그래프 라플라시안의 고유벡터를 사용하여 클러스터링합니다.

    Args:
        n_clusters: 클러스터 수
        gamma: RBF 커널 파라미터
        n_neighbors: k-NN 그래프의 이웃 수 (None이면 완전 그래프)

    Example:
        >>> sc = SpectralClustering(n_clusters=3)
        >>> X = torch.randn(100, 2)
        >>> labels = sc.fit_predict(X)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        gamma: float = 1.0,
        n_neighbors: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.n_neighbors = n_neighbors

        self.labels_: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor) -> "SpectralClustering":
        """스펙트럴 클러스터링 학습"""
        n_samples = X.size(0)

        # 친화도 행렬 계산 (RBF 커널)
        distances = torch.cdist(X, X)
        affinity = torch.exp(-self.gamma * distances ** 2)

        # k-NN 그래프로 제한 (선택적)
        if self.n_neighbors is not None:
            k = self.n_neighbors
            _, indices = torch.topk(affinity, k + 1, dim=1)  # +1은 자기 자신
            mask = torch.zeros_like(affinity, dtype=torch.bool)
            for i in range(n_samples):
                mask[i, indices[i]] = True
            # 대칭화
            mask = mask | mask.T
            affinity = affinity * mask

        # 라플라시안 계산 (정규화된 라플라시안)
        D = affinity.sum(dim=1)
        D_inv_sqrt = 1.0 / torch.sqrt(D + 1e-10)
        D_inv_sqrt = torch.diag(D_inv_sqrt)
        L_sym = torch.eye(n_samples) - D_inv_sqrt @ affinity @ D_inv_sqrt

        # 고유값 분해
        eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)

        # 가장 작은 k개의 고유벡터 선택 (첫 번째는 제외)
        embedding = eigenvectors[:, 1:self.n_clusters + 1]

        # 임베딩 정규화
        embedding = embedding / (embedding.norm(dim=1, keepdim=True) + 1e-10)

        # K-Means로 최종 클러스터링
        kmeans = KMeans(n_clusters=self.n_clusters)
        self.labels_ = kmeans.fit_predict(embedding)

        return self

    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        """학습 후 예측"""
        self.fit(X)
        return self.labels_
