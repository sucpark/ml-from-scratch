"""
차원 축소 모듈 (mlfs.classical.reduction)
==========================================

PCA, LLE, Isomap 등을 직접 구현합니다.
"""

import torch
from typing import Optional


class PCA:
    """
    주성분 분석 (Principal Component Analysis)

    Args:
        n_components: 유지할 주성분 수

    Example:
        >>> pca = PCA(n_components=2)
        >>> X = torch.randn(100, 50)
        >>> X_reduced = pca.fit_transform(X)
        >>> X_reduced.shape
        torch.Size([100, 2])
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_: Optional[torch.Tensor] = None
        self.mean_: Optional[torch.Tensor] = None
        self.explained_variance_: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor) -> "PCA":
        """
        PCA 학습

        Args:
            X: (n_samples, n_features) 형태의 데이터

        Returns:
            self
        """
        # 평균 중심화
        self.mean_ = X.mean(dim=0)
        X_centered = X - self.mean_

        # 공분산 행렬 계산
        n_samples = X.size(0)
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

        # 고유값 분해
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

        # 내림차순 정렬
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # 상위 n_components개 선택
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]

        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        데이터를 주성분 공간으로 변환

        Args:
            X: (n_samples, n_features) 형태의 데이터

        Returns:
            (n_samples, n_components) 형태의 변환된 데이터
        """
        if self.components_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """학습 후 변환"""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced: torch.Tensor) -> torch.Tensor:
        """변환된 데이터를 원래 공간으로 복원"""
        if self.components_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return X_reduced @ self.components_ + self.mean_


class LLE:
    """
    지역 선형 임베딩 (Locally Linear Embedding)

    Args:
        n_components: 출력 차원
        n_neighbors: 이웃 수

    Example:
        >>> lle = LLE(n_components=2, n_neighbors=10)
        >>> X = torch.randn(100, 50)
        >>> X_reduced = lle.fit_transform(X)
    """

    def __init__(self, n_components: int = 2, n_neighbors: int = 10):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.embedding_: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor) -> "LLE":
        """LLE 학습"""
        n_samples, n_features = X.shape
        k = self.n_neighbors

        # 1. k-NN 찾기
        distances = torch.cdist(X, X)
        distances.fill_diagonal_(float('inf'))
        _, neighbors = torch.topk(distances, k, largest=False, dim=1)

        # 2. 재구성 가중치 계산
        W = torch.zeros(n_samples, n_samples)

        for i in range(n_samples):
            # 이웃 데이터
            Xi_neighbors = X[neighbors[i]]  # (k, n_features)
            Xi = X[i]  # (n_features,)

            # 중심화
            Z = Xi_neighbors - Xi  # (k, n_features)

            # 지역 공분산 행렬
            C = Z @ Z.T  # (k, k)

            # 정규화 (수치 안정성)
            C = C + torch.eye(k) * 1e-3 * C.trace()

            # 가중치 계산: C * w = 1, sum(w) = 1
            w = torch.linalg.solve(C, torch.ones(k))
            w = w / w.sum()

            W[i, neighbors[i]] = w

        # 3. 임베딩 계산
        M = (torch.eye(n_samples) - W).T @ (torch.eye(n_samples) - W)

        # 고유값 분해
        eigenvalues, eigenvectors = torch.linalg.eigh(M)

        # 가장 작은 고유값의 고유벡터 선택 (첫 번째는 상수벡터이므로 제외)
        self.embedding_ = eigenvectors[:, 1:self.n_components + 1]

        return self

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """학습 후 변환"""
        self.fit(X)
        return self.embedding_


class Isomap:
    """
    Isomap (Isometric Feature Mapping)

    측지선 거리를 사용한 차원 축소입니다.

    Args:
        n_components: 출력 차원
        n_neighbors: 이웃 수

    Example:
        >>> isomap = Isomap(n_components=2, n_neighbors=10)
        >>> X = torch.randn(100, 50)
        >>> X_reduced = isomap.fit_transform(X)
    """

    def __init__(self, n_components: int = 2, n_neighbors: int = 10):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.embedding_: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor) -> "Isomap":
        """Isomap 학습"""
        n_samples = X.size(0)
        k = self.n_neighbors

        # 1. k-NN 그래프 구성
        distances = torch.cdist(X, X)
        _, neighbors = torch.topk(distances, k + 1, largest=False, dim=1)

        # 그래프 거리 행렬 초기화
        graph_dist = torch.full((n_samples, n_samples), float('inf'))
        graph_dist.fill_diagonal_(0)

        for i in range(n_samples):
            for j in neighbors[i]:
                j = j.item()
                if i != j:
                    graph_dist[i, j] = distances[i, j]
                    graph_dist[j, i] = distances[i, j]

        # 2. 최단 경로 (Floyd-Warshall)
        for k_node in range(n_samples):
            for i in range(n_samples):
                for j in range(n_samples):
                    if graph_dist[i, k_node] + graph_dist[k_node, j] < graph_dist[i, j]:
                        graph_dist[i, j] = graph_dist[i, k_node] + graph_dist[k_node, j]

        # 3. MDS (Multi-Dimensional Scaling)
        # 중심화된 내적 행렬 계산
        D_sq = graph_dist ** 2
        n = n_samples
        H = torch.eye(n) - torch.ones(n, n) / n
        B = -0.5 * H @ D_sq @ H

        # 고유값 분해
        eigenvalues, eigenvectors = torch.linalg.eigh(B)

        # 가장 큰 양의 고유값 선택
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # 임베딩 계산
        eigenvalues = torch.clamp(eigenvalues[:self.n_components], min=0)
        self.embedding_ = eigenvectors[:, :self.n_components] * torch.sqrt(eigenvalues)

        return self

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """학습 후 변환"""
        self.fit(X)
        return self.embedding_
