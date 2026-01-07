"""
시각화 유틸리티 (mlfs.utils.viz)
================================

학습 과정 및 결과를 시각화하는 함수들입니다.
"""

from typing import Optional, List, Union, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_images(
    images: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    predictions: Optional[torch.Tensor] = None,
    n_rows: int = 2,
    n_cols: int = 5,
    figsize: Tuple[int, int] = (12, 5),
    title: Optional[str] = None,
) -> None:
    """
    이미지 그리드를 시각화합니다.

    Args:
        images: (N, C, H, W) 또는 (N, H, W) 형태의 이미지 텐서
        labels: 실제 레이블 (선택)
        predictions: 예측 레이블 (선택)
        n_rows: 행 수
        n_cols: 열 수
        figsize: 그림 크기
        title: 전체 제목

    Example:
        >>> from mlfs.utils.data import load_mnist
        >>> X, y = load_mnist(train=False)
        >>> plot_images(X[:10], labels=y[:10])
    """
    n_images = min(n_rows * n_cols, len(images))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i in range(n_images):
        img = images[i]

        # (C, H, W) -> (H, W, C) 또는 (H, W)
        if img.dim() == 3:
            if img.shape[0] == 1:  # 흑백
                img = img.squeeze(0)
            else:  # 컬러
                img = img.permute(1, 2, 0)

        # numpy로 변환
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        # 시각화
        if img.ndim == 2:  # 흑백
            axes[i].imshow(img, cmap="gray")
        else:  # 컬러
            axes[i].imshow(img)

        axes[i].axis("off")

        # 레이블 표시
        if labels is not None:
            label_text = str(int(labels[i]))
            if predictions is not None:
                pred = int(predictions[i])
                color = "green" if pred == int(labels[i]) else "red"
                axes[i].set_title(f"T:{label_text} P:{pred}", color=color)
            else:
                axes[i].set_title(f"Label: {label_text}")

    # 남은 axes 숨기기
    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_loss(
    losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training Loss",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    figsize: Tuple[int, int] = (10, 5),
) -> None:
    """
    학습 손실 곡선을 시각화합니다.

    Args:
        losses: 훈련 손실 리스트
        val_losses: 검증 손실 리스트 (선택)
        title: 그래프 제목
        xlabel: x축 레이블
        ylabel: y축 레이블
        figsize: 그림 크기

    Example:
        >>> losses = [1.0, 0.8, 0.6, 0.4, 0.3]
        >>> val_losses = [1.1, 0.9, 0.7, 0.5, 0.45]
        >>> plot_loss(losses, val_losses)
    """
    plt.figure(figsize=figsize)

    plt.plot(losses, label="Train Loss", color="blue", linewidth=2)

    if val_losses is not None:
        plt.plot(val_losses, label="Val Loss", color="orange", linewidth=2)
        plt.legend()

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_accuracy(
    accuracies: List[float],
    val_accuracies: Optional[List[float]] = None,
    title: str = "Training Accuracy",
    figsize: Tuple[int, int] = (10, 5),
) -> None:
    """
    정확도 곡선을 시각화합니다.

    Args:
        accuracies: 훈련 정확도 리스트
        val_accuracies: 검증 정확도 리스트 (선택)
        title: 그래프 제목
        figsize: 그림 크기
    """
    plt.figure(figsize=figsize)

    plt.plot(accuracies, label="Train Acc", color="blue", linewidth=2)

    if val_accuracies is not None:
        plt.plot(val_accuracies, label="Val Acc", color="orange", linewidth=2)
        plt.legend()

    plt.title(title, fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_clusters(
    X: torch.Tensor,
    labels: torch.Tensor,
    centers: Optional[torch.Tensor] = None,
    title: str = "Clustering Result",
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    2D 클러스터링 결과를 시각화합니다.

    Args:
        X: (N, 2) 형태의 데이터 포인트
        labels: (N,) 형태의 클러스터 레이블
        centers: (K, 2) 형태의 클러스터 중심 (선택)
        title: 그래프 제목
        figsize: 그림 크기

    Example:
        >>> X = torch.randn(100, 2)
        >>> labels = torch.randint(0, 3, (100,))
        >>> plot_clusters(X, labels)
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    plt.figure(figsize=figsize)

    # 고유한 레이블
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            X[mask, 0], X[mask, 1],
            c=[colors[i]],
            label=f"Cluster {label}",
            alpha=0.7,
            s=50,
        )

    # 클러스터 중심 표시
    if centers is not None:
        if isinstance(centers, torch.Tensor):
            centers = centers.detach().cpu().numpy()
        plt.scatter(
            centers[:, 0], centers[:, 1],
            c="black",
            marker="X",
            s=200,
            edgecolors="white",
            linewidths=2,
            label="Centers",
        )

    plt.title(title, fontsize=14)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_decision_boundary(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    resolution: int = 100,
    title: str = "Decision Boundary",
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    2D 분류 모델의 결정 경계를 시각화합니다.

    Args:
        model: predict(X) 메서드가 있는 모델
        X: (N, 2) 형태의 입력 데이터
        y: (N,) 형태의 레이블
        resolution: 그리드 해상도
        title: 그래프 제목
        figsize: 그림 크기
    """
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = X

    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    else:
        y_np = y

    # 그리드 생성
    x_min, x_max = X_np[:, 0].min() - 1, X_np[:, 0].max() + 1
    y_min, y_max = X_np[:, 1].min() - 1, X_np[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # 그리드 포인트 예측
    grid_points = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()],
        dtype=torch.float32
    )
    Z = model.predict(grid_points)
    if isinstance(Z, torch.Tensor):
        Z = Z.detach().cpu().numpy()
    Z = Z.reshape(xx.shape)

    # 시각화
    plt.figure(figsize=figsize)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap="viridis")
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap="viridis", edgecolors="black")
    plt.title(title, fontsize=14)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.colorbar(label="Class")
    plt.show()


def plot_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 8),
) -> None:
    """
    혼동 행렬을 시각화합니다.

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        class_names: 클래스 이름 리스트
        title: 그래프 제목
        figsize: 그림 크기
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    n_classes = max(y_true.max(), y_pred.max()) + 1

    # 혼동 행렬 계산
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title, fontsize=14)
    plt.colorbar()

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # 값 표시
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()
