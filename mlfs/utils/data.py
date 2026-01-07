"""
데이터 로딩 유틸리티 (mlfs.utils.data)
======================================

MNIST, CIFAR-10 등의 데이터셋을 쉽게 로드할 수 있는 함수들입니다.
데이터가 없으면 자동으로 다운로드합니다.
"""

from pathlib import Path
from typing import Tuple, Optional

import torch
from torchvision import datasets, transforms


# 데이터 저장 경로 (프로젝트 루트/data/)
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def load_mnist(
    train: bool = True,
    flatten: bool = False,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MNIST 데이터셋을 로드합니다. 없으면 자동 다운로드합니다.

    Args:
        train: True면 훈련셋(60,000), False면 테스트셋(10,000)
        flatten: True면 (N, 784) 형태로, False면 (N, 1, 28, 28) 형태로 반환
        normalize: True면 [0, 1] 범위로 정규화

    Returns:
        images: 이미지 텐서
        labels: 레이블 텐서 (0-9)

    Example:
        >>> X_train, y_train = load_mnist(train=True)
        >>> X_test, y_test = load_mnist(train=False)
        >>> print(f"훈련셋: {X_train.shape}, 테스트셋: {X_test.shape}")
        훈련셋: torch.Size([60000, 1, 28, 28]), 테스트셋: torch.Size([10000, 1, 28, 28])

        >>> X_flat, y = load_mnist(train=True, flatten=True)
        >>> print(f"Flatten: {X_flat.shape}")
        Flatten: torch.Size([60000, 784])
    """
    # 변환 정의
    transform_list = [transforms.ToTensor()]
    if not normalize:
        # ToTensor는 기본적으로 [0, 1]로 정규화하므로 255를 곱해서 원복
        pass  # 일단 정규화된 상태로 유지

    transform = transforms.Compose(transform_list)

    # 데이터셋 로드 (없으면 자동 다운로드)
    dataset = datasets.MNIST(
        root=DATA_DIR,
        train=train,
        download=True,
        transform=transform,
    )

    # 텐서로 변환
    images = torch.stack([img for img, _ in dataset])
    labels = torch.tensor([label for _, label in dataset])

    # Flatten 옵션
    if flatten:
        images = images.view(images.size(0), -1)

    return images, labels


def load_cifar10(
    train: bool = True,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CIFAR-10 데이터셋을 로드합니다. 없으면 자동 다운로드합니다.

    Args:
        train: True면 훈련셋(50,000), False면 테스트셋(10,000)
        normalize: True면 [0, 1] 범위로 정규화

    Returns:
        images: (N, 3, 32, 32) 형태의 이미지 텐서
        labels: (N,) 형태의 레이블 텐서 (0-9)

    Example:
        >>> X_train, y_train = load_cifar10(train=True)
        >>> X_test, y_test = load_cifar10(train=False)
        >>> print(f"훈련셋: {X_train.shape}")
        훈련셋: torch.Size([50000, 3, 32, 32])

    Note:
        CIFAR-10 클래스:
        0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer,
        5: dog, 6: frog, 7: horse, 8: ship, 9: truck
    """
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=train,
        download=True,
        transform=transform,
    )

    images = torch.stack([img for img, _ in dataset])
    labels = torch.tensor([label for _, label in dataset])

    return images, labels


def load_mnist_subset(
    n_samples: int = 1000,
    train: bool = True,
    flatten: bool = False,
    seed: Optional[int] = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MNIST 데이터셋의 일부만 로드합니다. 빠른 실험에 유용합니다.

    Args:
        n_samples: 로드할 샘플 수
        train: True면 훈련셋에서, False면 테스트셋에서 샘플링
        flatten: True면 (N, 784) 형태로 반환
        seed: 랜덤 시드 (재현성을 위해)

    Returns:
        images: 이미지 텐서
        labels: 레이블 텐서

    Example:
        >>> X, y = load_mnist_subset(n_samples=100)
        >>> print(f"샘플 수: {len(X)}")
        샘플 수: 100
    """
    images, labels = load_mnist(train=train, flatten=flatten)

    if seed is not None:
        torch.manual_seed(seed)

    n_samples = min(n_samples, len(images))
    indices = torch.randperm(len(images))[:n_samples]

    return images[indices], labels[indices]


# CIFAR-10 클래스 이름
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
