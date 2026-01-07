"""
모델 모듈 (mlfs.nn.models)
==========================

MLP, CNN 등의 신경망 모델을 구현합니다.
"""

import torch
from typing import List, Optional

from .layers import Linear, Conv2d, MaxPool2d, Flatten, Dropout, BatchNorm1d
from .activations import relu, softmax


class MLP:
    """
    다층 퍼셉트론 (Multi-Layer Perceptron)

    Args:
        layer_sizes: 각 레이어의 뉴런 수 리스트
                     예: [784, 256, 128, 10]
        dropout: 드롭아웃 확률 (0이면 사용 안 함)

    Example:
        >>> model = MLP([784, 256, 128, 10])
        >>> x = torch.randn(32, 784)
        >>> output = model.forward(x)
        >>> output.shape
        torch.Size([32, 10])

        >>> # 학습
        >>> from mlfs.nn.losses import CrossEntropyLoss
        >>> from mlfs.nn.optim import Adam
        >>> criterion = CrossEntropyLoss()
        >>> optimizer = Adam(model.parameters(), lr=0.001)
        >>> loss = criterion(output, labels)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        layer_sizes: List[int],
        dropout: float = 0.0,
    ):
        self.layer_sizes = layer_sizes
        self.layers = []
        self.dropouts = []
        self.training = True

        # 레이어 생성
        for i in range(len(layer_sizes) - 1):
            layer = Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)

            # 마지막 레이어가 아니면 드롭아웃 추가
            if dropout > 0 and i < len(layer_sizes) - 2:
                self.dropouts.append(Dropout(dropout))
            else:
                self.dropouts.append(None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            x: (batch_size, input_size) 형태의 입력

        Returns:
            (batch_size, output_size) 형태의 출력 (로짓)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # 마지막 레이어가 아니면 ReLU 적용
            if i < len(self.layers) - 1:
                x = relu(x)

                # 드롭아웃 적용
                if self.dropouts[i] is not None and self.training:
                    x = self.dropouts[i](x)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        예측 (클래스 레이블 반환)

        Args:
            x: 입력 텐서

        Returns:
            예측된 클래스 레이블
        """
        self.training = False
        logits = self.forward(x)
        self.training = True
        return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        확률 예측

        Args:
            x: 입력 텐서

        Returns:
            각 클래스에 대한 확률
        """
        self.training = False
        logits = self.forward(x)
        self.training = True
        return softmax(logits, dim=1)

    def parameters(self) -> List[torch.Tensor]:
        """학습 가능한 파라미터 반환"""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def train(self):
        """훈련 모드 설정"""
        self.training = True
        for dropout in self.dropouts:
            if dropout is not None:
                dropout.train()

    def eval(self):
        """평가 모드 설정"""
        self.training = False
        for dropout in self.dropouts:
            if dropout is not None:
                dropout.eval()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"MLP(layer_sizes={self.layer_sizes})"


class CNN:
    """
    합성곱 신경망 (Convolutional Neural Network)

    MNIST, CIFAR-10 등의 이미지 분류에 사용됩니다.

    Args:
        in_channels: 입력 채널 수 (MNIST: 1, CIFAR: 3)
        num_classes: 출력 클래스 수
        image_size: 입력 이미지 크기 (정사각형 가정)

    Example:
        >>> # MNIST용 CNN
        >>> model = CNN(in_channels=1, num_classes=10, image_size=28)
        >>> x = torch.randn(32, 1, 28, 28)
        >>> output = model.forward(x)
        >>> output.shape
        torch.Size([32, 10])

        >>> # CIFAR-10용 CNN
        >>> model = CNN(in_channels=3, num_classes=10, image_size=32)
        >>> x = torch.randn(32, 3, 32, 32)
        >>> output = model.forward(x)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        image_size: int = 28,
    ):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.training = True

        # Conv 레이어 정의
        self.conv1 = Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = Conv2d(64, 128, kernel_size=3, padding=1)

        # 풀링
        self.pool = MaxPool2d(kernel_size=2, stride=2)

        # Flatten
        self.flatten = Flatten()

        # FC 레이어 크기 계산
        # image_size -> pool -> pool -> pool
        final_size = image_size // 8  # 3번의 2x2 풀링
        fc_input_size = 128 * final_size * final_size

        # FC 레이어
        self.fc1 = Linear(fc_input_size, 256)
        self.fc2 = Linear(256, num_classes)

        # 드롭아웃
        self.dropout = Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            x: (batch, channels, H, W) 형태의 입력

        Returns:
            (batch, num_classes) 형태의 로짓
        """
        # Conv Block 1
        x = self.conv1(x)
        x = relu(x)
        x = self.pool(x)

        # Conv Block 2
        x = self.conv2(x)
        x = relu(x)
        x = self.pool(x)

        # Conv Block 3
        x = self.conv3(x)
        x = relu(x)
        x = self.pool(x)

        # Flatten
        x = self.flatten(x)

        # FC layers
        x = self.fc1(x)
        x = relu(x)

        if self.training:
            x = self.dropout(x)

        x = self.fc2(x)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """예측"""
        self.training = False
        logits = self.forward(x)
        self.training = True
        return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """확률 예측"""
        self.training = False
        logits = self.forward(x)
        self.training = True
        return softmax(logits, dim=1)

    def parameters(self) -> List[torch.Tensor]:
        """학습 가능한 파라미터 반환"""
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.conv3.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params

    def train(self):
        """훈련 모드"""
        self.training = True
        self.dropout.train()

    def eval(self):
        """평가 모드"""
        self.training = False
        self.dropout.eval()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"CNN(in_channels={self.in_channels}, num_classes={self.num_classes})"


class Perceptron:
    """
    퍼셉트론 (단일 뉴런)

    이진 분류를 위한 가장 간단한 신경망입니다.

    Args:
        input_size: 입력 특성 수

    Example:
        >>> model = Perceptron(input_size=2)
        >>> x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> output = model.forward(x)
    """

    def __init__(self, input_size: int):
        self.input_size = input_size

        # 가중치 초기화
        self.weight = torch.randn(input_size) * 0.01
        self.weight.requires_grad_(True)

        self.bias = torch.zeros(1)
        self.bias.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            x: (batch_size, input_size) 형태의 입력

        Returns:
            (batch_size,) 형태의 출력 (로짓)
        """
        return x @ self.weight + self.bias

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        예측 (0 또는 1)

        step function 적용
        """
        output = self.forward(x)
        return (output > 0).long()

    def parameters(self) -> List[torch.Tensor]:
        return [self.weight, self.bias]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"Perceptron(input_size={self.input_size})"
