"""
ML From Scratch (mlfs)
======================

PyTorch의 autograd만 활용하여 머신러닝 알고리즘을 밑바닥부터 직접 구현한 패키지입니다.

Subpackages:
    nn: 신경망 관련 모듈 (layers, activations, losses, optimizers)
    classical: 전통 ML 알고리즘 (clustering, reduction, ensemble)
    probabilistic: 확률 모델 (EM, GMM, Sparse Coding)
    utils: 유틸리티 (data loading, visualization)
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from . import nn
from . import classical
from . import probabilistic
from . import utils
