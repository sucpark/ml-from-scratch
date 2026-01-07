"""
Probabilistic Models Module (mlfs.probabilistic)
=================================================

확률 모델을 직접 구현한 패키지입니다.

Modules:
    em: EM Algorithm, GMM (Gaussian Mixture Model)
    sparse: Sparse Coding, Dictionary Learning
"""

from .em import EM, GMM
from .sparse import SparseCoding
