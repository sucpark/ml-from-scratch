"""
Classical Machine Learning Module (mlfs.classical)
===================================================

전통적인 머신러닝 알고리즘을 직접 구현한 패키지입니다.

Modules:
    clustering: KMeans, KMeansPP, SpectralClustering
    reduction: PCA, LLE, Isomap
    ensemble: AdaBoost
"""

from .clustering import KMeans, KMeansPP, SpectralClustering
from .reduction import PCA, LLE, Isomap
from .ensemble import AdaBoost
