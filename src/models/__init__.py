"""
Models subpackage initialization
"""

# Import models - using try/except for robustness
try:
    from .logreg_model import LogisticRegressionModel, TfidfVectorizer
except ImportError as e:
    print(f"Warning: Could not import LogReg model: {e}")
    LogisticRegressionModel = None
    TfidfVectorizer = None

try:
    from .crf_model import CRFModel, CRFFeatureExtractor
except ImportError as e:
    print(f"Warning: Could not import CRF model: {e}")
    CRFModel = None
    CRFFeatureExtractor = None

__all__ = [
    'LogisticRegressionModel',
    'TfidfVectorizer',
    'CRFModel',
    'CRFFeatureExtractor'
]
