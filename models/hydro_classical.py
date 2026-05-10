"""
HydroClassical — Classical classification heads for CNN feature vectors.

Provides a uniform fit / predict / predict_proba interface over:
  - RVM  (skrvm — Relevance Vector Classifier)
  - SVM  (sklearn)
  - XGBoost
  - LightGBM
  - AdaBoost

All heads are returned by build_classical_heads() as a dict so
training/eval_classical.py can iterate over them identically.

Missing optional packages produce a RuntimeWarning (not an error) so
the comparison run still completes with the available heads.
"""

import logging
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import AdaBoostClassifier

# GPU-accelerated SVM via RAPIDS cuML when available; sklearn on CPU otherwise.
try:
    from cuml.svm import SVC  # type: ignore
    _SVC_BACKEND = "cuml"
except ImportError:
    from sklearn.svm import SVC  # type: ignore
    _SVC_BACKEND = "sklearn"

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Uniform wrapper
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassicalHead:
    """
    Thin wrapper giving any sklearn-compatible estimator a named interface.

    Attributes:
        name      (str)   : Display name used in comparison tables and logs.
        estimator (object): sklearn-compatible object with fit/predict.
    """
    name: str
    estimator: object

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalHead":
        logger.info("Fitting %s on %d samples (dim=%d) ...", self.name, X.shape[0], X.shape[1])
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probability matrix (N, C).

        Falls back to a softmax over decision_function scores when
        predict_proba is unavailable (e.g. some AdaBoost configurations).
        """
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        df = self.estimator.decision_function(X)
        if df.ndim == 1:
            # binary fallback: sigmoid
            p = 1.0 / (1.0 + np.exp(-df))
            return np.stack([1.0 - p, p], axis=1)
        # multiclass softmax
        exp = np.exp(df - df.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_classical_heads(random_state: int = 42) -> dict[str, ClassicalHead]:
    """
    Build all classical heads with consistent hyperparameters.

    Args:
        random_state (int): Shared seed for reproducibility.

    Returns:
        Ordered dict of {name: ClassicalHead}.  Missing packages are warned
        and skipped — the returned dict contains only available heads.
    """
    heads: dict[str, ClassicalHead] = {}

    # ── RVM ───────────────────────────────────────────────────────────────────
    # Uses sklearn_rvm.EMRVC (Expectation-Maximisation Relevance Vector
    # Classifier). Can be slow on large datasets (>10k samples); consider
    # passing --tsne_max_samples or sub-sampling X_train if needed.
    try:
        from sklearn_rvm import EMRVC  # type: ignore
        heads["RVM"] = ClassicalHead(
            name="RVM",
            estimator=EMRVC(kernel="rbf", gamma="scale"),
        )
    except ImportError:
        warnings.warn(
            "sklearn_rvm not installed — RVM head skipped. "
            "Install with: pip install sklearn-rvm",
            RuntimeWarning,
            stacklevel=2,
        )

    # ── SVM ───────────────────────────────────────────────────────────────────
    logger.info("SVM backend: %s", _SVC_BACKEND)
    heads["SVM"] = ClassicalHead(
        name="SVM",
        estimator=SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
    )

    # ── XGBoost ───────────────────────────────────────────────────────────────
    try:
        from xgboost import XGBClassifier  # type: ignore
        heads["XGBoost"] = ClassicalHead(
            name="XGBoost",
            estimator=XGBClassifier(
                n_estimators=300,
                max_depth=6,
                tree_method="hist",
                eval_metric="mlogloss",
                random_state=random_state,
                verbosity=0,
            ),
        )
    except ImportError:
        warnings.warn(
            "xgboost not installed — XGBoost head skipped. "
            "Install with: pip install xgboost",
            RuntimeWarning,
            stacklevel=2,
        )

    # ── LightGBM ──────────────────────────────────────────────────────────────
    try:
        import lightgbm as lgb  # type: ignore
        heads["LightGBM"] = ClassicalHead(
            name="LightGBM",
            estimator=lgb.LGBMClassifier(
                n_estimators=300,
                num_leaves=63,
                class_weight="balanced",
                random_state=random_state,
                verbose=-1,
            ),
        )
    except ImportError:
        warnings.warn(
            "lightgbm not installed — LightGBM head skipped. "
            "Install with: pip install lightgbm",
            RuntimeWarning,
            stacklevel=2,
        )

    # ── AdaBoost ──────────────────────────────────────────────────────────────
    # sklearn >= 1.6 removed the `algorithm` parameter (SAMME is the only
    # algorithm left). Detect via signature so this works across versions.
    import inspect
    ada_kwargs: dict = {"n_estimators": 200, "random_state": random_state}
    if "algorithm" in inspect.signature(AdaBoostClassifier.__init__).parameters:
        ada_kwargs["algorithm"] = "SAMME"
    heads["AdaBoost"] = ClassicalHead(
        name="AdaBoost",
        estimator=AdaBoostClassifier(**ada_kwargs),
    )

    logger.info("Built classical heads: %s", list(heads.keys()))
    return heads
