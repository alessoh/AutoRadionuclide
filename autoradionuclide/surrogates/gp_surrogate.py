"""Gaussian Process surrogate models — one per objective, refitted as results arrive."""
from __future__ import annotations
import numpy as np
from typing import Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from autoradionuclide.domain.models import CandidateConstruct, ObjectiveValue, ProvenanceTag


# Feature encoding constants
_KNOWN_TARGETS = ["PSMA", "SSTR2", "FAP", "integrin_avb3", "NET", "VEGFR", "unknown"]
_KNOWN_CHELATORS = ["DOTA", "NOTA", "DOTAGA", "PSMA", "other"]
_KNOWN_ISOTOPES = ["Lu-177", "Ac-225", "Ga-68", "Y-90", "I-131", "Bi-213", "At-211"]
_KNOWN_VTYPES = ["small_molecule", "peptide", "antibody_fragment"]


def featurize(construct: CandidateConstruct) -> np.ndarray:
    """Return a fixed-length feature vector for a construct.

    Uses one-hot encoding of categorical components plus binary flags.
    This is a simplified representation; a real system would use
    molecular fingerprints from the full SMILES.
    """
    target = construct.targeting_vector.target
    chelator = construct.chelator.name
    isotope = construct.radionuclide.value
    vtype = construct.targeting_vector.vector_type
    has_linker = 1.0 if construct.linker else 0.0

    t_enc = _one_hot(target, _KNOWN_TARGETS)
    c_enc = _one_hot(chelator, _KNOWN_CHELATORS)
    i_enc = _one_hot(isotope, _KNOWN_ISOTOPES)
    v_enc = _one_hot(vtype, _KNOWN_VTYPES)
    return np.concatenate([t_enc, c_enc, i_enc, v_enc, [has_linker]])


def _one_hot(val: str, options: list[str]) -> np.ndarray:
    vec = np.zeros(len(options))
    idx = options.index(val) if val in options else len(options) - 1
    vec[idx] = 1.0
    return vec


class ObjectiveSurrogate:
    """GP surrogate for a single objective.

    Stores all observed (feature, value) pairs and refits the GP
    whenever update() is called. Returns (mean, std) for any construct.
    """

    def __init__(self, objective_name: str, seed: int = 42) -> None:
        self.objective_name = objective_name
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._scaler = StandardScaler()
        self._gp: Optional[GaussianProcessRegressor] = None
        self._seed = seed
        self._fitted = False

    def update(self, constructs: list[CandidateConstruct], values: list[float]) -> None:
        """Add observations and refit the GP."""
        for c, v in zip(constructs, values):
            self._X.append(featurize(c))
            self._y.append(v)
        if len(self._X) >= 2:
            X = np.array(self._X)
            y = np.array(self._y)
            X_scaled = self._scaler.fit_transform(X)
            kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=0.01)
            self._gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=3,
                random_state=self._seed,
                normalize_y=True,
            )
            self._gp.fit(X_scaled, y)
            self._fitted = True

    def predict(self, construct: CandidateConstruct) -> ObjectiveValue:
        """Return predicted mean ± std for a candidate."""
        if not self._fitted:
            # Prior: return heuristic from frozen harness
            import frozen.harness as h
            scores = h.score_all(construct)
            if self.objective_name in scores:
                return scores[self.objective_name]
            return ObjectiveValue(estimate=0.5, uncertainty=0.3, source=ProvenanceTag.HEURISTIC)

        feat = featurize(construct).reshape(1, -1)
        feat_scaled = self._scaler.transform(feat)
        mean, std = self._gp.predict(feat_scaled, return_std=True)
        return ObjectiveValue(
            estimate=float(np.clip(mean[0], 0, 1)),
            uncertainty=float(std[0]),
            source=ProvenanceTag.LEARNED,
        )

    @property
    def n_observations(self) -> int:
        return len(self._y)


class SurrogateBank:
    """One surrogate per objective. The policy queries this bank."""

    def __init__(self, objective_names: list[str], seed: int = 42) -> None:
        self._surrogates = {
            name: ObjectiveSurrogate(name, seed) for name in objective_names
        }

    def update(
        self,
        constructs: list[CandidateConstruct],
        objective_values: dict[str, list[float]],
    ) -> None:
        for name, surrogate in self._surrogates.items():
            if name in objective_values:
                surrogate.update(constructs, objective_values[name])

    def predict_all(self, construct: CandidateConstruct) -> dict[str, ObjectiveValue]:
        return {name: s.predict(construct) for name, s in self._surrogates.items()}

    def get(self, objective_name: str) -> ObjectiveSurrogate:
        return self._surrogates[objective_name]
