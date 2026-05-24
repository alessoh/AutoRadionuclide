"""Gaussian Process surrogate models — one per objective, refitted as results arrive.

Featurization: the surrogates fit on the standardized descriptor vector from
``autoradionuclide.featurization``. This is a compact set of 8 standard RDKit
physicochemical descriptors (MW, logP, TPSA, HBD, HBA, RotBonds, Rings, FracCSP3).

Degraded-record handling:
  FALLBACK quality records (no organic structure resolved) are excluded from the
  GP fit. Their presence emits a warning (raised by the featurizer). Predictions
  for FALLBACK constructs fall back to the heuristic prior from the frozen harness
  rather than using the GP.

  PARTIAL quality records (some parts resolved) are included in the fit. Their
  descriptors reflect only the resolved portion, which is noted in the provenance
  but does not raise an additional warning here.

Note: fitting on 8 real descriptors from a handful of observations may still
produce the GP convergence warnings seen with the previous one-hot representation.
The limiting factor is the tiny amount of labelled data, not the representation.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from autoradionuclide.domain.models import CandidateConstruct, ObjectiveValue, ProvenanceTag
from autoradionuclide.featurization import FeatureQuality, featurize


class ObjectiveSurrogate:
    """GP surrogate for a single objective.

    Stores all observed (descriptor_vector, value) pairs and refits the GP
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
        """Add observations and refit the GP.

        FALLBACK-quality records are excluded: their descriptor vectors are
        explicit zeros that do not represent a real chemical structure, so
        including them would corrupt the training set.
        """
        for c, v in zip(constructs, values):
            record = featurize(c)
            if record.quality is FeatureQuality.FALLBACK:
                # Warning already emitted by featurize(); nothing to add here.
                continue
            self._X.append(record.descriptor_vector.copy())
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
        """Return predicted mean ± std for a candidate.

        FALLBACK constructs cannot be featurized and fall back to the heuristic
        prior from the frozen harness, regardless of whether the GP is fitted.
        """
        record = featurize(construct)

        if not self._fitted or record.quality is FeatureQuality.FALLBACK:
            import frozen.harness as h
            scores = h.score_all(construct)
            if self.objective_name in scores:
                return scores[self.objective_name]
            return ObjectiveValue(
                estimate=0.5, uncertainty=0.3, source=ProvenanceTag.HEURISTIC
            )

        feat_scaled = self._scaler.transform(record.descriptor_vector.reshape(1, -1))
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
