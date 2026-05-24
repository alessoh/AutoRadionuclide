"""Shared types for the featurization package."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class FeatureQuality(str, Enum):
    """Quality flag for a feature record.

    FULL    — every organic part of the construct resolved to a chemical structure.
    PARTIAL — some parts resolved (e.g. chelator present in registry, targeting
              vector structure unknown). Descriptors reflect the resolved portions only.
    FALLBACK — no organic structure could be resolved. Descriptor vector and
               fingerprint are explicitly zero and must not be used for regression.
    """

    FULL = "full"
    PARTIAL = "partial"
    FALLBACK = "fallback"


@dataclass
class FeatureRecord:
    """Feature record for one candidate construct.

    Carries two representations for two distinct purposes:
    - ``descriptor_vector``: compact physicochemical descriptors for GP surrogate
      regression. Small set intentional — GPs fit on few observations and a
      high-dimensional vector would overfit.
    - ``fingerprint``: Morgan fingerprint bit array for Tanimoto-based diversity
      selection in the policy.
    - ``isotope_features``: factual physics features (atomic number, half-life,
      decay-mode encoding) sourced from domain data, not from cheminformatics.

    LIMITS (stated here so they cannot be removed without touching this class):
      Metal coordination chemistry is NOT modeled. The metal-organic bond and
      coordination geometry are not represented in these features.
      Radiation effects (particle energy, LET, bond-breaking) are NOT captured.
      Large-peptide 3D conformation is NOT represented by 2D descriptors.
    """

    construct_id: str
    descriptor_vector: np.ndarray     # shape (N_DESCRIPTORS,), raw (pre-scaler)
    fingerprint: np.ndarray           # shape (FINGERPRINT_NBITS,), dtype uint8
    isotope_features: np.ndarray      # shape (3,): [atomic_number, half_life_days, decay_mode]
    quality: FeatureQuality
    unresolved_parts: list[str]
    resolution_reasons: dict[str, str]
    featurizer_version: str
    rdkit_version: str
    descriptor_names: list[str]
    fingerprint_params: dict
    provenance_tag: str = "computed_structural_features_metal_coordination_not_modeled"
