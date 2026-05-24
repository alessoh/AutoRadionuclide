"""Molecular featurization for radioligand constructs.

Exports the public API used by surrogates and the policy:
  featurize()         — compute a FeatureRecord for a construct
  tanimoto_distance() — Tanimoto distance between two Morgan fingerprints
  FeatureRecord       — typed result object carrying descriptors + fingerprint
  FeatureQuality      — quality/completeness flag (FULL / PARTIAL / FALLBACK)
"""
from autoradionuclide.featurization._types import FeatureQuality, FeatureRecord
from autoradionuclide.featurization.featurizer import (
    DESCRIPTOR_NAMES,
    FEATURIZER_VERSION,
    FINGERPRINT_NBITS,
    FINGERPRINT_RADIUS,
    featurize,
    tanimoto_distance,
)

__all__ = [
    "featurize",
    "tanimoto_distance",
    "FeatureRecord",
    "FeatureQuality",
    "DESCRIPTOR_NAMES",
    "FEATURIZER_VERSION",
    "FINGERPRINT_NBITS",
    "FINGERPRINT_RADIUS",
]
