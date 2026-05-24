"""
Molecular featurization for radioligand constructs.

IMPORTANT LIMITS — stated here so they cannot be removed without touching this module:

  1. Metal coordination chemistry is NOT modeled. The metal-organic bond between
     the radionuclide and the chelator is not represented in these features.
     Coordination geometry, thermodynamic stability, and kinetic inertness are
     not captured by 2D organic-molecule descriptors.

  2. Radiation effects are NOT captured. The energy and type of emitted particles
     (β⁻, α, γ, Auger), the linear energy transfer (LET), and the capacity to
     break chemical bonds or damage DNA are not represented.

  3. Large-peptide 3D conformation is NOT represented. Standard 2D physicochemical
     descriptors and Morgan fingerprints were designed for drug-like small molecules.
     They do not capture backbone geometry, secondary structure, or the spatial
     arrangement of a large peptide targeting vector.

These features are designed for two narrow, honest purposes:
  - GP surrogate regression: a compact, fixed-length descriptor vector for
    Gaussian-process fitting on a small number of observations.
  - Structural diversity selection: a Morgan fingerprint for Tanimoto-based
    deduplication of proposed batches.

The descriptor set is intentionally small (8 features). Fitting a GP on very few
observations with a high-dimensional representation would overfit and be useless.

Featurization is deterministic and versioned; the FEATURIZER_VERSION constant must
be incremented whenever the descriptor set, fingerprint parameters, or resolution
logic changes.
"""
from __future__ import annotations

import warnings

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from autoradionuclide.domain.models import HALF_LIFE_DAYS, CandidateConstruct
from autoradionuclide.featurization._types import FeatureQuality, FeatureRecord
from autoradionuclide.featurization.isotope_data import ISOTOPE_PHYSICS
from autoradionuclide.featurization.registry import resolve_organic_smiles


FEATURIZER_VERSION = "1.0.0"

# Ordered descriptor names — must match the computation in _compute_descriptors.
# Do not reorder without incrementing FEATURIZER_VERSION.
DESCRIPTOR_NAMES: list[str] = [
    "mw",         # molecular weight (Da)
    "logp",       # Wildman-Crippen logP
    "tpsa",       # topological polar surface area (Å²)
    "hbd",        # hydrogen-bond donor count
    "hba",        # hydrogen-bond acceptor count
    "rotbonds",   # rotatable bond count
    "rings",      # ring count
    "frac_csp3",  # fraction of sp3 carbons
]

FINGERPRINT_RADIUS: int = 2
FINGERPRINT_NBITS: int = 2048


def featurize(construct: CandidateConstruct) -> FeatureRecord:
    """Compute the feature record for a candidate construct.

    Resolves the organic structure of the construct (chelator + targeting vector),
    computes 2D physicochemical descriptors and a Morgan fingerprint for the
    resolved organic portion, and computes isotope physics features from factual
    domain data.

    When no organic structure can be resolved, emits a warning, sets quality to
    FALLBACK, and returns an explicit zero vector and zero fingerprint — it does
    NOT fabricate descriptor values.
    """
    smiles, quality, unresolved, reasons = resolve_organic_smiles(construct)

    if quality is FeatureQuality.FALLBACK:
        warnings.warn(
            f"Construct '{construct.name}' ({construct.id}): no organic structure "
            f"resolved. Feature record quality=FALLBACK. Unresolved: {unresolved}. "
            "Descriptor vector and fingerprint are explicit zeros and must not be "
            "used for GP regression.",
            UserWarning,
            stacklevel=2,
        )
        descriptor_vector = np.zeros(len(DESCRIPTOR_NAMES), dtype=float)
        fingerprint = np.zeros(FINGERPRINT_NBITS, dtype=np.uint8)
    else:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # SMILES parsed from an external source but RDKit rejected it —
            # treat as FALLBACK and record the failure honestly.
            warnings.warn(
                f"Construct '{construct.name}': resolved SMILES '{smiles}' could "
                "not be parsed by RDKit. Falling back to zero features.",
                UserWarning,
                stacklevel=2,
            )
            quality = FeatureQuality.FALLBACK
            unresolved.append("rdkit_parse_failed")
            descriptor_vector = np.zeros(len(DESCRIPTOR_NAMES), dtype=float)
            fingerprint = np.zeros(FINGERPRINT_NBITS, dtype=np.uint8)
        else:
            descriptor_vector = _compute_descriptors(mol)
            fingerprint = _compute_fingerprint(mol)

    isotope_features = _compute_isotope_features(construct)

    return FeatureRecord(
        construct_id=construct.id,
        descriptor_vector=descriptor_vector,
        fingerprint=fingerprint,
        isotope_features=isotope_features,
        quality=quality,
        unresolved_parts=unresolved,
        resolution_reasons=reasons,
        featurizer_version=FEATURIZER_VERSION,
        rdkit_version=rdkit.__version__,
        descriptor_names=DESCRIPTOR_NAMES,
        fingerprint_params={
            "radius": FINGERPRINT_RADIUS,
            "n_bits": FINGERPRINT_NBITS,
        },
    )


def tanimoto_distance(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Tanimoto (Jaccard) distance between two Morgan fingerprint arrays.

    Returns a value in [0, 1] where 0 = identical and 1 = no shared bits.
    If both arrays are all-zeros (FALLBACK records), returns 1.0 to signal
    that the structures are unknown and cannot be declared similar.
    """
    intersection = float(np.dot(fp1.astype(float), fp2.astype(float)))
    union = float(np.sum(fp1) + np.sum(fp2) - intersection)
    if union == 0.0:
        # Both fingerprints are all-zeros (both FALLBACK): structural similarity
        # is unknown; treat as maximally distant to allow selection.
        return 1.0
    return 1.0 - intersection / union


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_descriptors(mol: Chem.Mol) -> np.ndarray:
    """Compute the 8-feature descriptor vector.

    The set is intentionally small: Gaussian-process surrogates fit on very few
    labelled observations, and a high-dimensional representation would overfit.
    All descriptors are standard RDKit functions with well-defined semantics.
    """
    return np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            float(rdMolDescriptors.CalcNumHBD(mol)),
            float(rdMolDescriptors.CalcNumHBA(mol)),
            float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            float(rdMolDescriptors.CalcNumRings(mol)),
            Descriptors.FractionCSP3(mol),
        ],
        dtype=float,
    )


def _compute_fingerprint(mol: Chem.Mol) -> np.ndarray:
    """Compute a 2048-bit Morgan fingerprint (radius 2) as a uint8 bit array."""
    gen = GetMorganGenerator(radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_NBITS)
    return gen.GetFingerprintAsNumPy(mol)


def _compute_isotope_features(construct: CandidateConstruct) -> np.ndarray:
    """Compute three isotope physics features from factual domain data.

    Returns [atomic_number, half_life_days, decay_mode_encoded].
    All values come from the project's single sources of truth:
      - atomic_number from RDKit's periodic table (by element symbol)
      - half_life_days from domain.models.HALF_LIFE_DAYS
      - decay_mode_encoded from featurization.isotope_data.ISOTOPE_PHYSICS
    """
    from rdkit.Chem import GetPeriodicTable

    pt = GetPeriodicTable()
    physics = ISOTOPE_PHYSICS.get(construct.radionuclide)

    atomic_number = (
        float(pt.GetAtomicNumber(physics["element_symbol"])) if physics else 0.0
    )
    half_life = float(HALF_LIFE_DAYS.get(construct.radionuclide, 0.0))
    decay_mode_encoded = float(physics["decay_mode_encoded"]) if physics else -1.0

    return np.array([atomic_number, half_life, decay_mode_encoded], dtype=float)
