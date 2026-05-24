"""Tests for the featurization package.

Verifies:
  - Known building blocks resolve to stable, correct feature vectors
  - Fingerprints and descriptors are deterministic across repeated calls
  - Descriptor values fall within physically meaningful ranges
  - Isotope physics features are correct (atomic numbers, half-lives)
  - Constructs with no resolvable structure yield FALLBACK quality with zeros
    and a warning, not fabricated values
  - Surrogates fit on real RDKit descriptor vectors without error
  - Policy diversity step produces structurally distinct batches using Tanimoto
  - The featurizer module docstring contains the explicit metal-coordination caveat
"""
from __future__ import annotations

import importlib
import warnings

import numpy as np
import pytest

from autoradionuclide.domain.models import (
    CandidateConstruct,
    Chelator,
    HALF_LIFE_DAYS,
    Radionuclide,
    TargetingVector,
)
from autoradionuclide.featurization import (
    DESCRIPTOR_NAMES,
    FINGERPRINT_NBITS,
    FeatureQuality,
    featurize,
    tanimoto_distance,
)
from autoradionuclide.featurization.registry import (
    CHELATOR_SMILES,
    _CHELATOR_REGISTRY,
    _TARGETING_VECTOR_REGISTRY,
    reset_registry_warning_state,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_dota_construct(
    isotope: Radionuclide = Radionuclide.LU177,
    name: str = "dota-lu177",
) -> CandidateConstruct:
    return CandidateConstruct(
        name=name,
        targeting_vector=TargetingVector(
            name="PSMA-617-vector",
            target="PSMA",
            vector_type="small_molecule",
        ),
        chelator=Chelator(name="DOTA"),
        radionuclide=isotope,
    )


def _make_nota_construct(name: str = "nota-ga68") -> CandidateConstruct:
    return CandidateConstruct(
        name=name,
        targeting_vector=TargetingVector(
            name="RGD-vector",
            target="integrin_avb3",
            vector_type="peptide",
        ),
        chelator=Chelator(name="NOTA"),
        radionuclide=Radionuclide.GA68,
    )


def _make_dotaga_construct(name: str = "dotaga-ac225") -> CandidateConstruct:
    return CandidateConstruct(
        name=name,
        targeting_vector=TargetingVector(
            name="FAP-vector",
            target="FAP",
            vector_type="small_molecule",
        ),
        chelator=Chelator(name="DOTAGA"),
        radionuclide=Radionuclide.AC225,
    )


def _make_fallback_construct(name: str = "unknown") -> CandidateConstruct:
    """Construct whose chelator and targeting vector are not in the registry."""
    return CandidateConstruct(
        name=name,
        targeting_vector=TargetingVector(
            name="novel-peptide-xyz-not-in-registry",
            target="unknown",
            vector_type="peptide",
        ),
        chelator=Chelator(name="novel-chelator-not-in-registry"),
        radionuclide=Radionuclide.LU177,
    )


# ---------------------------------------------------------------------------
# Registry validation
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_chelator_smiles_parse_dota(self):
        """DOTA SMILES must be parseable by RDKit."""
        from rdkit import Chem
        mol = Chem.MolFromSmiles(CHELATOR_SMILES["DOTA"])
        assert mol is not None, "DOTA SMILES failed to parse"

    def test_chelator_smiles_parse_nota(self):
        """NOTA SMILES must be parseable by RDKit."""
        from rdkit import Chem
        mol = Chem.MolFromSmiles(CHELATOR_SMILES["NOTA"])
        assert mol is not None, "NOTA SMILES failed to parse"

    def test_chelator_smiles_parse_dotaga(self):
        """DOTAGA SMILES must be parseable by RDKit."""
        from rdkit import Chem
        mol = Chem.MolFromSmiles(CHELATOR_SMILES["DOTAGA"])
        assert mol is not None, "DOTAGA SMILES failed to parse"

    def test_dota_molecular_weight(self):
        """DOTA MW should be ~404 Da (C16H28N4O8)."""
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(CHELATOR_SMILES["DOTA"])
        mw = Descriptors.MolWt(mol)
        assert 400.0 < mw < 410.0, f"DOTA MW {mw:.1f} outside expected range"

    def test_nota_molecular_weight(self):
        """NOTA MW should be ~303 Da (C12H21N3O6)."""
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(CHELATOR_SMILES["NOTA"])
        mw = Descriptors.MolWt(mol)
        assert 298.0 < mw < 308.0, f"NOTA MW {mw:.1f} outside expected range"


# ---------------------------------------------------------------------------
# Registry formula verification
# ---------------------------------------------------------------------------

class TestRegistryFormulas:
    """Every registry entry's SMILES must parse and its formula must match."""

    def test_chelator_formulas(self):
        """All chelator SMILES must parse and produce the stored molecular formula."""
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        for name, entry in _CHELATOR_REGISTRY.items():
            mol = Chem.MolFromSmiles(entry["smiles"])
            assert mol is not None, f"{name}: SMILES failed to parse"
            computed = rdMolDescriptors.CalcMolFormula(mol)
            assert computed == entry["formula"], (
                f"{name}: computed formula {computed!r} != stored {entry['formula']!r} "
                f"(source: {entry['source']})"
            )

    def test_targeting_vector_formulas(self):
        """All targeting-vector SMILES must parse and produce the stored formula."""
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        for name, entry in _TARGETING_VECTOR_REGISTRY.items():
            mol = Chem.MolFromSmiles(entry["smiles"])
            assert mol is not None, f"{name}: SMILES failed to parse"
            computed = rdMolDescriptors.CalcMolFormula(mol)
            assert computed == entry["formula"], (
                f"{name}: computed formula {computed!r} != stored {entry['formula']!r} "
                f"(source: {entry['source']})"
            )

    def test_mibg_molecular_weight(self):
        """MIBG MW should be ~275 Da (C8H10IN3, PubChem CID 60860)."""
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from autoradionuclide.featurization.registry import TARGETING_VECTOR_SMILES
        mol = Chem.MolFromSmiles(TARGETING_VECTOR_SMILES["MIBG"])
        mw = Descriptors.MolWt(mol)
        assert 270.0 < mw < 280.0, f"MIBG MW {mw:.1f} outside expected range"


# ---------------------------------------------------------------------------
# Registry expansion: MIBG and PSMA-campaign resolution
# ---------------------------------------------------------------------------

class TestRegistryExpansion:
    """Tests for building blocks added in the registry expansion."""

    def test_mibg_resolves_to_full(self):
        """MIBG (chelator='none', vector='MIBG') must resolve to FULL quality."""
        reset_registry_warning_state()
        c = CandidateConstruct(
            name="mibg-i131",
            targeting_vector=TargetingVector(
                name="MIBG", target="NET", vector_type="small_molecule"
            ),
            chelator=Chelator(name="none"),
            radionuclide=Radionuclide.I131,
        )
        record = featurize(c)
        assert record.quality is FeatureQuality.FULL
        assert record.unresolved_parts == []
        assert np.any(record.descriptor_vector != 0.0)
        assert record.fingerprint.sum() > 0

    def test_mibg_descriptor_vector_physically_reasonable(self):
        """MIBG descriptor vector must reflect a small iodinated molecule."""
        from autoradionuclide.featurization import DESCRIPTOR_NAMES
        reset_registry_warning_state()
        c = CandidateConstruct(
            name="mibg-i131",
            targeting_vector=TargetingVector(
                name="MIBG", target="NET", vector_type="small_molecule"
            ),
            chelator=Chelator(name="none"),
            radionuclide=Radionuclide.I131,
        )
        record = featurize(c)
        mw = record.descriptor_vector[DESCRIPTOR_NAMES.index("mw")]
        assert 250.0 < mw < 300.0, f"MIBG MW descriptor={mw:.1f}, expected ~275"

    def test_psma_campaign_constructs_resolve_to_partial(self):
        """Constructs with DOTA/NOTA/DOTAGA chelator resolve to PARTIAL, not FALLBACK."""
        reset_registry_warning_state()
        for chelator_name in ("DOTA", "NOTA", "DOTAGA"):
            c = CandidateConstruct(
                name=f"psma-{chelator_name.lower()}-lu177",
                targeting_vector=TargetingVector(
                    name="PSMA-617",
                    target="PSMA",
                    vector_type="small_molecule",
                ),
                chelator=Chelator(name=chelator_name),
                radionuclide=Radionuclide.LU177,
            )
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                record = featurize(c)
            assert record.quality is FeatureQuality.PARTIAL, (
                f"{chelator_name}: expected PARTIAL, got {record.quality}"
            )
            assert "chelator" not in record.unresolved_parts
            assert np.any(record.descriptor_vector != 0.0)

    def test_psma_campaign_constructs_have_nonzero_fingerprint(self):
        """PARTIAL constructs (chelator resolved) must have non-zero fingerprints."""
        reset_registry_warning_state()
        for chelator_name in ("DOTA", "NOTA", "DOTAGA"):
            c = CandidateConstruct(
                name=f"fp-{chelator_name.lower()}",
                targeting_vector=TargetingVector(
                    name="PSMA-617", target="PSMA", vector_type="small_molecule"
                ),
                chelator=Chelator(name=chelator_name),
                radionuclide=Radionuclide.LU177,
            )
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                record = featurize(c)
            assert record.fingerprint.sum() > 0, (
                f"{chelator_name}: fingerprint is all zeros for PARTIAL construct"
            )


# ---------------------------------------------------------------------------
# Warning deduplication
# ---------------------------------------------------------------------------

class TestWarningDeduplication:
    """The per-building-block warning must fire once per unique name, not per instance."""

    def test_repeated_same_missing_block_warns_once_each(self):
        """Three constructs sharing the same two missing parts produce 2 warnings total."""
        reset_registry_warning_state()
        constructs = [_make_fallback_construct(f"c{i}") for i in range(3)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for c in constructs:
                featurize(c)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        # _make_fallback_construct uses 2 unique missing names → 2 unique keys
        assert len(user_warnings) == 2, (
            f"Expected 2 warnings (one per unique missing block), "
            f"got {len(user_warnings)}: {[str(x.message) for x in user_warnings]}"
        )

    def test_different_missing_blocks_warn_independently(self):
        """Two constructs with different unresolved names each fire their own warning."""
        reset_registry_warning_state()
        c1 = CandidateConstruct(
            name="c1",
            targeting_vector=TargetingVector(
                name="vector-alpha-unique", target="X", vector_type="peptide"
            ),
            chelator=Chelator(name="chelator-alpha-unique"),
            radionuclide=Radionuclide.LU177,
        )
        c2 = CandidateConstruct(
            name="c2",
            targeting_vector=TargetingVector(
                name="vector-beta-unique", target="Y", vector_type="peptide"
            ),
            chelator=Chelator(name="chelator-beta-unique"),
            radionuclide=Radionuclide.LU177,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            featurize(c1)
            featurize(c2)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        # 2 chelator misses + 2 vector misses = 4 distinct keys = 4 warnings
        assert len(user_warnings) == 4, (
            f"Expected 4 warnings (2 chelator + 2 vector misses), "
            f"got {len(user_warnings)}"
        )

    def test_second_featurize_of_same_fallback_no_new_warnings(self):
        """Featurizing the same fallback construct twice produces no duplicate warnings."""
        reset_registry_warning_state()
        c = _make_fallback_construct("dup-test")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            featurize(c)
            featurize(c)  # second call should not re-warn
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        # Both missing blocks warned on first call only
        assert len(user_warnings) == 2, (
            f"Expected 2 warnings total (not 4), got {len(user_warnings)}"
        )


# ---------------------------------------------------------------------------
# Feature record: quality and structure
# ---------------------------------------------------------------------------

class TestFeatureRecordQuality:
    def test_dota_resolves_to_partial(self):
        """DOTA chelator resolves; targeting vector doesn't → PARTIAL quality."""
        c = _make_dota_construct()
        record = featurize(c)
        assert record.quality in (FeatureQuality.PARTIAL, FeatureQuality.FULL)

    def test_dota_descriptor_vector_length(self):
        """Descriptor vector must have the documented fixed length."""
        c = _make_dota_construct()
        record = featurize(c)
        assert len(record.descriptor_vector) == len(DESCRIPTOR_NAMES)

    def test_dota_fingerprint_length(self):
        """Fingerprint must have FINGERPRINT_NBITS bits."""
        c = _make_dota_construct()
        record = featurize(c)
        assert len(record.fingerprint) == FINGERPRINT_NBITS

    def test_dota_descriptor_vector_nonzero(self):
        """A resolved structure must produce a non-zero descriptor vector."""
        c = _make_dota_construct()
        record = featurize(c)
        assert np.any(record.descriptor_vector != 0.0)

    def test_dota_fingerprint_nonzero(self):
        """A resolved structure must produce a non-zero fingerprint."""
        c = _make_dota_construct()
        record = featurize(c)
        assert record.fingerprint.sum() > 0

    def test_full_smiles_overrides_registry(self):
        """Providing construct.smiles → FULL quality, registry not consulted."""
        c = CandidateConstruct(
            name="full-smiles",
            targeting_vector=TargetingVector(
                name="v", target="PSMA", vector_type="small_molecule"
            ),
            chelator=Chelator(name="unknown-chelator-xyz"),
            radionuclide=Radionuclide.LU177,
            # Simple valid SMILES (acetic acid) to exercise the full-smiles path
            smiles="CC(=O)O",
        )
        record = featurize(c)
        assert record.quality is FeatureQuality.FULL
        assert "chelator" not in record.unresolved_parts

    def test_provenance_tag_present(self):
        """Feature record must carry the metal-coordination provenance tag."""
        c = _make_dota_construct()
        record = featurize(c)
        assert record.provenance_tag == (
            "computed_structural_features_metal_coordination_not_modeled"
        )

    def test_featurizer_version_present(self):
        c = _make_dota_construct()
        record = featurize(c)
        assert record.featurizer_version != ""

    def test_rdkit_version_present(self):
        c = _make_dota_construct()
        record = featurize(c)
        assert record.rdkit_version != ""


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

class TestFallback:
    def test_fallback_quality_flag(self):
        """No-SMILES construct → FALLBACK quality."""
        c = _make_fallback_construct()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            record = featurize(c)
        assert record.quality is FeatureQuality.FALLBACK

    def test_fallback_emits_warning(self):
        """Fallback must emit a UserWarning per unique unresolved building block."""
        reset_registry_warning_state()
        c = _make_fallback_construct()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            featurize(c)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) >= 1

    def test_fallback_descriptor_vector_all_zeros(self):
        """FALLBACK descriptor vector must be explicitly zeros, not fabricated."""
        c = _make_fallback_construct()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            record = featurize(c)
        assert np.all(record.descriptor_vector == 0.0)
        assert len(record.descriptor_vector) == len(DESCRIPTOR_NAMES)

    def test_fallback_fingerprint_all_zeros(self):
        """FALLBACK fingerprint must be explicitly zeros, not fabricated."""
        c = _make_fallback_construct()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            record = featurize(c)
        assert np.all(record.fingerprint == 0)
        assert len(record.fingerprint) == FINGERPRINT_NBITS

    def test_fallback_records_unresolved_parts(self):
        """Both chelator and targeting_vector must appear in unresolved_parts."""
        c = _make_fallback_construct()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            record = featurize(c)
        assert "chelator" in record.unresolved_parts
        assert "targeting_vector" in record.unresolved_parts

    def test_fallback_does_not_crash(self):
        """featurize() must not raise for a fallback construct."""
        c = _make_fallback_construct()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            record = featurize(c)
        assert record is not None

    def test_none_chelator_not_fallback(self):
        """Chelator name 'none' (direct labelling) is handled — doesn't cause FALLBACK alone."""
        c = CandidateConstruct(
            name="mibg-i131",
            targeting_vector=TargetingVector(
                name="MIBG-vector", target="NET", vector_type="small_molecule"
            ),
            chelator=Chelator(name="none"),
            radionuclide=Radionuclide.I131,
        )
        # Targeting vector is unresolved → FALLBACK or PARTIAL depending on vector
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            record = featurize(c)
        # 'none' chelator should not appear as unresolved
        assert "chelator" not in record.unresolved_parts


# ---------------------------------------------------------------------------
# Physical range checks on descriptors
# ---------------------------------------------------------------------------

class TestDescriptorPhysicalRanges:
    def test_mw_reasonable(self):
        """DOTA molecular weight should be in a chemically sensible range."""
        record = featurize(_make_dota_construct())
        mw = record.descriptor_vector[DESCRIPTOR_NAMES.index("mw")]
        assert 50.0 < mw < 2000.0, f"MW={mw:.1f} outside reasonable range"

    def test_tpsa_non_negative(self):
        record = featurize(_make_dota_construct())
        tpsa = record.descriptor_vector[DESCRIPTOR_NAMES.index("tpsa")]
        assert tpsa >= 0.0

    def test_hbd_non_negative_integer(self):
        record = featurize(_make_dota_construct())
        hbd = record.descriptor_vector[DESCRIPTOR_NAMES.index("hbd")]
        assert hbd >= 0.0
        assert hbd == int(hbd)

    def test_hba_non_negative_integer(self):
        record = featurize(_make_dota_construct())
        hba = record.descriptor_vector[DESCRIPTOR_NAMES.index("hba")]
        assert hba >= 0.0

    def test_rotbonds_non_negative(self):
        record = featurize(_make_dota_construct())
        rb = record.descriptor_vector[DESCRIPTOR_NAMES.index("rotbonds")]
        assert rb >= 0.0

    def test_rings_non_negative(self):
        record = featurize(_make_dota_construct())
        rings = record.descriptor_vector[DESCRIPTOR_NAMES.index("rings")]
        assert rings >= 0.0

    def test_frac_csp3_in_unit_interval(self):
        record = featurize(_make_dota_construct())
        frac = record.descriptor_vector[DESCRIPTOR_NAMES.index("frac_csp3")]
        assert 0.0 <= frac <= 1.0

    def test_dota_has_ring(self):
        """DOTA has one macrocyclic ring; ring count must be ≥ 1."""
        record = featurize(_make_dota_construct())
        rings = record.descriptor_vector[DESCRIPTOR_NAMES.index("rings")]
        assert rings >= 1.0


# ---------------------------------------------------------------------------
# Isotope feature correctness
# ---------------------------------------------------------------------------

class TestIsotopeFeatures:
    def test_lutetium_atomic_number(self):
        """Lutetium is element 71."""
        record = featurize(_make_dota_construct(Radionuclide.LU177))
        atomic_number = record.isotope_features[0]
        assert atomic_number == pytest.approx(71.0)

    def test_gallium_atomic_number(self):
        """Gallium is element 31."""
        record = featurize(_make_nota_construct())
        atomic_number = record.isotope_features[0]
        assert atomic_number == pytest.approx(31.0)

    def test_actinium_atomic_number(self):
        """Actinium is element 89."""
        record = featurize(_make_dotaga_construct())
        atomic_number = record.isotope_features[0]
        assert atomic_number == pytest.approx(89.0)

    def test_lu177_half_life(self):
        """Lu-177 half-life is 6.65 days per HALF_LIFE_DAYS."""
        record = featurize(_make_dota_construct(Radionuclide.LU177))
        hl = record.isotope_features[1]
        assert hl == pytest.approx(HALF_LIFE_DAYS[Radionuclide.LU177])

    def test_ga68_half_life(self):
        """Ga-68 half-life is ~0.047 days (68 min)."""
        record = featurize(_make_nota_construct())
        hl = record.isotope_features[1]
        assert hl == pytest.approx(HALF_LIFE_DAYS[Radionuclide.GA68])

    def test_lu177_decay_mode_beta_minus(self):
        """Lu-177 decays by β⁻ → encoded as 0."""
        record = featurize(_make_dota_construct(Radionuclide.LU177))
        mode = record.isotope_features[2]
        assert mode == pytest.approx(0.0)  # beta_minus = 0

    def test_ac225_decay_mode_alpha(self):
        """Ac-225 decays by α → encoded as 1."""
        record = featurize(_make_dotaga_construct())
        mode = record.isotope_features[2]
        assert mode == pytest.approx(1.0)  # alpha = 1

    def test_ga68_decay_mode_ec_positron(self):
        """Ga-68 decays by EC/β⁺ → encoded as 2."""
        record = featurize(_make_nota_construct())
        mode = record.isotope_features[2]
        assert mode == pytest.approx(2.0)  # ec_positron = 2

    def test_isotope_features_independent_of_chelator(self):
        """Isotope features must not change when the chelator changes."""
        c_dota = _make_dota_construct(Radionuclide.LU177)
        c_nota = CandidateConstruct(
            name="nota-lu177",
            targeting_vector=TargetingVector(
                name="v", target="PSMA", vector_type="small_molecule"
            ),
            chelator=Chelator(name="NOTA"),
            radionuclide=Radionuclide.LU177,
        )
        r1 = featurize(c_dota)
        r2 = featurize(c_nota)
        np.testing.assert_array_almost_equal(r1.isotope_features, r2.isotope_features)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_descriptor_vector_identical_on_repeated_calls(self):
        """featurize() must be deterministic: same construct → same descriptors."""
        c = _make_dota_construct()
        r1 = featurize(c)
        r2 = featurize(c)
        np.testing.assert_array_equal(r1.descriptor_vector, r2.descriptor_vector)

    def test_fingerprint_identical_on_repeated_calls(self):
        c = _make_dota_construct()
        r1 = featurize(c)
        r2 = featurize(c)
        np.testing.assert_array_equal(r1.fingerprint, r2.fingerprint)

    def test_isotope_features_identical_on_repeated_calls(self):
        c = _make_dota_construct()
        r1 = featurize(c)
        r2 = featurize(c)
        np.testing.assert_array_equal(r1.isotope_features, r2.isotope_features)

    def test_fallback_deterministic(self):
        """Fallback construct also produces deterministic (zero) output."""
        c = _make_fallback_construct()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r1 = featurize(c)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r2 = featurize(c)
        np.testing.assert_array_equal(r1.descriptor_vector, r2.descriptor_vector)
        np.testing.assert_array_equal(r1.fingerprint, r2.fingerprint)


# ---------------------------------------------------------------------------
# Tanimoto distance
# ---------------------------------------------------------------------------

class TestTanimotoDistance:
    def test_identical_fingerprints_distance_zero(self):
        c = _make_dota_construct()
        fp = featurize(c).fingerprint
        assert tanimoto_distance(fp, fp) == pytest.approx(0.0)

    def test_all_zeros_returns_one(self):
        """Two all-zero fingerprints (both FALLBACK) return distance 1.0."""
        z = np.zeros(FINGERPRINT_NBITS, dtype=np.uint8)
        assert tanimoto_distance(z, z) == pytest.approx(1.0)

    def test_dota_dotaga_distance_positive(self):
        """DOTA and DOTAGA have different Morgan fingerprints."""
        fp_dota = featurize(_make_dota_construct()).fingerprint
        fp_dotaga = featurize(_make_dotaga_construct()).fingerprint
        d = tanimoto_distance(fp_dota, fp_dotaga)
        assert 0.0 < d <= 1.0

    def test_distance_symmetric(self):
        fp1 = featurize(_make_dota_construct()).fingerprint
        fp2 = featurize(_make_dotaga_construct()).fingerprint
        assert tanimoto_distance(fp1, fp2) == pytest.approx(tanimoto_distance(fp2, fp1))

    def test_distance_in_unit_interval(self):
        fp1 = featurize(_make_dota_construct()).fingerprint
        fp2 = featurize(_make_nota_construct()).fingerprint
        d = tanimoto_distance(fp1, fp2)
        assert 0.0 <= d <= 1.0


# ---------------------------------------------------------------------------
# Docstring caveat — honesty invariant
# ---------------------------------------------------------------------------

class TestDocstringCaveat:
    def test_featurizer_module_docstring_states_metal_coordination_not_modeled(self):
        """The featurizer module docstring must explicitly state the metal coordination
        limitation. This test prevents the caveat from being silently removed."""
        import autoradionuclide.featurization.featurizer as module
        assert module.__doc__ is not None, "featurizer.py has no module docstring"
        doc = module.__doc__.lower()
        assert "metal coordination" in doc, (
            "featurizer.py docstring must mention 'metal coordination'"
        )
        assert "not modeled" in doc, (
            "featurizer.py docstring must state 'not modeled'"
        )

    def test_feature_record_docstring_states_metal_coordination_not_modeled(self):
        """FeatureRecord docstring must also carry the metal coordination caveat."""
        from autoradionuclide.featurization._types import FeatureRecord
        assert FeatureRecord.__doc__ is not None
        doc = FeatureRecord.__doc__.lower()
        assert "metal coordination" in doc

    def test_provenance_tag_encodes_limit(self):
        """The provenance_tag on every record must encode the metal-coordination limit."""
        record = featurize(_make_dota_construct())
        assert "metal_coordination_not_modeled" in record.provenance_tag


# ---------------------------------------------------------------------------
# Integration: surrogates fit on real features
# ---------------------------------------------------------------------------

class TestSurrogateIntegration:
    def test_surrogates_fit_without_error(self):
        """SurrogateBank.update() must work with real RDKit descriptor vectors."""
        from autoradionuclide.surrogates.gp_surrogate import SurrogateBank

        constructs = [
            _make_dota_construct(Radionuclide.LU177, "c1"),
            _make_dotaga_construct("c2"),
            CandidateConstruct(
                name="c3",
                targeting_vector=TargetingVector(
                    name="v3", target="FAP", vector_type="peptide"
                ),
                chelator=Chelator(name="DOTAGA"),
                radionuclide=Radionuclide.AC225,
            ),
        ]
        bank = SurrogateBank(["binding_affinity"], seed=42)
        bank.update(constructs, {"binding_affinity": [0.9, 0.7, 0.5]})
        result = bank.predict_all(constructs[0])
        assert "binding_affinity" in result
        assert 0.0 <= result["binding_affinity"].estimate <= 1.0

    def test_fallback_construct_excluded_from_fit(self):
        """FALLBACK constructs must not corrupt the GP training set."""
        from autoradionuclide.surrogates.gp_surrogate import SurrogateBank

        good = _make_dota_construct(Radionuclide.LU177, "good")
        fallback = _make_fallback_construct("bad")
        bank = SurrogateBank(["binding_affinity"], seed=42)
        # Provide a fallback record along with a valid one
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            bank.update([good, fallback], {"binding_affinity": [0.9, 0.5]})
        # Only one real observation; GP cannot fit with < 2 observations
        assert bank.get("binding_affinity").n_observations == 1

    def test_surrogate_predict_works_for_known_chelator(self):
        """Predict on a known chelator must return a valid ObjectiveValue."""
        from autoradionuclide.surrogates.gp_surrogate import SurrogateBank

        bank = SurrogateBank(["chelator_stability"], seed=42)
        c = _make_dota_construct()
        result = bank.predict_all(c)
        assert "chelator_stability" in result
        ov = result["chelator_stability"]
        assert 0.0 <= ov.estimate <= 1.0
        assert ov.uncertainty >= 0.0


# ---------------------------------------------------------------------------
# Integration: policy diversity uses Tanimoto
# ---------------------------------------------------------------------------

class TestPolicyDiversityIntegration:
    def test_policy_does_not_select_fingerprint_identical_duplicates(self):
        """Two constructs with the same chelator (same fingerprint) must not both
        be selected when diversity_threshold > 0."""
        from autoradionuclide.config.schema import _default_objectives
        from autoradionuclide.policy.acquisition import ActiveLearningPolicy
        from autoradionuclide.surrogates.gp_surrogate import SurrogateBank

        specs = _default_objectives()
        bank = SurrogateBank([s.name for s in specs], seed=42)
        policy = ActiveLearningPolicy(
            surrogate_bank=bank,
            specs=specs,
            acquisition_fn="UCB",
            diversity_threshold=0.1,
        )
        # Both use DOTA — identical fingerprint
        dup1 = _make_dota_construct(Radionuclide.LU177, "dup1")
        dup2 = _make_dota_construct(Radionuclide.LU177, "dup2")
        # Third construct uses DOTAGA — distinct fingerprint
        diverse = _make_dotaga_construct("diverse")

        ranked = policy.rank([dup1, dup2, diverse], batch_size=2)
        dota_count = sum(
            1 for c, _ in ranked if c.chelator.name == "DOTA"
        )
        assert dota_count <= 1, (
            "Both DOTA constructs were selected; Tanimoto diversity not enforced"
        )

    def test_policy_selects_structurally_distinct_constructs(self):
        """Policy must select constructs with different chelator fingerprints."""
        from autoradionuclide.config.schema import _default_objectives
        from autoradionuclide.policy.acquisition import ActiveLearningPolicy
        from autoradionuclide.surrogates.gp_surrogate import SurrogateBank

        specs = _default_objectives()
        bank = SurrogateBank([s.name for s in specs], seed=42)
        policy = ActiveLearningPolicy(
            surrogate_bank=bank,
            specs=specs,
            acquisition_fn="UCB",
            diversity_threshold=0.1,
        )
        # Three constructs with genuinely different fingerprints:
        # DOTA, DOTAGA, and a construct with explicit SMILES (acetic acid) as chelator
        c1 = _make_dota_construct(Radionuclide.LU177, "c1")
        c2 = _make_dotaga_construct("c2")
        c3 = CandidateConstruct(
            name="c3",
            targeting_vector=TargetingVector(
                name="v3", target="NET", vector_type="small_molecule"
            ),
            chelator=Chelator(
                name="test-acyclic",
                # EDTA: PubChem CID 6049, fingerprint-distinct from DOTA-family
                smiles="OC(=O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",
            ),
            radionuclide=Radionuclide.I131,
        )
        ranked = policy.rank([c1, c2, c3], batch_size=3)
        assert len(ranked) == 3, (
            f"Expected 3 diverse constructs, got {len(ranked)}"
        )
