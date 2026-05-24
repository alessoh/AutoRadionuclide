"""
Factual isotope physics data for radionuclides used in radioligand therapy.

IMPORTANT: Half-life values are NOT duplicated here. They are read from
``autoradionuclide.domain.models.HALF_LIFE_DAYS`` (single source of truth).

Decay-mode data source:
  IAEA Live Chart of Nuclides — Nuclear Data Section
  https://www-nds.iaea.org/relnsd/vcharthtml/VChartHTML.html
  Accessed: 2026
"""
from __future__ import annotations

from autoradionuclide.domain.models import Radionuclide


# Primary decay-mode encoding
# 0 = beta-minus (β⁻), 1 = alpha (α), 2 = electron-capture / positron (EC/β⁺)
DECAY_MODE_ENCODING: dict[str, int] = {
    "beta_minus": 0,
    "alpha": 1,
    "ec_positron": 2,
}

# Per-isotope physics: element symbol and primary decay mode.
# Source: IAEA Live Chart of Nuclides (see module docstring).
ISOTOPE_PHYSICS: dict[Radionuclide, dict] = {
    Radionuclide.LU177: {
        "element_symbol": "Lu",
        # β⁻ to Hf-177 (primary; 100 % β⁻)
        "primary_decay": "beta_minus",
        "decay_mode_encoded": DECAY_MODE_ENCODING["beta_minus"],
    },
    Radionuclide.AC225: {
        "element_symbol": "Ac",
        # α decay chain ending at Bi-209 via multiple daughters
        "primary_decay": "alpha",
        "decay_mode_encoded": DECAY_MODE_ENCODING["alpha"],
    },
    Radionuclide.GA68: {
        "element_symbol": "Ga",
        # EC / β⁺ to Zn-68 (89 % β⁺ + 11 % EC); used as PET imaging agent
        "primary_decay": "ec_positron",
        "decay_mode_encoded": DECAY_MODE_ENCODING["ec_positron"],
    },
    Radionuclide.Y90: {
        "element_symbol": "Y",
        # β⁻ to Zr-90 (100 % β⁻)
        "primary_decay": "beta_minus",
        "decay_mode_encoded": DECAY_MODE_ENCODING["beta_minus"],
    },
    Radionuclide.I131: {
        "element_symbol": "I",
        # β⁻ + γ to Xe-131 (primary β⁻ branch)
        "primary_decay": "beta_minus",
        "decay_mode_encoded": DECAY_MODE_ENCODING["beta_minus"],
    },
    Radionuclide.BI213: {
        "element_symbol": "Bi",
        # β⁻ (97.8 %) to Po-213, which α-decays with t½ = 4 µs.
        # Bi-213 itself is β⁻; therapy benefit comes from the α-emitting Po-213 daughter.
        # Encoding reflects the direct decay of the Bi-213 nucleus.
        "primary_decay": "beta_minus",
        "decay_mode_encoded": DECAY_MODE_ENCODING["beta_minus"],
    },
    Radionuclide.AT211: {
        "element_symbol": "At",
        # α (58.2 %) + EC (41.8 %) to Po-211 / Bi-207; classified as α-emitter for therapy
        "primary_decay": "alpha",
        "decay_mode_encoded": DECAY_MODE_ENCODING["alpha"],
    },
}
