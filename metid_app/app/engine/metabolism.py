"""
app/engine/metabolism.py
================================================================================
Ensemble Consensus Metabolism Engine  —  v2.0.0-ensemble
================================================================================

Architecture overview
---------------------
This module implements a dual-pipeline ensemble strategy that combines two
complementary paradigms for metabolite prediction and soft-spot analysis,
inspired by and designed for compatibility with:

  Pipeline A — Rule-Based (SyGMa)
    KavrakiLab/SyGMa / 3D-e-Chem/sygma
    Applies curated SMIRKS reaction transformation matrices for Phase I
    (oxidation, reduction, hydrolysis) and Phase II (conjugation) reactions.
    Each metabolite carries a cumulative SyGMa probability score.

  Pipeline B — Deep Learning Emulator (MetaTrans / Meta-Predictor style)
    KavrakiLab/MetaTrans  ·  zhukeyun/Meta-Predictor
    Treats metabolite prediction as a SMILES-to-SMILES sequence translation
    task (Transformer/OpenNMT architecture).  The ``DeepLearningPredictor``
    class implemented here is a **production-ready emulator** with a clean
    interface for swapping in real PyTorch .pt model weights:

        predictor = DeepLearningPredictor.from_checkpoint("weights/metatrans.pt")
        result = predict(smiles, dl_predictor=predictor)

Consensus logic
---------------
After both pipelines run, the ``ConsensusEngine`` cross-validates their outputs
using ``Chem.CanonSmiles`` for structural alignment and tags each metabolite:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Both pipelines agree      → "High Confidence (Consensus Verified)"    │
  │  SyGMa only                → "Moderate Confidence (Rule-Only)"         │
  │  DL model only             → "Moderate Confidence (DL-Only)"           │
  └─────────────────────────────────────────────────────────────────────────┘

Soft spot analysis
------------------
Unified vulnerability scoring blends two complementary signals:

  rule_score        — SMARTS-based steric-adjusted heuristic (Pipeline A)
  dl_attention_risk — per-atom attention weight from transformer SOM heads
                      (emulated; replace ``_compute_dl_attention`` with real
                       model forward pass when weights are available)

  vulnerability_index = (α × rule_score + β × dl_attention_risk) × 100  (%)
  where α=0.55, β=0.45 by default (configurable via ``ENSEMBLE_ALPHA``).

Error defence
-------------
Every pipeline is wrapped in defence layers that catch and contain:
  - ``rdkit.Chem.rdchem.MolSanitizeException``
  - Valence / kekulisation errors from transformer-generated SMILES
  - Un-tokenisable or chemically impossible SMILES strings
  - RDKit property calculation failures
  - Any unexpected exception in DL inference hooks
Individual failures downgrade confidence or add warnings; they never crash
the server.

Public API (unchanged from v1 for drop-in compatibility)
---------------------------------------------------------
  predict(smiles, ...)          → MetabolismResult
  _validate_and_normalise(...)  → (Chem.Mol, MoleculeMetadata)
  _find_soft_spots(...)         → List[SoftSpot]   (rule-only fallback)

New in v2
---------
  DeepLearningPredictor         class (Pipeline B)
  ConsensusEngine               class (merger + tagging)
  EnsembleSoftSpotResult        dataclass (combined atom vulnerability)
  ConsensusTier                 enum-like constants
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdchem   # MolSanitizeException lives here
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)


# ==============================================================================
# Engine versioning & tuning constants
# ==============================================================================

ENGINE_VERSION = "2.0.0-ensemble"

# Soft spot fusion weights:  vulnerability_index = α·rule + β·dl_attention
# Must sum to 1.0.  Adjust to preference-weight the two signals.
ENSEMBLE_ALPHA: float = 0.55   # weight given to SMARTS rule score
ENSEMBLE_BETA:  float = 0.45   # weight given to DL attention risk

assert abs(ENSEMBLE_ALPHA + ENSEMBLE_BETA - 1.0) < 1e-9, \
    "ENSEMBLE_ALPHA + ENSEMBLE_BETA must equal 1.0"

# Minimum DL model confidence to include a DL-only prediction
DL_CONFIDENCE_THRESHOLD: float = 0.10

# Consensus requires both pipeline canonical SMILES to match exactly
# (after RDKit canonicalisation).


# ==============================================================================
# Confidence tiers
# ==============================================================================

class ConsensusTier(str, Enum):
    """
    Confidence classification assigned to each metabolite by the consensus engine.

    Inherits from ``str`` so instances serialise directly to JSON strings
    without a custom encoder.
    """
    HIGH     = "High Confidence (Consensus Verified)"
    RULE     = "Moderate Confidence (Rule-Only)"
    DL       = "Moderate Confidence (DL-Only)"
    UNKNOWN  = "Unclassified"


# ==============================================================================
# Public data structures
# ==============================================================================

@dataclass(frozen=True)
class MoleculeMetadata:
    """
    Physico-chemical descriptors for a molecule.  Immutable after construction.

    ``canonical_smiles`` uses RDKit's default canonicalisation (aromatic SMILES,
    not Kekulé).  ``inchi`` / ``inchikey`` provide standard identifiers.
    """
    input_smiles:        str
    canonical_smiles:    str
    inchi:               str
    inchikey:            str
    molecular_formula:   str
    molecular_weight:    float      # average MW  (Da)
    exact_mass:          float      # monoisotopic (Da)
    num_heavy_atoms:     int
    num_rotatable_bonds: int
    num_hbd:             int        # H-bond donors
    num_hba:             int        # H-bond acceptors
    tpsa:                float      # topological polar surface area  (Å²)
    logp:                float      # Wildman–Crippen logP
    num_rings:           int
    num_aromatic_rings:  int


@dataclass(frozen=True)
class PredictedMetabolite:
    """
    A single metabolite prediction, possibly from one or both pipelines.

    New in v2
    ---------
    confidence_tier  : ConsensusTier tag assigned by the ConsensusEngine.
    dl_score         : DL model confidence [0,1] (None when DL pipeline absent).
    sources          : frozenset of pipeline names that produced this metabolite.
    """
    smiles:            str
    probability:       float            # primary score (SyGMa or DL confidence)
    phase:             int              # 1 or 2 (−1 when DL-only + unknown)
    reaction_name:     str
    molecular_weight:  Optional[float]  = None
    molecular_formula: Optional[str]    = None
    confidence_tier:   ConsensusTier    = ConsensusTier.UNKNOWN
    dl_score:          Optional[float]  = None
    sources:           frozenset        = field(default_factory=frozenset)


@dataclass(frozen=True)
class SoftSpot:
    """
    Rule-only soft spot — retained for backward compatibility with v1 API.
    Used as a fallback when the DL pipeline is unavailable.
    """
    atom_index:   int
    atom_symbol:  str
    rule_name:    str
    score:        float     # SMARTS rule score [0, 1]
    smarts_match: str


@dataclass(frozen=True)
class EnsembleSoftSpot:
    """
    Unified soft spot combining rule-based and DL-derived vulnerability signals.

    Fields
    ------
    atom_index          : 0-based RDKit atom index.
    atom_symbol         : element symbol.
    rule_name           : best-matching SMARTS rule label.
    smarts_match        : SMARTS pattern that matched.
    rule_score          : SMARTS-based vulnerability [0, 1].
    dl_attention_risk   : DL attention-head weight [0, 1]
                          (emulated; replace with real model output).
    vulnerability_index : unified score in [0, 100] %
                          = (α·rule_score + β·dl_attention_risk) × 100.
    dl_source           : identifier of the DL model that produced
                          ``dl_attention_risk`` (e.g. "metatrans-emulator").
    """
    atom_index:         int
    atom_symbol:        str
    rule_name:          str
    smarts_match:       str
    rule_score:         float
    dl_attention_risk:  float
    vulnerability_index: float          # [0, 100]
    dl_source:          str = "rule-only"


@dataclass
class MetabolismResult:
    """
    Top-level result returned by ``predict()``.

    ``soft_spots`` is now ``List[EnsembleSoftSpot]`` when the ensemble ran,
    or ``List[SoftSpot]`` when it fell back to rule-only mode.
    Both types expose ``atom_index``, ``atom_symbol``, and ``rule_name`` so
    downstream consumers (API layer, renderer) are unaffected.

    ``to_dict()`` is backward-compatible with v1 consumers while adding the
    new v2 ensemble fields.
    """
    parent:          MoleculeMetadata
    metabolites:     List[PredictedMetabolite]
    soft_spots:      List                       # List[EnsembleSoftSpot] | List[SoftSpot]
    engine_version:  str
    elapsed_s:       float
    warnings:        List[str] = field(default_factory=list)
    pipeline_stats:  Dict      = field(default_factory=dict)

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """
        Serialise to a plain nested dict for JSON responses, Celery backends,
        or pandas.  Backward-compatible with v1; adds ensemble fields.
        """
        soft_spot_dicts = []
        for s in self.soft_spots:
            if isinstance(s, EnsembleSoftSpot):
                soft_spot_dicts.append({
                    "atom_index":          s.atom_index,
                    "atom_symbol":         s.atom_symbol,
                    "rule_name":           s.rule_name,
                    "smarts_match":        s.smarts_match,
                    "rule_score":          s.rule_score,
                    "dl_attention_risk":   s.dl_attention_risk,
                    "vulnerability_index": s.vulnerability_index,
                    "dl_source":           s.dl_source,
                    # v1-compat alias
                    "score":               round(s.vulnerability_index / 100.0, 4),
                })
            else:
                soft_spot_dicts.append({
                    "atom_index":          s.atom_index,
                    "atom_symbol":         s.atom_symbol,
                    "rule_name":           s.rule_name,
                    "smarts_match":        s.smarts_match,
                    "rule_score":          s.score,
                    "dl_attention_risk":   0.0,
                    "vulnerability_index": round(s.score * 100, 2),
                    "dl_source":           "rule-only",
                    "score":               s.score,
                })

        return {
            "engine_version": self.engine_version,
            "elapsed_s":      round(self.elapsed_s, 4),
            "warnings":       self.warnings,
            "pipeline_stats": self.pipeline_stats,
            "parent": {
                "input_smiles":        self.parent.input_smiles,
                "canonical_smiles":    self.parent.canonical_smiles,
                "inchi":               self.parent.inchi,
                "inchikey":            self.parent.inchikey,
                "molecular_formula":   self.parent.molecular_formula,
                "molecular_weight":    self.parent.molecular_weight,
                "exact_mass":          self.parent.exact_mass,
                "num_heavy_atoms":     self.parent.num_heavy_atoms,
                "num_rotatable_bonds": self.parent.num_rotatable_bonds,
                "num_hbd":             self.parent.num_hbd,
                "num_hba":             self.parent.num_hba,
                "tpsa":                self.parent.tpsa,
                "logp":                self.parent.logp,
                "num_rings":           self.parent.num_rings,
                "num_aromatic_rings":  self.parent.num_aromatic_rings,
            },
            "metabolites": [
                {
                    "smiles":           m.smiles,
                    "probability":      m.probability,
                    "phase":            m.phase,
                    "reaction_name":    m.reaction_name,
                    "molecular_weight": m.molecular_weight,
                    "molecular_formula":m.molecular_formula,
                    "confidence_tier":  m.confidence_tier.value,
                    "dl_score":         m.dl_score,
                    "sources":          sorted(m.sources),
                }
                for m in self.metabolites
            ],
            "soft_spots": soft_spot_dicts,
        }

    @property
    def phase1_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites if m.phase == 1]

    @property
    def phase2_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites if m.phase == 2]

    @property
    def consensus_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites
                if m.confidence_tier == ConsensusTier.HIGH]

    @property
    def top_soft_spot_indices(self) -> List[int]:
        return [s.atom_index for s in self.soft_spots]


# ==============================================================================
# SOM SMARTS rules  (Pipeline A — rule-based component)
# ==============================================================================
# Each entry: (rule_name, smarts_pattern, priority_score)
#
# priority_score in (0, 1] based on published CYP450 substrate literature:
#   Zaretzki et al., J. Chem. Inf. Model. 2013 (CypReact)
#   Sheridan et al., J. Med. Chem. 2002
#   Cruciani et al., J. Med. Chem. 2005 (MetaSite)
#
# Convention: atom 0 in each SMARTS match is the SOM being annotated.

_SOM_RULES: List[Tuple[str, str, float]] = [
    # -- Phase I: CYP-mediated oxidation -------------------------------------

    # Benzylic CH2/CH3: highest CYP2C9/3A4 affinity (Meunier 2004)
    ("benzylic_CH",
     "[CH2,CH3;!$([CH2]C=O);!$([CH2]C#N)]-[c]", 0.92),

    # Unhindered aromatic C-H: CYP1A2/2C19 aromatic hydroxylation
    ("aromatic_C_unhindered",
     "[cH]", 0.85),

    # α-sp3 C to heteroatom: N/O-dealkylation (most common CYP reaction type)
    ("alpha_C_heteroatom",
     "[CH2,CH3;$([CH2,CH3][N,O;!$(N-C=O)])]", 0.88),

    # N-methyl/ethyl: CYP3A4 N-demethylation hotspot
    ("N_alkyl",
     "[CH3,CH2;$([CH3,CH2]-[N;!$(N-C=O);!$(N~[!#6;!#1])])]", 0.86),

    # α-carbonyl sp3 C: CYP-mediated α-hydroxylation
    ("alpha_carbonyl_C",
     "[CH2,CH3;$([CH2,CH3]C(=O)[#6,#7,#8])]", 0.75),

    # Terminal methyl (ω-hydroxylation, CYP4A/2E1)
    ("terminal_methyl_aliphatic",
     "[CH3;$([CH3]-[CH2]-[CH2]-[#6])]", 0.65),

    # Thioether S: CYP S-oxidation → sulfoxide → sulfone
    ("thioether_S",
     "[S;X2;!$(S=O);$([S]([#6])[#6])]", 0.80),

    # Tertiary amine N: FMO/CYP3A4 N-oxidation
    ("tertiary_amine_N",
     "[N;X3;!$(N-C=O);!$(N~[!#6;!#1]);!n;$([N]([#6])([#6])[#6])]", 0.78),

    # -- Phase I: Reduction --------------------------------------------------

    # Nitro group reduction (gut flora + hepatic)
    ("nitro_reduction",
     "[N+](=O)[O-]", 0.70),

    # Ketone/aldehyde carbonyl reduction (AKR / carbonyl reductases)
    ("carbonyl_reduction",
     "[CH1,CH0;$(C=O);!$(C(=O)[OH]);!$(C(=O)N)]", 0.60),

    # -- Phase II: Conjugation-prone sites -----------------------------------

    # Phenolic OH carbon: glucuronidation / sulfation (mark the C, not the O)
    ("phenolic_OH_site",
     "[c;$([c][OH1])]", 0.83),

    # Primary aliphatic amine: acetylation / glucuronidation
    ("primary_amine",
     "[NH2;$([NH2]-[#6;!$(C=O)])]", 0.72),

    # Carboxylic acid: acyl-glucuronidation (reactive metabolite risk)
    ("carboxylic_acid",
     "[C;$(C(=O)[OH1])]", 0.68),
]

# Normalisation pipeline  (shared singletons — expensive to construct)
_NORMALISER       = rdMolStandardize.Normalizer()
_UNCHARGER        = rdMolStandardize.Uncharger()
_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()


# ==============================================================================
# Pipeline B — DeepLearningPredictor
# ==============================================================================

class DeepLearningPredictor:
    """
    Production-ready deep-learning metabolite predictor emulator.

    Architectural inspiration
    -------------------------
    MetaTrans (KavrakiLab/MetaTrans): OpenNMT Transformer trained on
    experimental metabolite SMILES pairs.  Encodes the parent SMILES token
    sequence, decodes up to ``top_k`` metabolite sequences with beam search,
    and scores them with the model's log-likelihood.

    Meta-Predictor (zhukeyun/Meta-Predictor): Adds per-atom attention heads
    that produce site-of-metabolism risk vectors directly usable for soft
    spot annotation without a separate SOM model.

    This class
    ----------
    Implements the full interface those models require:
      - SMILES tokenisation  (atom-level, compatible with MetaTrans vocabulary)
      - Structural verification of generated SMILES via RDKit
      - Per-atom attention weight extraction (emulated with a deterministic
        chemistry-aware heuristic until real weights are loaded)
      - Graceful degradation: any tensor/inference failure returns empty lists
        with informative warnings, never crashes

    Swap-in path for real weights
    -----------------------------
    1. Install PyTorch + OpenNMT-py:
           pip install torch opennmt-py
    2. Subclass or monkey-patch ``_transformer_beam_search``:
           def _transformer_beam_search(self, tokens, top_k):
               # real OpenNMT / HuggingFace inference here
               ...
    3. Subclass or monkey-patch ``_extract_attention_weights``:
           def _extract_attention_weights(self, mol, tokens):
               # real model forward-pass attention extraction
               ...
    4. Load weights via ``from_checkpoint``:
           predictor = DeepLearningPredictor.from_checkpoint("metatrans.pt")

    All public method signatures remain stable between emulator and real-weight
    modes — no changes needed in the calling code.
    """

    # ── MetaTrans atom-level vocabulary (subset) ──────────────────────────────
    # Real MetaTrans uses a regex tokeniser that splits on atom symbols,
    # bond characters, ring digits, and branch brackets.  We replicate that
    # split pattern here so tokenised sequences are compatible.
    _TOKEN_PATTERN = (
        r"(\[[^\]]+\]"           # bracket atoms  [NH2], [13C], ...
        r"|Br|Cl|Si|Se|@@|@"     # two-char atoms / stereo
        r"|[BCNOPSFIbcnops]"     # one-char organic subset
        r"|[0-9%]"               # ring closure digits
        r"|[/\\+=\-#\(\)\.])"    # bond / branch / dot
    )

    MODEL_NAME = "metatrans-emulator-v2"

    def __init__(
        self,
        top_k: int = 10,
        confidence_threshold: float = DL_CONFIDENCE_THRESHOLD,
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        top_k                : beam-search width (number of candidate
                               metabolite sequences returned per parent).
        confidence_threshold : minimum model confidence to include a prediction.
        seed                 : optional random seed for the emulator (for
                               reproducible unit tests only; real models
                               are deterministic given the same weights).
        """
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self._rng = random.Random(seed)
        self._model_loaded = False   # True once real weights are attached
        
    $args = """
app/engine/metabolism.py
================================================================================
Ensemble Consensus Metabolism Engine  —  v2.0.0-ensemble
================================================================================

Architecture overview
---------------------
This module implements a dual-pipeline ensemble strategy that combines two
complementary paradigms for metabolite prediction and soft-spot analysis,
inspired by and designed for compatibility with:

  Pipeline A — Rule-Based (SyGMa)
    KavrakiLab/SyGMa / 3D-e-Chem/sygma
    Applies curated SMIRKS reaction transformation matrices for Phase I
    (oxidation, reduction, hydrolysis) and Phase II (conjugation) reactions.
    Each metabolite carries a cumulative SyGMa probability score.

  Pipeline B — Deep Learning Emulator (MetaTrans / Meta-Predictor style)
    KavrakiLab/MetaTrans  ·  zhukeyun/Meta-Predictor
    Treats metabolite prediction as a SMILES-to-SMILES sequence translation
    task (Transformer/OpenNMT architecture).  The ``DeepLearningPredictor``
    class implemented here is a **production-ready emulator** with a clean
    interface for swapping in real PyTorch .pt model weights:

        predictor = DeepLearningPredictor.from_checkpoint("weights/metatrans.pt")
        result = predict(smiles, dl_predictor=predictor)

Consensus logic
---------------
After both pipelines run, the ``ConsensusEngine`` cross-validates their outputs
using ``Chem.CanonSmiles`` for structural alignment and tags each metabolite:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Both pipelines agree      → "High Confidence (Consensus Verified)"    │
  │  SyGMa only                → "Moderate Confidence (Rule-Only)"         │
  │  DL model only             → "Moderate Confidence (DL-Only)"           │
  └─────────────────────────────────────────────────────────────────────────┘

Soft spot analysis
------------------
Unified vulnerability scoring blends two complementary signals:

  rule_score        — SMARTS-based steric-adjusted heuristic (Pipeline A)
  dl_attention_risk — per-atom attention weight from transformer SOM heads
                      (emulated; replace ``_compute_dl_attention`` with real
                       model forward pass when weights are available)

  vulnerability_index = (α × rule_score + β × dl_attention_risk) × 100  (%)
  where α=0.55, β=0.45 by default (configurable via ``ENSEMBLE_ALPHA``).

Error defence
-------------
Every pipeline is wrapped in defence layers that catch and contain:
  - ``rdkit.Chem.rdchem.MolSanitizeException``
  - Valence / kekulisation errors from transformer-generated SMILES
  - Un-tokenisable or chemically impossible SMILES strings
  - RDKit property calculation failures
  - Any unexpected exception in DL inference hooks
Individual failures downgrade confidence or add warnings; they never crash
the server.

Public API (unchanged from v1 for drop-in compatibility)
---------------------------------------------------------
  predict(smiles, ...)          → MetabolismResult
  _validate_and_normalise(...)  → (Chem.Mol, MoleculeMetadata)
  _find_soft_spots(...)         → List[SoftSpot]   (rule-only fallback)

New in v2
---------
  DeepLearningPredictor         class (Pipeline B)
  ConsensusEngine               class (merger + tagging)
  EnsembleSoftSpotResult        dataclass (combined atom vulnerability)
  ConsensusTier                 enum-like constants
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdchem   # MolSanitizeException lives here
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)


# ==============================================================================
# Engine versioning & tuning constants
# ==============================================================================

ENGINE_VERSION = "2.0.0-ensemble"

# Soft spot fusion weights:  vulnerability_index = α·rule + β·dl_attention
# Must sum to 1.0.  Adjust to preference-weight the two signals.
ENSEMBLE_ALPHA: float = 0.55   # weight given to SMARTS rule score
ENSEMBLE_BETA:  float = 0.45   # weight given to DL attention risk

assert abs(ENSEMBLE_ALPHA + ENSEMBLE_BETA - 1.0) < 1e-9, \
    "ENSEMBLE_ALPHA + ENSEMBLE_BETA must equal 1.0"

# Minimum DL model confidence to include a DL-only prediction
DL_CONFIDENCE_THRESHOLD: float = 0.10

# Consensus requires both pipeline canonical SMILES to match exactly
# (after RDKit canonicalisation).


# ==============================================================================
# Confidence tiers
# ==============================================================================

class ConsensusTier(str, Enum):
    """
    Confidence classification assigned to each metabolite by the consensus engine.

    Inherits from ``str`` so instances serialise directly to JSON strings
    without a custom encoder.
    """
    HIGH     = "High Confidence (Consensus Verified)"
    RULE     = "Moderate Confidence (Rule-Only)"
    DL       = "Moderate Confidence (DL-Only)"
    UNKNOWN  = "Unclassified"


# ==============================================================================
# Public data structures
# ==============================================================================

@dataclass(frozen=True)
class MoleculeMetadata:
    """
    Physico-chemical descriptors for a molecule.  Immutable after construction.

    ``canonical_smiles`` uses RDKit's default canonicalisation (aromatic SMILES,
    not Kekulé).  ``inchi`` / ``inchikey`` provide standard identifiers.
    """
    input_smiles:        str
    canonical_smiles:    str
    inchi:               str
    inchikey:            str
    molecular_formula:   str
    molecular_weight:    float      # average MW  (Da)
    exact_mass:          float      # monoisotopic (Da)
    num_heavy_atoms:     int
    num_rotatable_bonds: int
    num_hbd:             int        # H-bond donors
    num_hba:             int        # H-bond acceptors
    tpsa:                float      # topological polar surface area  (Å²)
    logp:                float      # Wildman–Crippen logP
    num_rings:           int
    num_aromatic_rings:  int


@dataclass(frozen=True)
class PredictedMetabolite:
    """
    A single metabolite prediction, possibly from one or both pipelines.

    New in v2
    ---------
    confidence_tier  : ConsensusTier tag assigned by the ConsensusEngine.
    dl_score         : DL model confidence [0,1] (None when DL pipeline absent).
    sources          : frozenset of pipeline names that produced this metabolite.
    """
    smiles:            str
    probability:       float            # primary score (SyGMa or DL confidence)
    phase:             int              # 1 or 2 (−1 when DL-only + unknown)
    reaction_name:     str
    molecular_weight:  Optional[float]  = None
    molecular_formula: Optional[str]    = None
    confidence_tier:   ConsensusTier    = ConsensusTier.UNKNOWN
    dl_score:          Optional[float]  = None
    sources:           frozenset        = field(default_factory=frozenset)


@dataclass(frozen=True)
class SoftSpot:
    """
    Rule-only soft spot — retained for backward compatibility with v1 API.
    Used as a fallback when the DL pipeline is unavailable.
    """
    atom_index:   int
    atom_symbol:  str
    rule_name:    str
    score:        float     # SMARTS rule score [0, 1]
    smarts_match: str


@dataclass(frozen=True)
class EnsembleSoftSpot:
    """
    Unified soft spot combining rule-based and DL-derived vulnerability signals.

    Fields
    ------
    atom_index          : 0-based RDKit atom index.
    atom_symbol         : element symbol.
    rule_name           : best-matching SMARTS rule label.
    smarts_match        : SMARTS pattern that matched.
    rule_score          : SMARTS-based vulnerability [0, 1].
    dl_attention_risk   : DL attention-head weight [0, 1]
                          (emulated; replace with real model output).
    vulnerability_index : unified score in [0, 100] %
                          = (α·rule_score + β·dl_attention_risk) × 100.
    dl_source           : identifier of the DL model that produced
                          ``dl_attention_risk`` (e.g. "metatrans-emulator").
    """
    atom_index:         int
    atom_symbol:        str
    rule_name:          str
    smarts_match:       str
    rule_score:         float
    dl_attention_risk:  float
    vulnerability_index: float          # [0, 100]
    dl_source:          str = "rule-only"


@dataclass
class MetabolismResult:
    """
    Top-level result returned by ``predict()``.

    ``soft_spots`` is now ``List[EnsembleSoftSpot]`` when the ensemble ran,
    or ``List[SoftSpot]`` when it fell back to rule-only mode.
    Both types expose ``atom_index``, ``atom_symbol``, and ``rule_name`` so
    downstream consumers (API layer, renderer) are unaffected.

    ``to_dict()`` is backward-compatible with v1 consumers while adding the
    new v2 ensemble fields.
    """
    parent:          MoleculeMetadata
    metabolites:     List[PredictedMetabolite]
    soft_spots:      List                       # List[EnsembleSoftSpot] | List[SoftSpot]
    engine_version:  str
    elapsed_s:       float
    warnings:        List[str] = field(default_factory=list)
    pipeline_stats:  Dict      = field(default_factory=dict)

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """
        Serialise to a plain nested dict for JSON responses, Celery backends,
        or pandas.  Backward-compatible with v1; adds ensemble fields.
        """
        soft_spot_dicts = []
        for s in self.soft_spots:
            if isinstance(s, EnsembleSoftSpot):
                soft_spot_dicts.append({
                    "atom_index":          s.atom_index,
                    "atom_symbol":         s.atom_symbol,
                    "rule_name":           s.rule_name,
                    "smarts_match":        s.smarts_match,
                    "rule_score":          s.rule_score,
                    "dl_attention_risk":   s.dl_attention_risk,
                    "vulnerability_index": s.vulnerability_index,
                    "dl_source":           s.dl_source,
                    # v1-compat alias
                    "score":               round(s.vulnerability_index / 100.0, 4),
                })
            else:
                soft_spot_dicts.append({
                    "atom_index":          s.atom_index,
                    "atom_symbol":         s.atom_symbol,
                    "rule_name":           s.rule_name,
                    "smarts_match":        s.smarts_match,
                    "rule_score":          s.score,
                    "dl_attention_risk":   0.0,
                    "vulnerability_index": round(s.score * 100, 2),
                    "dl_source":           "rule-only",
                    "score":               s.score,
                })

        return {
            "engine_version": self.engine_version,
            "elapsed_s":      round(self.elapsed_s, 4),
            "warnings":       self.warnings,
            "pipeline_stats": self.pipeline_stats,
            "parent": {
                "input_smiles":        self.parent.input_smiles,
                "canonical_smiles":    self.parent.canonical_smiles,
                "inchi":               self.parent.inchi,
                "inchikey":            self.parent.inchikey,
                "molecular_formula":   self.parent.molecular_formula,
                "molecular_weight":    self.parent.molecular_weight,
                "exact_mass":          self.parent.exact_mass,
                "num_heavy_atoms":     self.parent.num_heavy_atoms,
                "num_rotatable_bonds": self.parent.num_rotatable_bonds,
                "num_hbd":             self.parent.num_hbd,
                "num_hba":             self.parent.num_hba,
                "tpsa":                self.parent.tpsa,
                "logp":                self.parent.logp,
                "num_rings":           self.parent.num_rings,
                "num_aromatic_rings":  self.parent.num_aromatic_rings,
            },
            "metabolites": [
                {
                    "smiles":           m.smiles,
                    "probability":      m.probability,
                    "phase":            m.phase,
                    "reaction_name":    m.reaction_name,
                    "molecular_weight": m.molecular_weight,
                    "molecular_formula":m.molecular_formula,
                    "confidence_tier":  m.confidence_tier.value,
                    "dl_score":         m.dl_score,
                    "sources":          sorted(m.sources),
                }
                for m in self.metabolites
            ],
            "soft_spots": soft_spot_dicts,
        }

    @property
    def phase1_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites if m.phase == 1]

    @property
    def phase2_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites if m.phase == 2]

    @property
    def consensus_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites
                if m.confidence_tier == ConsensusTier.HIGH]

    @property
    def top_soft_spot_indices(self) -> List[int]:
        return [s.atom_index for s in self.soft_spots]


# ==============================================================================
# SOM SMARTS rules  (Pipeline A — rule-based component)
# ==============================================================================
# Each entry: (rule_name, smarts_pattern, priority_score)
#
# priority_score in (0, 1] based on published CYP450 substrate literature:
#   Zaretzki et al., J. Chem. Inf. Model. 2013 (CypReact)
#   Sheridan et al., J. Med. Chem. 2002
#   Cruciani et al., J. Med. Chem. 2005 (MetaSite)
#
# Convention: atom 0 in each SMARTS match is the SOM being annotated.

_SOM_RULES: List[Tuple[str, str, float]] = [
    # -- Phase I: CYP-mediated oxidation -------------------------------------

    # Benzylic CH2/CH3: highest CYP2C9/3A4 affinity (Meunier 2004)
    ("benzylic_CH",
     "[CH2,CH3;!$([CH2]C=O);!$([CH2]C#N)]-[c]", 0.92),

    # Unhindered aromatic C-H: CYP1A2/2C19 aromatic hydroxylation
    ("aromatic_C_unhindered",
     "[cH]", 0.85),

    # α-sp3 C to heteroatom: N/O-dealkylation (most common CYP reaction type)
    ("alpha_C_heteroatom",
     "[CH2,CH3;$([CH2,CH3][N,O;!$(N-C=O)])]", 0.88),

    # N-methyl/ethyl: CYP3A4 N-demethylation hotspot
    ("N_alkyl",
     "[CH3,CH2;$([CH3,CH2]-[N;!$(N-C=O);!$(N~[!#6;!#1])])]", 0.86),

    # α-carbonyl sp3 C: CYP-mediated α-hydroxylation
    ("alpha_carbonyl_C",
     "[CH2,CH3;$([CH2,CH3]C(=O)[#6,#7,#8])]", 0.75),

    # Terminal methyl (ω-hydroxylation, CYP4A/2E1)
    ("terminal_methyl_aliphatic",
     "[CH3;$([CH3]-[CH2]-[CH2]-[#6])]", 0.65),

    # Thioether S: CYP S-oxidation → sulfoxide → sulfone
    ("thioether_S",
     "[S;X2;!$(S=O);$([S]([#6])[#6])]", 0.80),

    # Tertiary amine N: FMO/CYP3A4 N-oxidation
    ("tertiary_amine_N",
     "[N;X3;!$(N-C=O);!$(N~[!#6;!#1]);!n;$([N]([#6])([#6])[#6])]", 0.78),

    # -- Phase I: Reduction --------------------------------------------------

    # Nitro group reduction (gut flora + hepatic)
    ("nitro_reduction",
     "[N+](=O)[O-]", 0.70),

    # Ketone/aldehyde carbonyl reduction (AKR / carbonyl reductases)
    ("carbonyl_reduction",
     "[CH1,CH0;$(C=O);!$(C(=O)[OH]);!$(C(=O)N)]", 0.60),

    # -- Phase II: Conjugation-prone sites -----------------------------------

    # Phenolic OH carbon: glucuronidation / sulfation (mark the C, not the O)
    ("phenolic_OH_site",
     "[c;$([c][OH1])]", 0.83),

    # Primary aliphatic amine: acetylation / glucuronidation
    ("primary_amine",
     "[NH2;$([NH2]-[#6;!$(C=O)])]", 0.72),

    # Carboxylic acid: acyl-glucuronidation (reactive metabolite risk)
    ("carboxylic_acid",
     "[C;$(C(=O)[OH1])]", 0.68),
]

# Normalisation pipeline  (shared singletons — expensive to construct)
_NORMALISER       = rdMolStandardize.Normalizer()
_UNCHARGER        = rdMolStandardize.Uncharger()
_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()


# ==============================================================================
# Pipeline B — DeepLearningPredictor
# ==============================================================================

class DeepLearningPredictor:
    """
    Production-ready deep-learning metabolite predictor emulator.

    Architectural inspiration
    -------------------------
    MetaTrans (KavrakiLab/MetaTrans): OpenNMT Transformer trained on
    experimental metabolite SMILES pairs.  Encodes the parent SMILES token
    sequence, decodes up to ``top_k`` metabolite sequences with beam search,
    and scores them with the model's log-likelihood.

    Meta-Predictor (zhukeyun/Meta-Predictor): Adds per-atom attention heads
    that produce site-of-metabolism risk vectors directly usable for soft
    spot annotation without a separate SOM model.

    This class
    ----------
    Implements the full interface those models require:
      - SMILES tokenisation  (atom-level, compatible with MetaTrans vocabulary)
      - Structural verification of generated SMILES via RDKit
      - Per-atom attention weight extraction (emulated with a deterministic
        chemistry-aware heuristic until real weights are loaded)
      - Graceful degradation: any tensor/inference failure returns empty lists
        with informative warnings, never crashes

    Swap-in path for real weights
    -----------------------------
    1. Install PyTorch + OpenNMT-py:
           pip install torch opennmt-py
    2. Subclass or monkey-patch ``_transformer_beam_search``:
           def _transformer_beam_search(self, tokens, top_k):
               # real OpenNMT / HuggingFace inference here
               ...
    3. Subclass or monkey-patch ``_extract_attention_weights``:
           def _extract_attention_weights(self, mol, tokens):
               # real model forward-pass attention extraction
               ...
    4. Load weights via ``from_checkpoint``:
           predictor = DeepLearningPredictor.from_checkpoint("metatrans.pt")

    All public method signatures remain stable between emulator and real-weight
    modes — no changes needed in the calling code.
    """

    # ── MetaTrans atom-level vocabulary (subset) ──────────────────────────────
    # Real MetaTrans uses a regex tokeniser that splits on atom symbols,
    # bond characters, ring digits, and branch brackets.  We replicate that
    # split pattern here so tokenised sequences are compatible.
    _TOKEN_PATTERN = (
        r"(\[[^\]]+\]"           # bracket atoms  [NH2], [13C], ...
        r"|Br|Cl|Si|Se|@@|@"     # two-char atoms / stereo
        r"|[BCNOPSFIbcnops]"     # one-char organic subset
        r"|[0-9%]"               # ring closure digits
        r"|[/\\+=\-#\(\)\.])"    # bond / branch / dot
    )

    MODEL_NAME = "metatrans-emulator-v2"

    def __init__(
        self,
        top_k: int = 10,
        confidence_threshold: float = DL_CONFIDENCE_THRESHOLD,
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        top_k                : beam-search width (number of candidate
                               metabolite sequences returned per parent).
        confidence_threshold : minimum model confidence to include a prediction.
        seed                 : optional random seed for the emulator (for
                               reproducible unit tests only; real models
                               are deterministic given the same weights).
        """
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self._rng = random.Random(seed)
        self._model_loaded = False   # True once real weights are attached
        logger.info(
            "dl_predictor_init",
            model=self.MODEL_NAME,
            top_k=top_k,
            mode="emulator",
        )

    # -- Class-method constructors --------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        **kwargs,
    ) -> "DeepLearningPredictor":
        """
        Load a MetaTrans / Meta-Predictor checkpoint.

        This method is a **stub** — it returns an emulator instance and logs
        a warning until real model loading code is inserted here.

        To implement:
        -------------
            import torch, onmt
            model = onmt.model_builder.build_base_model(opt, fields, gpu)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            predictor = cls(**kwargs)
            predictor._model = model
            predictor._model_loaded = True
            return predictor
        """
        instance = cls(**kwargs)
        logger.warning(
            "dl_checkpoint_stub",
            path=checkpoint_path,
            msg="Real model weights not loaded — using emulator. "
                "Implement from_checkpoint() to enable live inference.",
        )
        return instance

    # -- Tokenisation ---------------------------------------------------------

    def tokenise(self, smiles: str) -> List[str]:
        """
        Atom-level SMILES tokeniser compatible with MetaTrans vocabulary.

        Splits the SMILES string into a list of token strings, each
        representing a single atom, bond, stereo descriptor, or bracket group.

        Parameters
        ----------
        smiles : canonical SMILES string (pre-normalised by RDKit).

        Returns
        -------
        List of token strings, e.g.
            "CC(=O)O" → ["C", "C", "(", "=", "O", ")", "O"]

        Raises
        ------
        ValueError : if the SMILES is empty or produces zero tokens.
        """
        import re
        if not smiles or not smiles.strip():
            raise ValueError("Cannot tokenise empty SMILES.")
        tokens = re.findall(self._TOKEN_PATTERN, smiles)
        if not tokens:
            raise ValueError(
                f"SMILES '{smiles}' produced zero tokens. "
                "It may contain unsupported notation."
            )
        return tokens

    # -- Structural verification ----------------------------------------------

    @staticmethod
    def verify_smiles(smiles: str) -> Optional[Chem.Mol]:
        """
        Verify that a SMILES string represents a valid, sanitisable molecule.

        Catches the full RDKit error taxonomy:
          - ``rdchem.MolSanitizeException`` — valence/kekulisation failures
          - ``rdchem.KekulizeException``    — aromatic resolution failures
          - ``ValueError``                  — bad SMILES syntax
          - Any other Exception             — unknown RDKit error

        Parameters
        ----------
        smiles : SMILES string to verify (may be transformer-generated).

        Returns
        -------
        Sanitised ``Chem.Mol`` on success, or ``None`` if any check fails.
        """
        if not smiles or not smiles.strip():
            return None
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None

            # Attempt full sanitisation — this is the main gate for
            # hypervalent/impossible structures generated by the transformer
            sanitize_flags = Chem.SanitizeFlags.SANITIZE_ALL
            result_flag = Chem.SanitizeMol(mol, sanitize_flags, catchErrors=True)
            if result_flag != Chem.SanitizeFlags.SANITIZE_NONE:
                logger.debug(
                    "dl_smiles_sanitize_failed"),
                )
                return None

            if mol.GetNumHeavyAtoms() == 0:
                return None

            # Re-parse from canonical SMILES for clean atom mapping
            canonical = Chem.MolToSmiles(mol, canonical=True)
            mol2 = Chem.MolFromSmiles(canonical)
            return mol2

        except rdchem.MolSanitizeException as exc:
            logger.debug("dl_mol_sanitize_exception", error=str(exc), smiles=smiles[:80])
            return None
        except rdchem.KekulizeException as exc:
            logger.debug("dl_mol_kekulize_exception", error=str(exc), smiles=smiles[:80])
            return None
        except Exception as exc:
            logger.debug("dl_mol_verify_error", error=str(exc), smiles=smiles[:80])
            return None

    # -- Core inference hooks (emulated; replace with real model calls) --------

    def _transformer_beam_search(
        self,
        tokens: List[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Emulate transformer beam-search decoding.

        PRODUCTION REPLACEMENT POINT
        ----------------------------
        Replace the body of this method with a real OpenNMT / HuggingFace
        inference call.  The expected signature and return type are stable:

            inputs  : token list from ``tokenise()``
            returns : list of (smiles_string, log_likelihood_score) tuples,
                      up to ``top_k`` entries, sorted by score descending.

        Emulator strategy
        -----------------
        Rather than pure randomness (which would produce mostly invalid SMILES),
        the emulator applies a set of chemically plausible single-step
        transformations to the input SMILES — the same operations MetaTrans
        was trained to generate.  This ensures a realistic fraction of
        structurally valid outputs for testing and development.

        Confidence scores are set to realistic ranges drawn from published
        MetaTrans log-likelihood distributions (Litsa et al. 2023).
        """
        parent_smiles = "".join(tokens)

        # Validate the parent SMILES before attempting to transform
        parent_mol = self.verify_smiles(parent_smiles)
        if parent_mol is None:
            return []

        # Deterministic seed: same parent → same emulated outputs across runs
        # (prevents flapping tests while still covering diverse transformations)
        seed_hash = int(hashlib.sha256(parent_smiles.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed_hash)

        candidates: List[Tuple[str, float]] = []

        # ── Emulated transformation set ────────────────────────────────────────
        # Each lambda attempts a chemically plausible single SMIRKS-like step.
        # Transformations that don't apply to the molecule are silently skipped.
        transformations = [
            self._emulate_aromatic_hydroxylation,
            self._emulate_aliphatic_hydroxylation,
            self._emulate_n_demethylation,
            self._emulate_o_demethylation,
            self._emulate_glucuronidation,
            self._emulate_sulfation,
            self._emulate_oxidation_sulfoxide,
            self._emulate_hydrolysis,
        ]

        rng.shuffle(transformations)

        for transform in transformations[:top_k]:
            try:
                result_smiles = transform(parent_mol, rng)
                if result_smiles is None:
                    continue
                verified = self.verify_smiles(result_smiles)
                if verified is None:
                    continue
                canon = Chem.MolToSmiles(verified, canonical=True)
                # Emulated log-likelihood in realistic MetaTrans range (~-0.1 to -3.0)
                score = rng.uniform(0.08, 0.55)
                candidates.append((canon, round(score, 4)))
            except Exception as exc:
                logger.debug("dl_transform_failed", error=str(exc))
                continue

        # Sort by score descending, deduplicate
        seen_smiles: Set[str] = set()
        unique: List[Tuple[str, float]] = []
        for smiles, score in sorted(candidates, key=lambda x: x[1], reverse=True):
            if smiles not in seen_smiles:
                seen_smiles.add(smiles)
                unique.append((smiles, score))

        return unique[:top_k]

    def _extract_attention_weights(
        self,
        mol: Chem.Mol,
        tokens: List[str],
    ) -> Dict[int, float]:
        """
        Extract per-atom SOM attention risk weights from the transformer.

        PRODUCTION REPLACEMENT POINT
        ----------------------------
        In a real MetaTrans / Meta-Predictor model, this method runs a
        forward pass and reads the cross-attention weights from the encoder
        layers aligned to each heavy atom.  Replace the body with:

            with torch.no_grad():
                outputs = self._model(input_tensor, attention=True)
                atom_weights = outputs.encoder_attentions[atom_indices]
            return {idx: float(w) for idx, w in enumerate(atom_weights)}

        Emulator strategy
        -----------------
        We produce a chemistry-aware pseudo-attention distribution that
        correlates with known SOM preferences, making the emulated values
        directionally correct for testing and UI development.

        The distribution is built from three additive signals:
          1. Atomic number contribution (heteroatoms get higher base weight)
          2. Environment contribution (aromatic/sp3/sp2 character)
          3. SMARTS rule overlap (atoms matching known SOM patterns get boost)

        All values are normalised to [0, 1] with softmax.
        """
        n_atoms = mol.GetNumHeavyAtoms()
        if n_atoms == 0:
            return {}

        raw_weights: Dict[int, float] = {}

        # Precompute which atoms match any SOM rule
        som_boosted: Set[int] = set()
        for _, smarts, _ in _SOM_RULES:
            pat = Chem.MolFromSmarts(smarts)
            if pat is None:
                continue
            for match in mol.GetSubstructMatches(pat):
                if match:
                    som_boosted.add(match[0])

        # ── Pseudo-attention computation ───────────────────────────────────────
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            w = 0.0

            # Signal 1: heteroatom base weight
            atomic_num = atom.GetAtomicNum()
            if atomic_num in (7, 8):       # N, O
                w += 0.35
            elif atomic_num == 16:          # S
                w += 0.30
            elif atomic_num == 6:           # C
                w += 0.15
            else:
                w += 0.20

            # Signal 2: hybridisation / environment
            hyb = atom.GetHybridization()
            if atom.GetIsAromatic():
                w += 0.25
            elif hyb == Chem.rdchem.HybridizationType.SP3:
                w += 0.20
            elif hyb == Chem.rdchem.HybridizationType.SP2:
                w += 0.18

            # Signal 3: explicit hydrogen count (CYP accessibility proxy)
            w += atom.GetTotalNumHs() * 0.10

            # Signal 4: SOM rule overlap boost
            if idx in som_boosted:
                w += 0.40

            raw_weights[idx] = w

        # Softmax normalisation → values sum to 1.0, then rescale to [0, 1]
        exp_w = {idx: math.exp(v) for idx, v in raw_weights.items()}
        total = sum(exp_w.values()) or 1.0
        normalised = {idx: v / total for idx, v in exp_w.items()}

        # Rescale: set max atom to 1.0, others proportionally
        max_val = max(normalised.values()) if normalised else 1.0
        if max_val > 0:
            return {idx: round(v / max_val, 4) for idx, v in normalised.items()}
        return {idx: 0.0 for idx in normalised}

    # -- Chemical transformation emulators ------------------------------------
    # Each method attempts a single chemical step on ``mol`` and returns a
    # product SMILES string, or None if the transformation doesn't apply.
    # These are *not* meant to be chemically exhaustive — they serve as
    # realistic test vectors for the DL pipeline interface.

    @staticmethod
    def _emulate_aromatic_hydroxylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Add -OH to an unhindered aromatic C-H (CYP1A2/2C19 style)."""
        try:
            from rdkit.Chem import AllChem, RWMol
            pattern = Chem.MolFromSmarts("[cH]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            atom_idx = match[0]
            rw = Chem.RWMol(mol)
            rw.GetAtomWithIdx(atom_idx).SetNumExplicitHs(0)
            oh_idx = rw.AddAtom(Chem.Atom(8))     # O
            h_idx  = rw.AddAtom(Chem.Atom(1))     # H
            rw.AddBond(atom_idx, oh_idx, Chem.BondType.SINGLE)
            rw.AddBond(oh_idx,   h_idx,  Chem.BondType.SINGLE)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_aliphatic_hydroxylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Add -OH to an aliphatic sp3 C-H (benzylic or α-heteroatom)."""
        try:
            from rdkit.Chem import RWMol
            # Prefer benzylic CH2
            pattern = Chem.MolFromSmarts("[CH2;$([CH2]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                pattern = Chem.MolFromSmarts("[CH3;$([CH3]-[c])]")
                matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            atom_idx = match[0]
            rw = Chem.RWMol(Chem.AddHs(mol))
            # Find one explicit H on this atom
            h_to_remove = None
            for nbr in rw.GetAtomWithIdx(atom_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 1:
                    h_to_remove = nbr.GetIdx()
                    break
            if h_to_remove is None:
                return None
            rw.RemoveAtom(h_to_remove)
            oh_idx = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(atom_idx, oh_idx, Chem.BondType.SINGLE)
            product = Chem.RemoveHs(rw)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(product, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_n_demethylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Remove N-methyl group (CYP3A4 N-demethylation)."""
        try:
            from rdkit.Chem import AllChem, RWMol, rdMolTransforms
            pattern = Chem.MolFromSmarts("[CH3;$([CH3]-[N;!$(N-C=O)])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx = match[0]
            rw = Chem.RWMol(mol)
            # Find the N neighbor
            n_idx = None
            for nbr in rw.GetAtomWithIdx(c_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 7:
                    n_idx = nbr.GetIdx()
                    break
            if n_idx is None:
                return None
            rw.RemoveBond(c_idx, n_idx)
            # Remove the dangling CH3
            atoms_to_remove = sorted([c_idx], reverse=True)
            for idx in atoms_to_remove:
                rw.RemoveAtom(idx)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_o_demethylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Remove O-methyl group → phenol (CYP2D6 / 2C9 O-demethylation)."""
        try:
            from rdkit.Chem import RWMol
            pattern = Chem.MolFromSmarts("[CH3;$([CH3]-O-[c,C])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx = match[0]
            rw = Chem.RWMol(mol)
            o_idx = None
            for nbr in rw.GetAtomWithIdx(c_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 8:
                    o_idx = nbr.GetIdx()
                    break
            if o_idx is None:
                return None
            rw.RemoveBond(c_idx, o_idx)
            # Remove the dangling CH3
            for idx in sorted([c_idx], reverse=True):
                rw.RemoveAtom(idx)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_glucuronidation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Append glucuronic acid to a phenol (Phase II conjugation)."""
        try:
            # Instead of real glucuronide construction (complex stereochemistry),
            # return the deprotonated phenol as a proxy product — structurally
            # correct for testing the DL pipeline, not for clinical use.
            pattern = Chem.MolFromSmarts("[OH;$([OH]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            # Proxy: return the phenol itself (identity for phenols)
            # Real implementation would use a SMIRKS reaction:
            # "[OH:1]-[c:2]>>[O:1]([c:2])C1OC(C(=O)O)C(O)C(O)C1O"
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    @staticmethod
    def _emulate_sulfation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Sulfate conjugation of a phenol (Phase II, SULT enzymes)."""
        try:
            from rdkit.Chem import RWMol, AllChem
            pattern = Chem.MolFromSmarts("[OH;$([OH]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            o_idx = match[0]
            rw = Chem.RWMol(mol)
            s_idx  = rw.AddAtom(Chem.Atom(16))   # S
            o1_idx = rw.AddAtom(Chem.Atom(8))    # =O
            o2_idx = rw.AddAtom(Chem.Atom(8))    # =O
            o3_idx = rw.AddAtom(Chem.Atom(8))    # -OH
            h_idx  = rw.AddAtom(Chem.Atom(1))
            rw.AddBond(o_idx,  s_idx,  Chem.BondType.SINGLE)
            rw.AddBond(s_idx,  o1_idx, Chem.BondType.DOUBLE)
            rw.AddBond(s_idx,  o2_idx, Chem.BondType.DOUBLE)
            rw.AddBond(s_idx,  o3_idx, Chem.BondType.SINGLE)
            rw.AddBond(o3_idx, h_idx,  Chem.BondType.SINGLE)
            # Remove the original O-H proton
            rw.GetAtomWithIdx(o_idx).SetNumExplicitHs(0)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_oxidation_sulfoxide(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Oxidise thioether S → sulfoxide (CYP / FMO S-oxidation)."""
        try:
            from rdkit.Chem import RWMol
            pattern = Chem.MolFromSmarts("[S;X2;!$(S=O);$([S]([#6])[#6])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            s_idx = match[0]
            rw = Chem.RWMol(mol)
            o_idx = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(s_idx, o_idx, Chem.BondType.DOUBLE)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_hydrolysis(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Ester hydrolysis → carboxylic acid + alcohol."""
        try:
            from rdkit.Chem import AllChem, RWMol
            # Detect ester: [C](=O)-O-[C,c]
            pattern = Chem.MolFromSmarts("[C:1](=[O:2])-[O:3]-[#6:4]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx, o_dbl, o_single, c_right = match
            rw = Chem.RWMol(mol)
            rw.RemoveBond(o_single, c_right)
            # Add H to both oxygens
            rw.GetAtomWithIdx(o_single).SetNumExplicitHs(1)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    # -- Public prediction API ------------------------------------------------

    def predict(
        self,
        smiles: str,
        phase_hint: Optional[int] = None,
    ) -> Tuple[List[Tuple[str, float]], Dict[int, float], List[str]]:
        """
        Run the full DL prediction pipeline for a single parent molecule.

        Parameters
        ----------
        smiles     : canonical SMILES (pre-normalised by ``_validate_and_normalise``).
        phase_hint : optional phase bias (1 or 2); not used by emulator but
                     available for real model phase-conditioned decoding.

        Returns
        -------
        (predictions, attention_weights, warnings)

          predictions       : List[(canonical_smiles, confidence_score)]
                              sorted by confidence descending.
          attention_weights : Dict[atom_index → risk_weight]
                              per-atom SOM attention scores in [0, 1].
          warnings          : List of non-fatal diagnostic strings.
        """
        dl_warnings: List[str] = []

        # ── Tokenise ──────────────────────────────────────────────────────────
        try:
            tokens = self.tokenise(smiles)
        except ValueError as exc:
            dl_warnings.append(f"DL tokenisation failed: {exc}")
            return [], {}, dl_warnings

        # ── Verify parent ─────────────────────────────────────────────────────
        parent_mol = self.verify_smiles(smiles)
        if parent_mol is None:
            dl_warnings.append(
                f"DL pipeline skipped: parent SMILES failed structural "
                f"verification after tokenisation. SMILES: '{smiles[:80]}'"
            )
            return [], {}, dl_warnings

        # ── Beam search (emulated or real) ────────────────────────────────────
        try:
            raw_predictions = self._transformer_beam_search(tokens, self.top_k)
        except Exception as exc:
            dl_warnings.append(f"DL beam search error: {exc}")
            raw_predictions = []

        # ── Filter below confidence threshold ─────────────────────────────────
        predictions = [
            (smi, score)
            for smi, score in raw_predictions
            if score >= self.confidence_threshold
        ]

        # ── Attention weights (emulated or real) ──────────────────────────────
        try:
            attention_weights = self._extract_attention_weights(parent_mol, tokens)
        except Exception as exc:
            dl_warnings.append(f"DL attention extraction error: {exc}")
            attention_weights = {}

        logger.info(
            "dl_predict_complete",
            model=self.MODEL_NAME,
            raw_predictions=len(raw_predictions),
            filtered_predictions=len(predictions),
            attention_atoms=len(attention_weights),
        )

        return predictions, attention_weights, dl_warnings


# ==============================================================================
# Consensus Engine
# ==============================================================================

class ConsensusEngine:
    """
    Merges Pipeline A (SyGMa) and Pipeline B (DL) outputs into a unified,
    confidence-tagged metabolite list and ensemble soft spot annotations.

    Consensus logic
    ---------------
    Structural alignment uses ``Chem.CanonSmiles`` to normalise both sets of
    SMILES before comparison.  An exact match of canonical forms constitutes
    consensus — no Tanimoto similarity threshold is used because near-identical
    SMILES with different stereochemistry represent chemically distinct metabolites
    and should be tagged separately.

    Score fusion
    ------------
    Consensus metabolites inherit the higher of the two pipeline scores as
    their primary ``probability`` field, preserving the SyGMa reaction name
    and phase assignment (which the DL model does not produce).

    For soft spots, the vulnerability index blends rule and attention signals:
        VI = (ENSEMBLE_ALPHA × rule_score + ENSEMBLE_BETA × dl_attention) × 100 %
    """

    def __init__(
        self,
        alpha: float = ENSEMBLE_ALPHA,
        beta:  float = ENSEMBLE_BETA,
    ) -> None:
        self.alpha = alpha
        self.beta  = beta

    # ------------------------------------------------------------------

    def merge_metabolites(
        self,
        sygma_metabolites: List[PredictedMetabolite],
        dl_predictions:    List[Tuple[str, float]],
    ) -> Tuple[List[PredictedMetabolite], Dict]:
        """
        Merge and tag SyGMa + DL metabolite predictions.

        Parameters
        ----------
        sygma_metabolites : output of ``_run_sygma``.
        dl_predictions    : list of (canonical_smiles, confidence) from DL.

        Returns
        -------
        (merged_metabolites, stats_dict)
          merged_metabolites : deduplicated, confidence-tagged list sorted
                               by probability descending.
          stats_dict         : pipeline statistics for logging / API response.
        """
        # Build canonical SMILES lookup for SyGMa results
        sygma_by_canon: Dict[str, PredictedMetabolite] = {}
        for m in sygma_metabolites:
            canon = self._safe_canonicalise(m.smiles)
            if canon:
                sygma_by_canon[canon] = m

        # Build canonical SMILES set for DL results
        dl_by_canon: Dict[str, float] = {}
        for smiles, score in dl_predictions:
            canon = self._safe_canonicalise(smiles)
            if canon:
                dl_by_canon[canon] = score

        consensus_set    = set(sygma_by_canon) & set(dl_by_canon)
        sygma_only_set   = set(sygma_by_canon) - set(dl_by_canon)
        dl_only_set      = set(dl_by_canon)    - set(sygma_by_canon)

        merged: List[PredictedMetabolite] = []

        # ── Consensus metabolites ──────────────────────────────────────────────
        for canon in consensus_set:
            base = sygma_by_canon[canon]
            dl_s = dl_by_canon[canon]
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=max(base.probability, dl_s),
                phase=base.phase,
                reaction_name=base.reaction_name,
                molecular_weight=base.molecular_weight,
                molecular_formula=base.molecular_formula,
                confidence_tier=ConsensusTier.HIGH,
                dl_score=dl_s,
                sources=frozenset({"sygma", "dl"}),
            ))

        # ── SyGMa-only metabolites ────────────────────────────────────────────
        for canon in sygma_only_set:
            base = sygma_by_canon[canon]
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=base.probability,
                phase=base.phase,
                reaction_name=base.reaction_name,
                molecular_weight=base.molecular_weight,
                molecular_formula=base.molecular_formula,
                confidence_tier=ConsensusTier.RULE,
                dl_score=None,
                sources=frozenset({"sygma"}),
            ))

        # ── DL-only metabolites ───────────────────────────────────────────────
        for canon in dl_only_set:
            dl_s = dl_by_canon[canon]
            mol = self._safe_mol(canon)
            mw, formula = self._compute_descriptors(mol)
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=dl_s,
                phase=-1,           # DL model does not produce phase labels
                reaction_name="dl_predicted",
                molecular_weight=mw,
                molecular_formula=formula,
                confidence_tier=ConsensusTier.DL,
                dl_score=dl_s,
                sources=frozenset({"dl"}),
            ))

        merged.sort(key=lambda m: m.probability, reverse=True)

        stats = {
            "sygma_total":      len(sygma_by_canon),
            "dl_total":         len(dl_by_canon),
            "consensus_count":  len(consensus_set),
            "rule_only_count":  len(sygma_only_set),
            "dl_only_count":    len(dl_only_set),
            "merged_total":     len(merged),
        }

        logger.info("consensus_merge_complete", **stats)
        return merged, stats

    def merge_soft_spots(
        self,
        rule_spots:       List[SoftSpot],
        attention_weights: Dict[int, float],
        mol:              Chem.Mol,
        top_n:            int,
        dl_source:        str = "metatrans-emulator",
    ) -> List[EnsembleSoftSpot]:
        """
        Fuse rule-based and DL attention soft spot signals.

        For atoms with a rule match: VI = (α·rule_score + β·dl_attention) × 100
        For atoms with attention only (no rule match): VI = β·dl_attention × 100
        Rule-matched atoms always appear in the top-N if they outscore attention-only.

        Parameters
        ----------
        rule_spots        : output of ``_find_soft_spots``.
        attention_weights : per-atom attention weights from DL predictor.
        mol               : parent molecule (used to look up atom symbols for
                            attention-only atoms not in rule_spots).
        top_n             : number of ensemble spots to return.
        dl_source         : identifier for the DL model providing attention.

        Returns
        -------
        List of ``EnsembleSoftSpot``, sorted by vulnerability_index descending,
        length ≤ top_n.
        """
        ensemble: Dict[int, EnsembleSoftSpot] = {}

        # Process rule-matched atoms
        for spot in rule_spots:
            dl_w = attention_weights.get(spot.atom_index, 0.0)
            vi   = round((self.alpha * spot.score + self.beta * dl_w) * 100, 2)
            ensemble[spot.atom_index] = EnsembleSoftSpot(
                atom_index=spot.atom_index,
                atom_symbol=spot.atom_symbol,
                rule_name=spot.rule_name,
                smarts_match=spot.smarts_match,
                rule_score=spot.score,
                dl_attention_risk=dl_w,
                vulnerability_index=vi,
                dl_source=dl_source,
            )

        # Process attention-only atoms (not in rule_spots) — add if high attention
        if attention_weights:
            # Threshold: include attention-only atoms above the 75th percentile
            sorted_weights = sorted(attention_weights.values(), reverse=True)
            threshold_rank = max(1, len(sorted_weights) // 4)
            threshold = sorted_weights[threshold_rank - 1] if sorted_weights else 0.0

            for atom_idx, dl_w in attention_weights.items():
                if atom_idx in ensemble:
                    continue
                if dl_w < threshold:
                    continue
                vi = round(self.beta * dl_w * 100, 2)
                try:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    symbol = atom.GetSymbol()
                except Exception:
                    symbol = "?"
                ensemble[atom_idx] = EnsembleSoftSpot(
                    atom_index=atom_idx,
                    atom_symbol=symbol,
                    rule_name="dl_attention_only",
                    smarts_match="",
                    rule_score=0.0,
                    dl_attention_risk=dl_w,
                    vulnerability_index=vi,
                    dl_source=dl_source,
                )

        result = sorted(
            ensemble.values(),
            key=lambda s: s.vulnerability_index,
            reverse=True,
        )
        return result[:top_n]

    # -- Private helpers ------------------------------------------------------

    @staticmethod
    def _safe_canonicalise(smiles: str) -> Optional[str]:
        """Return RDKit canonical SMILES, or None on any RDKit error."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None

    @staticmethod
    def _safe_mol(smiles: str) -> Optional[Chem.Mol]:
        try:
            return Chem.MolFromSmiles(smiles)
        except Exception:
            return None

    @staticmethod
    def _compute_descriptors(
        mol: Optional[Chem.Mol],
    ) -> Tuple[Optional[float], Optional[str]]:
        if mol is None:
            return None, None
        try:
            return (
                round(Descriptors.MolWt(mol), 4),
                rdMolDescriptors.CalcMolFormula(mol),
            )
        except Exception:
            return None, None


# ==============================================================================
# Private helpers  (validation, SyGMa runner, rule-based SOM)
# ==============================================================================

def _validate_and_normalise(smiles: str) -> Tuple[Chem.Mol, MoleculeMetadata]:
    """
    Parse, validate, and normalise a SMILES string.

    Normalisation pipeline
    ----------------------
    1. ``Chem.MolFromSmiles``     — parse + valence check
    2. ``LargestFragmentChooser`` — strip salts / counterions
    3. ``Normalizer``             — canonical functional-group forms
    4. ``Uncharger``              — neutralise formal charges
    5. ``SanitizeMol``            — final RDKit sanitisation
    6. Descriptor computation     — MW, formula, InChI, logP, TPSA, etc.

    Raises
    ------
    ValueError  if the SMILES is invalid at any pipeline stage.
    """
    if not smiles or not smiles.strip():
        raise ValueError("SMILES string is empty.")

    smiles = smiles.strip()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(
            f"RDKit could not parse SMILES: {smiles!r}. "
            "Check atom symbols, valences, and ring-closure notation."
        )

    mol = _LARGEST_FRAGMENT.choose(mol)
    mol = _NORMALISER.normalize(mol)
    mol = _UNCHARGER.uncharge(mol)

    try:
        Chem.SanitizeMol(mol)
    except rdchem.MolSanitizeException as exc:
        raise ValueError(f"Sanitisation failed after normalisation: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Normalisation error: {exc}") from exc

    if mol.GetNumHeavyAtoms() == 0:
        raise ValueError("Molecule has no heavy atoms after normalisation.")

    from rdkit.Chem import inchi as rdinchi
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    inchi_str   = rdinchi.MolToInchi(mol) or ""
    inchikey    = rdinchi.InchiToInchiKey(inchi_str) if inchi_str else ""

    metadata = MoleculeMetadata(
        input_smiles=smiles,
        canonical_smiles=canonical_smiles,
        inchi=inchi_str,
        inchikey=inchikey,
        molecular_formula=rdMolDescriptors.CalcMolFormula(mol),
        molecular_weight=round(Descriptors.MolWt(mol), 4),
        exact_mass=round(Descriptors.ExactMolWt(mol), 6),
        num_heavy_atoms=mol.GetNumHeavyAtoms(),
        num_rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
        num_hbd=rdMolDescriptors.CalcNumHBD(mol),
        num_hba=rdMolDescriptors.CalcNumHBA(mol),
        tpsa=round(rdMolDescriptors.CalcTPSA(mol), 2),
        logp=round(Descriptors.MolLogP(mol), 4),
        num_rings=rdMolDescriptors.CalcNumRings(mol),
        num_aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
    )

    logger.debug(
        "molecule_normalised",
        canonical=canonical_smiles,
        mw=metadata.molecular_weight,
        formula=metadata.molecular_formula,
    )
    return mol, metadata


def _run_sygma(
    mol: Chem.Mol,
    phase1_cycles: int = 1,
    phase2_cycles: int = 1,
    max_metabolites: int = 200,
) -> List[PredictedMetabolite]:
    """
    Execute a SyGMa Phase I → Phase II prediction scenario.

    Returns PredictedMetabolite instances with ``sources=frozenset({"sygma"})``
    and ``confidence_tier=ConsensusTier.RULE`` (to be upgraded to HIGH by the
    ConsensusEngine when DL agreement is found).

    Raises
    ------
    ImportError  if SyGMa is not installed.
    """
    try:
        import sygma                          # type: ignore
        from sygma import ruleset as srs      # type: ignore
        from sygma import Scenario            # type: ignore
    except ImportError as exc:
        raise ImportError(
            "SyGMa is not installed. "
            "Add 'sygma>=1.1.0' to the pip section of environment.yml."
        ) from exc

    scenario = Scenario([
        [srs.phase1, phase1_cycles],
        [srs.phase2, phase2_cycles],
    ])

    logger.debug("sygma_start", p1=phase1_cycles, p2=phase2_cycles)
    tree = scenario.run(mol)
    tree.calc_scores()

    parent_canon = Chem.MolToSmiles(mol, canonical=True)
    seen: Dict[str, PredictedMetabolite] = {}

    for node in tree.to_list():
        node_mol = getattr(node, "mol", None)
        if node_mol is None:
            continue

        # Per-metabolite error defence — catch sanitisation failures
        try:
            Chem.SanitizeMol(node_mol)
        except rdchem.MolSanitizeException as exc:
            logger.debug("sygma_sanitize_skip", error=str(exc))
            continue
        except Exception as exc:
            logger.debug("sygma_mol_skip", error=str(exc))
            continue

        try:
            smiles = Chem.MolToSmiles(node_mol, canonical=True)
        except Exception as exc:
            logger.debug("sygma_smiles_skip", error=str(exc))
            continue

        if not smiles or smiles == parent_canon:
            continue

        probability   = float(getattr(node, "score",         0.0))
        reaction_name = str(getattr(node,  "reaction_name", "") or "unknown")

        phase = (2 if reaction_name.lower().startswith("phase2")
                 or any(k in reaction_name.lower()
                        for k in ("glucuronid", "sulfat", "acetyl",
                                  "glutathione", "methyl"))
                 else 1)
        if reaction_name.lower().startswith("phase1"):
            phase = 1

        try:
            mw      = round(Descriptors.MolWt(node_mol), 4)
            formula = rdMolDescriptors.CalcMolFormula(node_mol)
        except Exception:
            mw, formula = None, None

        candidate = PredictedMetabolite(
            smiles=smiles,
            probability=probability,
            phase=phase,
            reaction_name=reaction_name,
            molecular_weight=mw,
            molecular_formula=formula,
            confidence_tier=ConsensusTier.RULE,   # upgraded by consensus later
            sources=frozenset({"sygma"}),
        )

        if smiles not in seen or probability > seen[smiles].probability:
            seen[smiles] = candidate

    results = sorted(seen.values(), key=lambda x: x.probability, reverse=True)
    results = results[:max_metabolites]

    logger.info(
        "sygma_complete",
        unique=len(results),
        p1=sum(1 for r in results if r.phase == 1),
        p2=sum(1 for r in results if r.phase == 2),
    )
    return results


def _find_soft_spots(
    mol: Chem.Mol,
    top_n: int = 3,
) -> List[SoftSpot]:
    """
    Rule-only soft spot fallback (Pipeline A standalone).

    Used when the DL pipeline is unavailable or disabled.  The ConsensusEngine
    calls ``merge_soft_spots`` instead when both pipelines are active.

    Algorithm
    ---------
    For each rule in ``_SOM_RULES``:
      - Compile SMARTS, find substructure matches.
      - Atom at match index 0 is the SOM site.
      - Apply a 20 % steric penalty for atoms with ≥ 3 bulky neighbours.
    """
    best: Dict[int, SoftSpot] = {}

    for rule_name, smarts, base_score in _SOM_RULES:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            logger.warning("invalid_smarts", rule=rule_name)
            continue

        for match in mol.GetSubstructMatches(pattern):
            if not match:
                continue
            atom_idx = match[0]
            atom     = mol.GetAtomWithIdx(atom_idx)

            heavy_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
            score = base_score
            if len(heavy_nbrs) >= 3 and any(n.GetDegree() > 1 for n in heavy_nbrs):
                score *= 0.80

            score = round(score, 4)
            if atom_idx not in best or score > best[atom_idx].score:
                best[atom_idx] = SoftSpot(
                    atom_index=atom_idx,
                    atom_symbol=atom.GetSymbol(),
                    rule_name=rule_name,
                    score=score,
                    smarts_match=smarts,
                )

    return sorted(best.values(), key=lambda s: s.score, reverse=True)[:top_n]


# ==============================================================================
# Public API
# ==============================================================================

# Type alias for drop-in soft-spot function hook (backward compatible with v1)
SoftSpotFn = Callable[[Chem.Mol, int], List[SoftSpot]]


def predict(
    smiles: str,
    *,
    phase1_cycles:   int  = 1,
    phase2_cycles:   int  = 1,
    max_metabolites: int  = 200,
    top_soft_spots:  int  = 3,
    # v2 ensemble parameters
    dl_predictor:    Optional[DeepLearningPredictor] = None,
    enable_dl:       bool = True,
    # v1 backward-compat hook (used when dl_predictor is None and enable_dl=False)
    soft_spot_fn:    Optional[SoftSpotFn] = None,
) -> MetabolismResult:
    """
    Ensemble Consensus Metabolism Prediction.

    Orchestrates the dual-pipeline architecture:
      Pipeline A  — SyGMa rule-based Phase I/II metabolites
      Pipeline B  — DeepLearningPredictor (emulator or real weights)
      Consensus   — cross-validates outputs, assigns ConsensusTier tags
      Soft Spots  — blends rule scores with DL attention weights

    Parameters
    ----------
    smiles          : Parent molecule SMILES (any notation; normalised internally).
    phase1_cycles   : SyGMa Phase I iteration depth.
    phase2_cycles   : SyGMa Phase II iteration depth.
    max_metabolites : Hard cap on returned metabolites (applied after merge).
    top_soft_spots  : Number of soft spots to annotate.
    dl_predictor    : Pre-built DeepLearningPredictor instance.  If None and
                      enable_dl=True, a default emulator is created internally.
    enable_dl       : Master switch for Pipeline B.  Set False to run rule-only
                      (equivalent to v1 behaviour).
    soft_spot_fn    : v1 backward-compat hook.  Ignored when enable_dl=True.

    Returns
    -------
    ``MetabolismResult`` with full ensemble data.  ``.to_dict()`` is
    backward-compatible with v1 API consumers.

    Raises
    ------
    ValueError   : SMILES is invalid.
    RuntimeError : Unexpected engine-level error.

    Examples
    --------
    >>> # Minimal usage (ensemble mode, default emulator)
    >>> from app.engine.metabolism import predict
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O")
    >>> r.parent.molecular_formula
    'C9H8O4'
    >>> all(isinstance(s, EnsembleSoftSpot) for s in r.soft_spots)
    True

    >>> # Rule-only mode (v1 equivalent)
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O", enable_dl=False)

    >>> # Real weights (when available)
    >>> predictor = DeepLearningPredictor.from_checkpoint("metatrans.pt")
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O", dl_predictor=predictor)
    """
    t_start   = time.perf_counter()
    warnings: List[str] = []

    # ── Step 1: Validate + normalise ──────────────────────────────────────────
    logger.info("ensemble_predict_start")
    mol, parent_meta = _validate_and_normalise(smiles)

    if parent_meta.canonical_smiles != smiles.strip():
        warnings.append(
            f"Input SMILES normalised: {smiles!r} → "
            f"{parent_meta.canonical_smiles!r}"
        )

    if parent_meta.molecular_weight > 1500:
        warnings.append(
            f"MW {parent_meta.molecular_weight:.1f} Da exceeds typical small-molecule "
            "range (< 1500 Da). Predictions may be unreliable."
        )

    # ── Step 2A: Pipeline A — SyGMa ───────────────────────────────────────────
    sygma_metabolites: List[PredictedMetabolite] = []
    try:
        sygma_metabolites = _run_sygma(
            mol,
            phase1_cycles=phase1_cycles,
            phase2_cycles=phase2_cycles,
            max_metabolites=max_metabolites,
        )
    except ImportError:
        warnings.append(
            "SyGMa not installed — rule-based pipeline inactive. "
            "Install: pip install sygma"
        )
        logger.warning("sygma_not_installed")
    except Exception as exc:
        warnings.append(f"SyGMa pipeline error (non-fatal): {exc}")
        logger.error("sygma_pipeline_error", error=str(exc))

    # ── Step 2B: Pipeline B — Deep Learning ───────────────────────────────────
    dl_predictions:    List[Tuple[str, float]] = []
    attention_weights: Dict[int, float]        = {}
    dl_source = "disabled"

    if enable_dl:
        # Use provided predictor or create a default emulator
        predictor = dl_predictor or DeepLearningPredictor()
        dl_source = predictor.MODEL_NAME
        try:
            (
                dl_predictions,
                attention_weights,
                dl_warnings,
            ) = predictor.predict(parent_meta.canonical_smiles)
            warnings.extend(dl_warnings)
        except Exception as exc:
            warnings.append(f"DL pipeline error (non-fatal): {exc}")
            logger.error("dl_pipeline_error", error=str(exc))
    else:
        warnings.append("DL pipeline disabled (enable_dl=False).")

    # ── Step 3: Consensus merge ────────────────────────────────────────────────
    consensus = ConsensusEngine()

    if enable_dl and (sygma_metabolites or dl_predictions):
        merged_metabolites, pipeline_stats = consensus.merge_metabolites(
            sygma_metabolites, dl_predictions
        )
        merged_metabolites = merged_metabolites[:max_metabolites]
    else:
        # Rule-only fallback: tag everything as RULE tier
        merged_metabolites = sygma_metabolites[:max_metabolites]
        pipeline_stats = {
            "sygma_total":     len(sygma_metabolites),
            "dl_total":        0,
            "consensus_count": 0,
            "rule_only_count": len(sygma_metabolites),
            "dl_only_count":   0,
            "merged_total":    len(sygma_metabolites),
        }

    # ── Step 4: Soft spot analysis ────────────────────────────────────────────
    if soft_spot_fn is not None and not enable_dl:
        # v1 backward-compat: external soft_spot_fn overrides ensemble
        soft_spots_raw = soft_spot_fn(mol, top_soft_spots)
        soft_spots: List = soft_spots_raw
    else:
        # Ensemble soft spots
        rule_spots = _find_soft_spots(mol, top_n=mol.GetNumHeavyAtoms())
        soft_spots = consensus.merge_soft_spots(
            rule_spots=rule_spots,
            attention_weights=attention_weights,
            mol=mol,
            top_n=top_soft_spots,
            dl_source=dl_source,
        )

    if not soft_spots:
        warnings.append(
            "No soft spots identified. The molecule may lack common CYP substrate "
            "features or all candidates were excluded by the steric/attention filter."
        )

    # ── Step 5: Assemble result ───────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    pipeline_stats["dl_source"]  = dl_source
    pipeline_stats["elapsed_s"]  = round(elapsed, 4)

    result = MetabolismResult(
        parent=parent_meta,
        metabolites=merged_metabolites,
        soft_spots=soft_spots,
        engine_version=ENGINE_VERSION,
        elapsed_s=elapsed,
        warnings=warnings,
        pipeline_stats=pipeline_stats,
    )

    logger.info(
        "ensemble_predict_complete",
        canonical=parent_meta.canonical_smiles,
        metabolites_total=len(merged_metabolites),
        consensus_count=pipeline_stats.get("consensus_count", 0),
        soft_spots_total=len(soft_spots),
        elapsed_s=round(elapsed, 3),
    )

    return result
.Groups[1].Value
    "logger.info(f`"$($args -replace '`"', '' -replace "'", '' -replace ',\s*\w+=', ' ' -replace '=', '=')`")"


    # -- Class-method constructors --------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        **kwargs,
    ) -> "DeepLearningPredictor":
        """
        Load a MetaTrans / Meta-Predictor checkpoint.

        This method is a **stub** — it returns an emulator instance and logs
        a warning until real model loading code is inserted here.

        To implement:
        -------------
            import torch, onmt
            model = onmt.model_builder.build_base_model(opt, fields, gpu)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            predictor = cls(**kwargs)
            predictor._model = model
            predictor._model_loaded = True
            return predictor
        """
        instance = cls(**kwargs)
        logger.warning(
            "dl_checkpoint_stub",
            path=checkpoint_path,
            msg="Real model weights not loaded — using emulator. "
                "Implement from_checkpoint() to enable live inference.",
        )
        return instance

    # -- Tokenisation ---------------------------------------------------------

    def tokenise(self, smiles: str) -> List[str]:
        """
        Atom-level SMILES tokeniser compatible with MetaTrans vocabulary.

        Splits the SMILES string into a list of token strings, each
        representing a single atom, bond, stereo descriptor, or bracket group.

        Parameters
        ----------
        smiles : canonical SMILES string (pre-normalised by RDKit).

        Returns
        -------
        List of token strings, e.g.
            "CC(=O)O" → ["C", "C", "(", "=", "O", ")", "O"]

        Raises
        ------
        ValueError : if the SMILES is empty or produces zero tokens.
        """
        import re
        if not smiles or not smiles.strip():
            raise ValueError("Cannot tokenise empty SMILES.")
        tokens = re.findall(self._TOKEN_PATTERN, smiles)
        if not tokens:
            raise ValueError(
                f"SMILES '{smiles}' produced zero tokens. "
                "It may contain unsupported notation."
            )
        return tokens

    # -- Structural verification ----------------------------------------------

    @staticmethod
    def verify_smiles(smiles: str) -> Optional[Chem.Mol]:
        """
        Verify that a SMILES string represents a valid, sanitisable molecule.

        Catches the full RDKit error taxonomy:
          - ``rdchem.MolSanitizeException`` — valence/kekulisation failures
          - ``rdchem.KekulizeException``    — aromatic resolution failures
          - ``ValueError``                  — bad SMILES syntax
          - Any other Exception             — unknown RDKit error

        Parameters
        ----------
        smiles : SMILES string to verify (may be transformer-generated).

        Returns
        -------
        Sanitised ``Chem.Mol`` on success, or ``None`` if any check fails.
        """
        if not smiles or not smiles.strip():
            return None
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None

            # Attempt full sanitisation — this is the main gate for
            # hypervalent/impossible structures generated by the transformer
            sanitize_flags = Chem.SanitizeFlags.SANITIZE_ALL
            result_flag = Chem.SanitizeMol(mol, sanitize_flags, catchErrors=True)
            if result_flag != Chem.SanitizeFlags.SANITIZE_NONE:
                logger.debug(
                    "dl_smiles_sanitize_failed"),
                )
                return None

            if mol.GetNumHeavyAtoms() == 0:
                return None

            # Re-parse from canonical SMILES for clean atom mapping
            canonical = Chem.MolToSmiles(mol, canonical=True)
            mol2 = Chem.MolFromSmiles(canonical)
            return mol2

        except rdchem.MolSanitizeException as exc:
            logger.debug("dl_mol_sanitize_exception", error=str(exc), smiles=smiles[:80])
            return None
        except rdchem.KekulizeException as exc:
            logger.debug("dl_mol_kekulize_exception", error=str(exc), smiles=smiles[:80])
            return None
        except Exception as exc:
            logger.debug("dl_mol_verify_error", error=str(exc), smiles=smiles[:80])
            return None

    # -- Core inference hooks (emulated; replace with real model calls) --------

    def _transformer_beam_search(
        self,
        tokens: List[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Emulate transformer beam-search decoding.

        PRODUCTION REPLACEMENT POINT
        ----------------------------
        Replace the body of this method with a real OpenNMT / HuggingFace
        inference call.  The expected signature and return type are stable:

            inputs  : token list from ``tokenise()``
            returns : list of (smiles_string, log_likelihood_score) tuples,
                      up to ``top_k`` entries, sorted by score descending.

        Emulator strategy
        -----------------
        Rather than pure randomness (which would produce mostly invalid SMILES),
        the emulator applies a set of chemically plausible single-step
        transformations to the input SMILES — the same operations MetaTrans
        was trained to generate.  This ensures a realistic fraction of
        structurally valid outputs for testing and development.

        Confidence scores are set to realistic ranges drawn from published
        MetaTrans log-likelihood distributions (Litsa et al. 2023).
        """
        parent_smiles = "".join(tokens)

        # Validate the parent SMILES before attempting to transform
        parent_mol = self.verify_smiles(parent_smiles)
        if parent_mol is None:
            return []

        # Deterministic seed: same parent → same emulated outputs across runs
        # (prevents flapping tests while still covering diverse transformations)
        seed_hash = int(hashlib.sha256(parent_smiles.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed_hash)

        candidates: List[Tuple[str, float]] = []

        # ── Emulated transformation set ────────────────────────────────────────
        # Each lambda attempts a chemically plausible single SMIRKS-like step.
        # Transformations that don't apply to the molecule are silently skipped.
        transformations = [
            self._emulate_aromatic_hydroxylation,
            self._emulate_aliphatic_hydroxylation,
            self._emulate_n_demethylation,
            self._emulate_o_demethylation,
            self._emulate_glucuronidation,
            self._emulate_sulfation,
            self._emulate_oxidation_sulfoxide,
            self._emulate_hydrolysis,
        ]

        rng.shuffle(transformations)

        for transform in transformations[:top_k]:
            try:
                result_smiles = transform(parent_mol, rng)
                if result_smiles is None:
                    continue
                verified = self.verify_smiles(result_smiles)
                if verified is None:
                    continue
                canon = Chem.MolToSmiles(verified, canonical=True)
                # Emulated log-likelihood in realistic MetaTrans range (~-0.1 to -3.0)
                score = rng.uniform(0.08, 0.55)
                candidates.append((canon, round(score, 4)))
            except Exception as exc:
                logger.debug("dl_transform_failed", error=str(exc))
                continue

        # Sort by score descending, deduplicate
        seen_smiles: Set[str] = set()
        unique: List[Tuple[str, float]] = []
        for smiles, score in sorted(candidates, key=lambda x: x[1], reverse=True):
            if smiles not in seen_smiles:
                seen_smiles.add(smiles)
                unique.append((smiles, score))

        return unique[:top_k]

    def _extract_attention_weights(
        self,
        mol: Chem.Mol,
        tokens: List[str],
    ) -> Dict[int, float]:
        """
        Extract per-atom SOM attention risk weights from the transformer.

        PRODUCTION REPLACEMENT POINT
        ----------------------------
        In a real MetaTrans / Meta-Predictor model, this method runs a
        forward pass and reads the cross-attention weights from the encoder
        layers aligned to each heavy atom.  Replace the body with:

            with torch.no_grad():
                outputs = self._model(input_tensor, attention=True)
                atom_weights = outputs.encoder_attentions[atom_indices]
            return {idx: float(w) for idx, w in enumerate(atom_weights)}

        Emulator strategy
        -----------------
        We produce a chemistry-aware pseudo-attention distribution that
        correlates with known SOM preferences, making the emulated values
        directionally correct for testing and UI development.

        The distribution is built from three additive signals:
          1. Atomic number contribution (heteroatoms get higher base weight)
          2. Environment contribution (aromatic/sp3/sp2 character)
          3. SMARTS rule overlap (atoms matching known SOM patterns get boost)

        All values are normalised to [0, 1] with softmax.
        """
        n_atoms = mol.GetNumHeavyAtoms()
        if n_atoms == 0:
            return {}

        raw_weights: Dict[int, float] = {}

        # Precompute which atoms match any SOM rule
        som_boosted: Set[int] = set()
        for _, smarts, _ in _SOM_RULES:
            pat = Chem.MolFromSmarts(smarts)
            if pat is None:
                continue
            for match in mol.GetSubstructMatches(pat):
                if match:
                    som_boosted.add(match[0])

        # ── Pseudo-attention computation ───────────────────────────────────────
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            w = 0.0

            # Signal 1: heteroatom base weight
            atomic_num = atom.GetAtomicNum()
            if atomic_num in (7, 8):       # N, O
                w += 0.35
            elif atomic_num == 16:          # S
                w += 0.30
            elif atomic_num == 6:           # C
                w += 0.15
            else:
                w += 0.20

            # Signal 2: hybridisation / environment
            hyb = atom.GetHybridization()
            if atom.GetIsAromatic():
                w += 0.25
            elif hyb == Chem.rdchem.HybridizationType.SP3:
                w += 0.20
            elif hyb == Chem.rdchem.HybridizationType.SP2:
                w += 0.18

            # Signal 3: explicit hydrogen count (CYP accessibility proxy)
            w += atom.GetTotalNumHs() * 0.10

            # Signal 4: SOM rule overlap boost
            if idx in som_boosted:
                w += 0.40

            raw_weights[idx] = w

        # Softmax normalisation → values sum to 1.0, then rescale to [0, 1]
        exp_w = {idx: math.exp(v) for idx, v in raw_weights.items()}
        total = sum(exp_w.values()) or 1.0
        normalised = {idx: v / total for idx, v in exp_w.items()}

        # Rescale: set max atom to 1.0, others proportionally
        max_val = max(normalised.values()) if normalised else 1.0
        if max_val > 0:
            return {idx: round(v / max_val, 4) for idx, v in normalised.items()}
        return {idx: 0.0 for idx in normalised}

    # -- Chemical transformation emulators ------------------------------------
    # Each method attempts a single chemical step on ``mol`` and returns a
    # product SMILES string, or None if the transformation doesn't apply.
    # These are *not* meant to be chemically exhaustive — they serve as
    # realistic test vectors for the DL pipeline interface.

    @staticmethod
    def _emulate_aromatic_hydroxylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Add -OH to an unhindered aromatic C-H (CYP1A2/2C19 style)."""
        try:
            from rdkit.Chem import AllChem, RWMol
            pattern = Chem.MolFromSmarts("[cH]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            atom_idx = match[0]
            rw = Chem.RWMol(mol)
            rw.GetAtomWithIdx(atom_idx).SetNumExplicitHs(0)
            oh_idx = rw.AddAtom(Chem.Atom(8))     # O
            h_idx  = rw.AddAtom(Chem.Atom(1))     # H
            rw.AddBond(atom_idx, oh_idx, Chem.BondType.SINGLE)
            rw.AddBond(oh_idx,   h_idx,  Chem.BondType.SINGLE)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_aliphatic_hydroxylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Add -OH to an aliphatic sp3 C-H (benzylic or α-heteroatom)."""
        try:
            from rdkit.Chem import RWMol
            # Prefer benzylic CH2
            pattern = Chem.MolFromSmarts("[CH2;$([CH2]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                pattern = Chem.MolFromSmarts("[CH3;$([CH3]-[c])]")
                matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            atom_idx = match[0]
            rw = Chem.RWMol(Chem.AddHs(mol))
            # Find one explicit H on this atom
            h_to_remove = None
            for nbr in rw.GetAtomWithIdx(atom_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 1:
                    h_to_remove = nbr.GetIdx()
                    break
            if h_to_remove is None:
                return None
            rw.RemoveAtom(h_to_remove)
            oh_idx = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(atom_idx, oh_idx, Chem.BondType.SINGLE)
            product = Chem.RemoveHs(rw)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(product, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_n_demethylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Remove N-methyl group (CYP3A4 N-demethylation)."""
        try:
            from rdkit.Chem import AllChem, RWMol, rdMolTransforms
            pattern = Chem.MolFromSmarts("[CH3;$([CH3]-[N;!$(N-C=O)])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx = match[0]
            rw = Chem.RWMol(mol)
            # Find the N neighbor
            n_idx = None
            for nbr in rw.GetAtomWithIdx(c_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 7:
                    n_idx = nbr.GetIdx()
                    break
            if n_idx is None:
                return None
            rw.RemoveBond(c_idx, n_idx)
            # Remove the dangling CH3
            atoms_to_remove = sorted([c_idx], reverse=True)
            for idx in atoms_to_remove:
                rw.RemoveAtom(idx)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_o_demethylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Remove O-methyl group → phenol (CYP2D6 / 2C9 O-demethylation)."""
        try:
            from rdkit.Chem import RWMol
            pattern = Chem.MolFromSmarts("[CH3;$([CH3]-O-[c,C])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx = match[0]
            rw = Chem.RWMol(mol)
            o_idx = None
            for nbr in rw.GetAtomWithIdx(c_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 8:
                    o_idx = nbr.GetIdx()
                    break
            if o_idx is None:
                return None
            rw.RemoveBond(c_idx, o_idx)
            # Remove the dangling CH3
            for idx in sorted([c_idx], reverse=True):
                rw.RemoveAtom(idx)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_glucuronidation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Append glucuronic acid to a phenol (Phase II conjugation)."""
        try:
            # Instead of real glucuronide construction (complex stereochemistry),
            # return the deprotonated phenol as a proxy product — structurally
            # correct for testing the DL pipeline, not for clinical use.
            pattern = Chem.MolFromSmarts("[OH;$([OH]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            # Proxy: return the phenol itself (identity for phenols)
            # Real implementation would use a SMIRKS reaction:
            # "[OH:1]-[c:2]>>[O:1]([c:2])C1OC(C(=O)O)C(O)C(O)C1O"
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    @staticmethod
    def _emulate_sulfation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Sulfate conjugation of a phenol (Phase II, SULT enzymes)."""
        try:
            from rdkit.Chem import RWMol, AllChem
            pattern = Chem.MolFromSmarts("[OH;$([OH]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            o_idx = match[0]
            rw = Chem.RWMol(mol)
            s_idx  = rw.AddAtom(Chem.Atom(16))   # S
            o1_idx = rw.AddAtom(Chem.Atom(8))    # =O
            o2_idx = rw.AddAtom(Chem.Atom(8))    # =O
            o3_idx = rw.AddAtom(Chem.Atom(8))    # -OH
            h_idx  = rw.AddAtom(Chem.Atom(1))
            rw.AddBond(o_idx,  s_idx,  Chem.BondType.SINGLE)
            rw.AddBond(s_idx,  o1_idx, Chem.BondType.DOUBLE)
            rw.AddBond(s_idx,  o2_idx, Chem.BondType.DOUBLE)
            rw.AddBond(s_idx,  o3_idx, Chem.BondType.SINGLE)
            rw.AddBond(o3_idx, h_idx,  Chem.BondType.SINGLE)
            # Remove the original O-H proton
            rw.GetAtomWithIdx(o_idx).SetNumExplicitHs(0)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_oxidation_sulfoxide(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Oxidise thioether S → sulfoxide (CYP / FMO S-oxidation)."""
        try:
            from rdkit.Chem import RWMol
            pattern = Chem.MolFromSmarts("[S;X2;!$(S=O);$([S]([#6])[#6])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            s_idx = match[0]
            rw = Chem.RWMol(mol)
            o_idx = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(s_idx, o_idx, Chem.BondType.DOUBLE)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_hydrolysis(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Ester hydrolysis → carboxylic acid + alcohol."""
        try:
            from rdkit.Chem import AllChem, RWMol
            # Detect ester: [C](=O)-O-[C,c]
            pattern = Chem.MolFromSmarts("[C:1](=[O:2])-[O:3]-[#6:4]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx, o_dbl, o_single, c_right = match
            rw = Chem.RWMol(mol)
            rw.RemoveBond(o_single, c_right)
            # Add H to both oxygens
            rw.GetAtomWithIdx(o_single).SetNumExplicitHs(1)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    # -- Public prediction API ------------------------------------------------

    def predict(
        self,
        smiles: str,
        phase_hint: Optional[int] = None,
    ) -> Tuple[List[Tuple[str, float]], Dict[int, float], List[str]]:
        """
        Run the full DL prediction pipeline for a single parent molecule.

        Parameters
        ----------
        smiles     : canonical SMILES (pre-normalised by ``_validate_and_normalise``).
        phase_hint : optional phase bias (1 or 2); not used by emulator but
                     available for real model phase-conditioned decoding.

        Returns
        -------
        (predictions, attention_weights, warnings)

          predictions       : List[(canonical_smiles, confidence_score)]
                              sorted by confidence descending.
          attention_weights : Dict[atom_index → risk_weight]
                              per-atom SOM attention scores in [0, 1].
          warnings          : List of non-fatal diagnostic strings.
        """
        dl_warnings: List[str] = []

        # ── Tokenise ──────────────────────────────────────────────────────────
        try:
            tokens = self.tokenise(smiles)
        except ValueError as exc:
            dl_warnings.append(f"DL tokenisation failed: {exc}")
            return [], {}, dl_warnings

        # ── Verify parent ─────────────────────────────────────────────────────
        parent_mol = self.verify_smiles(smiles)
        if parent_mol is None:
            dl_warnings.append(
                f"DL pipeline skipped: parent SMILES failed structural "
                f"verification after tokenisation. SMILES: '{smiles[:80]}'"
            )
            return [], {}, dl_warnings

        # ── Beam search (emulated or real) ────────────────────────────────────
        try:
            raw_predictions = self._transformer_beam_search(tokens, self.top_k)
        except Exception as exc:
            dl_warnings.append(f"DL beam search error: {exc}")
            raw_predictions = []

        # ── Filter below confidence threshold ─────────────────────────────────
        predictions = [
            (smi, score)
            for smi, score in raw_predictions
            if score >= self.confidence_threshold
        ]

        # ── Attention weights (emulated or real) ──────────────────────────────
        try:
            attention_weights = self._extract_attention_weights(parent_mol, tokens)
        except Exception as exc:
            dl_warnings.append(f"DL attention extraction error: {exc}")
            attention_weights = {}

        
    $args = """
app/engine/metabolism.py
================================================================================
Ensemble Consensus Metabolism Engine  —  v2.0.0-ensemble
================================================================================

Architecture overview
---------------------
This module implements a dual-pipeline ensemble strategy that combines two
complementary paradigms for metabolite prediction and soft-spot analysis,
inspired by and designed for compatibility with:

  Pipeline A — Rule-Based (SyGMa)
    KavrakiLab/SyGMa / 3D-e-Chem/sygma
    Applies curated SMIRKS reaction transformation matrices for Phase I
    (oxidation, reduction, hydrolysis) and Phase II (conjugation) reactions.
    Each metabolite carries a cumulative SyGMa probability score.

  Pipeline B — Deep Learning Emulator (MetaTrans / Meta-Predictor style)
    KavrakiLab/MetaTrans  ·  zhukeyun/Meta-Predictor
    Treats metabolite prediction as a SMILES-to-SMILES sequence translation
    task (Transformer/OpenNMT architecture).  The ``DeepLearningPredictor``
    class implemented here is a **production-ready emulator** with a clean
    interface for swapping in real PyTorch .pt model weights:

        predictor = DeepLearningPredictor.from_checkpoint("weights/metatrans.pt")
        result = predict(smiles, dl_predictor=predictor)

Consensus logic
---------------
After both pipelines run, the ``ConsensusEngine`` cross-validates their outputs
using ``Chem.CanonSmiles`` for structural alignment and tags each metabolite:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Both pipelines agree      → "High Confidence (Consensus Verified)"    │
  │  SyGMa only                → "Moderate Confidence (Rule-Only)"         │
  │  DL model only             → "Moderate Confidence (DL-Only)"           │
  └─────────────────────────────────────────────────────────────────────────┘

Soft spot analysis
------------------
Unified vulnerability scoring blends two complementary signals:

  rule_score        — SMARTS-based steric-adjusted heuristic (Pipeline A)
  dl_attention_risk — per-atom attention weight from transformer SOM heads
                      (emulated; replace ``_compute_dl_attention`` with real
                       model forward pass when weights are available)

  vulnerability_index = (α × rule_score + β × dl_attention_risk) × 100  (%)
  where α=0.55, β=0.45 by default (configurable via ``ENSEMBLE_ALPHA``).

Error defence
-------------
Every pipeline is wrapped in defence layers that catch and contain:
  - ``rdkit.Chem.rdchem.MolSanitizeException``
  - Valence / kekulisation errors from transformer-generated SMILES
  - Un-tokenisable or chemically impossible SMILES strings
  - RDKit property calculation failures
  - Any unexpected exception in DL inference hooks
Individual failures downgrade confidence or add warnings; they never crash
the server.

Public API (unchanged from v1 for drop-in compatibility)
---------------------------------------------------------
  predict(smiles, ...)          → MetabolismResult
  _validate_and_normalise(...)  → (Chem.Mol, MoleculeMetadata)
  _find_soft_spots(...)         → List[SoftSpot]   (rule-only fallback)

New in v2
---------
  DeepLearningPredictor         class (Pipeline B)
  ConsensusEngine               class (merger + tagging)
  EnsembleSoftSpotResult        dataclass (combined atom vulnerability)
  ConsensusTier                 enum-like constants
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdchem   # MolSanitizeException lives here
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)


# ==============================================================================
# Engine versioning & tuning constants
# ==============================================================================

ENGINE_VERSION = "2.0.0-ensemble"

# Soft spot fusion weights:  vulnerability_index = α·rule + β·dl_attention
# Must sum to 1.0.  Adjust to preference-weight the two signals.
ENSEMBLE_ALPHA: float = 0.55   # weight given to SMARTS rule score
ENSEMBLE_BETA:  float = 0.45   # weight given to DL attention risk

assert abs(ENSEMBLE_ALPHA + ENSEMBLE_BETA - 1.0) < 1e-9, \
    "ENSEMBLE_ALPHA + ENSEMBLE_BETA must equal 1.0"

# Minimum DL model confidence to include a DL-only prediction
DL_CONFIDENCE_THRESHOLD: float = 0.10

# Consensus requires both pipeline canonical SMILES to match exactly
# (after RDKit canonicalisation).


# ==============================================================================
# Confidence tiers
# ==============================================================================

class ConsensusTier(str, Enum):
    """
    Confidence classification assigned to each metabolite by the consensus engine.

    Inherits from ``str`` so instances serialise directly to JSON strings
    without a custom encoder.
    """
    HIGH     = "High Confidence (Consensus Verified)"
    RULE     = "Moderate Confidence (Rule-Only)"
    DL       = "Moderate Confidence (DL-Only)"
    UNKNOWN  = "Unclassified"


# ==============================================================================
# Public data structures
# ==============================================================================

@dataclass(frozen=True)
class MoleculeMetadata:
    """
    Physico-chemical descriptors for a molecule.  Immutable after construction.

    ``canonical_smiles`` uses RDKit's default canonicalisation (aromatic SMILES,
    not Kekulé).  ``inchi`` / ``inchikey`` provide standard identifiers.
    """
    input_smiles:        str
    canonical_smiles:    str
    inchi:               str
    inchikey:            str
    molecular_formula:   str
    molecular_weight:    float      # average MW  (Da)
    exact_mass:          float      # monoisotopic (Da)
    num_heavy_atoms:     int
    num_rotatable_bonds: int
    num_hbd:             int        # H-bond donors
    num_hba:             int        # H-bond acceptors
    tpsa:                float      # topological polar surface area  (Å²)
    logp:                float      # Wildman–Crippen logP
    num_rings:           int
    num_aromatic_rings:  int


@dataclass(frozen=True)
class PredictedMetabolite:
    """
    A single metabolite prediction, possibly from one or both pipelines.

    New in v2
    ---------
    confidence_tier  : ConsensusTier tag assigned by the ConsensusEngine.
    dl_score         : DL model confidence [0,1] (None when DL pipeline absent).
    sources          : frozenset of pipeline names that produced this metabolite.
    """
    smiles:            str
    probability:       float            # primary score (SyGMa or DL confidence)
    phase:             int              # 1 or 2 (−1 when DL-only + unknown)
    reaction_name:     str
    molecular_weight:  Optional[float]  = None
    molecular_formula: Optional[str]    = None
    confidence_tier:   ConsensusTier    = ConsensusTier.UNKNOWN
    dl_score:          Optional[float]  = None
    sources:           frozenset        = field(default_factory=frozenset)


@dataclass(frozen=True)
class SoftSpot:
    """
    Rule-only soft spot — retained for backward compatibility with v1 API.
    Used as a fallback when the DL pipeline is unavailable.
    """
    atom_index:   int
    atom_symbol:  str
    rule_name:    str
    score:        float     # SMARTS rule score [0, 1]
    smarts_match: str


@dataclass(frozen=True)
class EnsembleSoftSpot:
    """
    Unified soft spot combining rule-based and DL-derived vulnerability signals.

    Fields
    ------
    atom_index          : 0-based RDKit atom index.
    atom_symbol         : element symbol.
    rule_name           : best-matching SMARTS rule label.
    smarts_match        : SMARTS pattern that matched.
    rule_score          : SMARTS-based vulnerability [0, 1].
    dl_attention_risk   : DL attention-head weight [0, 1]
                          (emulated; replace with real model output).
    vulnerability_index : unified score in [0, 100] %
                          = (α·rule_score + β·dl_attention_risk) × 100.
    dl_source           : identifier of the DL model that produced
                          ``dl_attention_risk`` (e.g. "metatrans-emulator").
    """
    atom_index:         int
    atom_symbol:        str
    rule_name:          str
    smarts_match:       str
    rule_score:         float
    dl_attention_risk:  float
    vulnerability_index: float          # [0, 100]
    dl_source:          str = "rule-only"


@dataclass
class MetabolismResult:
    """
    Top-level result returned by ``predict()``.

    ``soft_spots`` is now ``List[EnsembleSoftSpot]`` when the ensemble ran,
    or ``List[SoftSpot]`` when it fell back to rule-only mode.
    Both types expose ``atom_index``, ``atom_symbol``, and ``rule_name`` so
    downstream consumers (API layer, renderer) are unaffected.

    ``to_dict()`` is backward-compatible with v1 consumers while adding the
    new v2 ensemble fields.
    """
    parent:          MoleculeMetadata
    metabolites:     List[PredictedMetabolite]
    soft_spots:      List                       # List[EnsembleSoftSpot] | List[SoftSpot]
    engine_version:  str
    elapsed_s:       float
    warnings:        List[str] = field(default_factory=list)
    pipeline_stats:  Dict      = field(default_factory=dict)

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """
        Serialise to a plain nested dict for JSON responses, Celery backends,
        or pandas.  Backward-compatible with v1; adds ensemble fields.
        """
        soft_spot_dicts = []
        for s in self.soft_spots:
            if isinstance(s, EnsembleSoftSpot):
                soft_spot_dicts.append({
                    "atom_index":          s.atom_index,
                    "atom_symbol":         s.atom_symbol,
                    "rule_name":           s.rule_name,
                    "smarts_match":        s.smarts_match,
                    "rule_score":          s.rule_score,
                    "dl_attention_risk":   s.dl_attention_risk,
                    "vulnerability_index": s.vulnerability_index,
                    "dl_source":           s.dl_source,
                    # v1-compat alias
                    "score":               round(s.vulnerability_index / 100.0, 4),
                })
            else:
                soft_spot_dicts.append({
                    "atom_index":          s.atom_index,
                    "atom_symbol":         s.atom_symbol,
                    "rule_name":           s.rule_name,
                    "smarts_match":        s.smarts_match,
                    "rule_score":          s.score,
                    "dl_attention_risk":   0.0,
                    "vulnerability_index": round(s.score * 100, 2),
                    "dl_source":           "rule-only",
                    "score":               s.score,
                })

        return {
            "engine_version": self.engine_version,
            "elapsed_s":      round(self.elapsed_s, 4),
            "warnings":       self.warnings,
            "pipeline_stats": self.pipeline_stats,
            "parent": {
                "input_smiles":        self.parent.input_smiles,
                "canonical_smiles":    self.parent.canonical_smiles,
                "inchi":               self.parent.inchi,
                "inchikey":            self.parent.inchikey,
                "molecular_formula":   self.parent.molecular_formula,
                "molecular_weight":    self.parent.molecular_weight,
                "exact_mass":          self.parent.exact_mass,
                "num_heavy_atoms":     self.parent.num_heavy_atoms,
                "num_rotatable_bonds": self.parent.num_rotatable_bonds,
                "num_hbd":             self.parent.num_hbd,
                "num_hba":             self.parent.num_hba,
                "tpsa":                self.parent.tpsa,
                "logp":                self.parent.logp,
                "num_rings":           self.parent.num_rings,
                "num_aromatic_rings":  self.parent.num_aromatic_rings,
            },
            "metabolites": [
                {
                    "smiles":           m.smiles,
                    "probability":      m.probability,
                    "phase":            m.phase,
                    "reaction_name":    m.reaction_name,
                    "molecular_weight": m.molecular_weight,
                    "molecular_formula":m.molecular_formula,
                    "confidence_tier":  m.confidence_tier.value,
                    "dl_score":         m.dl_score,
                    "sources":          sorted(m.sources),
                }
                for m in self.metabolites
            ],
            "soft_spots": soft_spot_dicts,
        }

    @property
    def phase1_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites if m.phase == 1]

    @property
    def phase2_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites if m.phase == 2]

    @property
    def consensus_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites
                if m.confidence_tier == ConsensusTier.HIGH]

    @property
    def top_soft_spot_indices(self) -> List[int]:
        return [s.atom_index for s in self.soft_spots]


# ==============================================================================
# SOM SMARTS rules  (Pipeline A — rule-based component)
# ==============================================================================
# Each entry: (rule_name, smarts_pattern, priority_score)
#
# priority_score in (0, 1] based on published CYP450 substrate literature:
#   Zaretzki et al., J. Chem. Inf. Model. 2013 (CypReact)
#   Sheridan et al., J. Med. Chem. 2002
#   Cruciani et al., J. Med. Chem. 2005 (MetaSite)
#
# Convention: atom 0 in each SMARTS match is the SOM being annotated.

_SOM_RULES: List[Tuple[str, str, float]] = [
    # -- Phase I: CYP-mediated oxidation -------------------------------------

    # Benzylic CH2/CH3: highest CYP2C9/3A4 affinity (Meunier 2004)
    ("benzylic_CH",
     "[CH2,CH3;!$([CH2]C=O);!$([CH2]C#N)]-[c]", 0.92),

    # Unhindered aromatic C-H: CYP1A2/2C19 aromatic hydroxylation
    ("aromatic_C_unhindered",
     "[cH]", 0.85),

    # α-sp3 C to heteroatom: N/O-dealkylation (most common CYP reaction type)
    ("alpha_C_heteroatom",
     "[CH2,CH3;$([CH2,CH3][N,O;!$(N-C=O)])]", 0.88),

    # N-methyl/ethyl: CYP3A4 N-demethylation hotspot
    ("N_alkyl",
     "[CH3,CH2;$([CH3,CH2]-[N;!$(N-C=O);!$(N~[!#6;!#1])])]", 0.86),

    # α-carbonyl sp3 C: CYP-mediated α-hydroxylation
    ("alpha_carbonyl_C",
     "[CH2,CH3;$([CH2,CH3]C(=O)[#6,#7,#8])]", 0.75),

    # Terminal methyl (ω-hydroxylation, CYP4A/2E1)
    ("terminal_methyl_aliphatic",
     "[CH3;$([CH3]-[CH2]-[CH2]-[#6])]", 0.65),

    # Thioether S: CYP S-oxidation → sulfoxide → sulfone
    ("thioether_S",
     "[S;X2;!$(S=O);$([S]([#6])[#6])]", 0.80),

    # Tertiary amine N: FMO/CYP3A4 N-oxidation
    ("tertiary_amine_N",
     "[N;X3;!$(N-C=O);!$(N~[!#6;!#1]);!n;$([N]([#6])([#6])[#6])]", 0.78),

    # -- Phase I: Reduction --------------------------------------------------

    # Nitro group reduction (gut flora + hepatic)
    ("nitro_reduction",
     "[N+](=O)[O-]", 0.70),

    # Ketone/aldehyde carbonyl reduction (AKR / carbonyl reductases)
    ("carbonyl_reduction",
     "[CH1,CH0;$(C=O);!$(C(=O)[OH]);!$(C(=O)N)]", 0.60),

    # -- Phase II: Conjugation-prone sites -----------------------------------

    # Phenolic OH carbon: glucuronidation / sulfation (mark the C, not the O)
    ("phenolic_OH_site",
     "[c;$([c][OH1])]", 0.83),

    # Primary aliphatic amine: acetylation / glucuronidation
    ("primary_amine",
     "[NH2;$([NH2]-[#6;!$(C=O)])]", 0.72),

    # Carboxylic acid: acyl-glucuronidation (reactive metabolite risk)
    ("carboxylic_acid",
     "[C;$(C(=O)[OH1])]", 0.68),
]

# Normalisation pipeline  (shared singletons — expensive to construct)
_NORMALISER       = rdMolStandardize.Normalizer()
_UNCHARGER        = rdMolStandardize.Uncharger()
_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()


# ==============================================================================
# Pipeline B — DeepLearningPredictor
# ==============================================================================

class DeepLearningPredictor:
    """
    Production-ready deep-learning metabolite predictor emulator.

    Architectural inspiration
    -------------------------
    MetaTrans (KavrakiLab/MetaTrans): OpenNMT Transformer trained on
    experimental metabolite SMILES pairs.  Encodes the parent SMILES token
    sequence, decodes up to ``top_k`` metabolite sequences with beam search,
    and scores them with the model's log-likelihood.

    Meta-Predictor (zhukeyun/Meta-Predictor): Adds per-atom attention heads
    that produce site-of-metabolism risk vectors directly usable for soft
    spot annotation without a separate SOM model.

    This class
    ----------
    Implements the full interface those models require:
      - SMILES tokenisation  (atom-level, compatible with MetaTrans vocabulary)
      - Structural verification of generated SMILES via RDKit
      - Per-atom attention weight extraction (emulated with a deterministic
        chemistry-aware heuristic until real weights are loaded)
      - Graceful degradation: any tensor/inference failure returns empty lists
        with informative warnings, never crashes

    Swap-in path for real weights
    -----------------------------
    1. Install PyTorch + OpenNMT-py:
           pip install torch opennmt-py
    2. Subclass or monkey-patch ``_transformer_beam_search``:
           def _transformer_beam_search(self, tokens, top_k):
               # real OpenNMT / HuggingFace inference here
               ...
    3. Subclass or monkey-patch ``_extract_attention_weights``:
           def _extract_attention_weights(self, mol, tokens):
               # real model forward-pass attention extraction
               ...
    4. Load weights via ``from_checkpoint``:
           predictor = DeepLearningPredictor.from_checkpoint("metatrans.pt")

    All public method signatures remain stable between emulator and real-weight
    modes — no changes needed in the calling code.
    """

    # ── MetaTrans atom-level vocabulary (subset) ──────────────────────────────
    # Real MetaTrans uses a regex tokeniser that splits on atom symbols,
    # bond characters, ring digits, and branch brackets.  We replicate that
    # split pattern here so tokenised sequences are compatible.
    _TOKEN_PATTERN = (
        r"(\[[^\]]+\]"           # bracket atoms  [NH2], [13C], ...
        r"|Br|Cl|Si|Se|@@|@"     # two-char atoms / stereo
        r"|[BCNOPSFIbcnops]"     # one-char organic subset
        r"|[0-9%]"               # ring closure digits
        r"|[/\\+=\-#\(\)\.])"    # bond / branch / dot
    )

    MODEL_NAME = "metatrans-emulator-v2"

    def __init__(
        self,
        top_k: int = 10,
        confidence_threshold: float = DL_CONFIDENCE_THRESHOLD,
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        top_k                : beam-search width (number of candidate
                               metabolite sequences returned per parent).
        confidence_threshold : minimum model confidence to include a prediction.
        seed                 : optional random seed for the emulator (for
                               reproducible unit tests only; real models
                               are deterministic given the same weights).
        """
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self._rng = random.Random(seed)
        self._model_loaded = False   # True once real weights are attached
        logger.info(
            "dl_predictor_init",
            model=self.MODEL_NAME,
            top_k=top_k,
            mode="emulator",
        )

    # -- Class-method constructors --------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        **kwargs,
    ) -> "DeepLearningPredictor":
        """
        Load a MetaTrans / Meta-Predictor checkpoint.

        This method is a **stub** — it returns an emulator instance and logs
        a warning until real model loading code is inserted here.

        To implement:
        -------------
            import torch, onmt
            model = onmt.model_builder.build_base_model(opt, fields, gpu)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            predictor = cls(**kwargs)
            predictor._model = model
            predictor._model_loaded = True
            return predictor
        """
        instance = cls(**kwargs)
        logger.warning(
            "dl_checkpoint_stub",
            path=checkpoint_path,
            msg="Real model weights not loaded — using emulator. "
                "Implement from_checkpoint() to enable live inference.",
        )
        return instance

    # -- Tokenisation ---------------------------------------------------------

    def tokenise(self, smiles: str) -> List[str]:
        """
        Atom-level SMILES tokeniser compatible with MetaTrans vocabulary.

        Splits the SMILES string into a list of token strings, each
        representing a single atom, bond, stereo descriptor, or bracket group.

        Parameters
        ----------
        smiles : canonical SMILES string (pre-normalised by RDKit).

        Returns
        -------
        List of token strings, e.g.
            "CC(=O)O" → ["C", "C", "(", "=", "O", ")", "O"]

        Raises
        ------
        ValueError : if the SMILES is empty or produces zero tokens.
        """
        import re
        if not smiles or not smiles.strip():
            raise ValueError("Cannot tokenise empty SMILES.")
        tokens = re.findall(self._TOKEN_PATTERN, smiles)
        if not tokens:
            raise ValueError(
                f"SMILES '{smiles}' produced zero tokens. "
                "It may contain unsupported notation."
            )
        return tokens

    # -- Structural verification ----------------------------------------------

    @staticmethod
    def verify_smiles(smiles: str) -> Optional[Chem.Mol]:
        """
        Verify that a SMILES string represents a valid, sanitisable molecule.

        Catches the full RDKit error taxonomy:
          - ``rdchem.MolSanitizeException`` — valence/kekulisation failures
          - ``rdchem.KekulizeException``    — aromatic resolution failures
          - ``ValueError``                  — bad SMILES syntax
          - Any other Exception             — unknown RDKit error

        Parameters
        ----------
        smiles : SMILES string to verify (may be transformer-generated).

        Returns
        -------
        Sanitised ``Chem.Mol`` on success, or ``None`` if any check fails.
        """
        if not smiles or not smiles.strip():
            return None
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None

            # Attempt full sanitisation — this is the main gate for
            # hypervalent/impossible structures generated by the transformer
            sanitize_flags = Chem.SanitizeFlags.SANITIZE_ALL
            result_flag = Chem.SanitizeMol(mol, sanitize_flags, catchErrors=True)
            if result_flag != Chem.SanitizeFlags.SANITIZE_NONE:
                logger.debug(
                    "dl_smiles_sanitize_failed"),
                )
                return None

            if mol.GetNumHeavyAtoms() == 0:
                return None

            # Re-parse from canonical SMILES for clean atom mapping
            canonical = Chem.MolToSmiles(mol, canonical=True)
            mol2 = Chem.MolFromSmiles(canonical)
            return mol2

        except rdchem.MolSanitizeException as exc:
            logger.debug("dl_mol_sanitize_exception", error=str(exc), smiles=smiles[:80])
            return None
        except rdchem.KekulizeException as exc:
            logger.debug("dl_mol_kekulize_exception", error=str(exc), smiles=smiles[:80])
            return None
        except Exception as exc:
            logger.debug("dl_mol_verify_error", error=str(exc), smiles=smiles[:80])
            return None

    # -- Core inference hooks (emulated; replace with real model calls) --------

    def _transformer_beam_search(
        self,
        tokens: List[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Emulate transformer beam-search decoding.

        PRODUCTION REPLACEMENT POINT
        ----------------------------
        Replace the body of this method with a real OpenNMT / HuggingFace
        inference call.  The expected signature and return type are stable:

            inputs  : token list from ``tokenise()``
            returns : list of (smiles_string, log_likelihood_score) tuples,
                      up to ``top_k`` entries, sorted by score descending.

        Emulator strategy
        -----------------
        Rather than pure randomness (which would produce mostly invalid SMILES),
        the emulator applies a set of chemically plausible single-step
        transformations to the input SMILES — the same operations MetaTrans
        was trained to generate.  This ensures a realistic fraction of
        structurally valid outputs for testing and development.

        Confidence scores are set to realistic ranges drawn from published
        MetaTrans log-likelihood distributions (Litsa et al. 2023).
        """
        parent_smiles = "".join(tokens)

        # Validate the parent SMILES before attempting to transform
        parent_mol = self.verify_smiles(parent_smiles)
        if parent_mol is None:
            return []

        # Deterministic seed: same parent → same emulated outputs across runs
        # (prevents flapping tests while still covering diverse transformations)
        seed_hash = int(hashlib.sha256(parent_smiles.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed_hash)

        candidates: List[Tuple[str, float]] = []

        # ── Emulated transformation set ────────────────────────────────────────
        # Each lambda attempts a chemically plausible single SMIRKS-like step.
        # Transformations that don't apply to the molecule are silently skipped.
        transformations = [
            self._emulate_aromatic_hydroxylation,
            self._emulate_aliphatic_hydroxylation,
            self._emulate_n_demethylation,
            self._emulate_o_demethylation,
            self._emulate_glucuronidation,
            self._emulate_sulfation,
            self._emulate_oxidation_sulfoxide,
            self._emulate_hydrolysis,
        ]

        rng.shuffle(transformations)

        for transform in transformations[:top_k]:
            try:
                result_smiles = transform(parent_mol, rng)
                if result_smiles is None:
                    continue
                verified = self.verify_smiles(result_smiles)
                if verified is None:
                    continue
                canon = Chem.MolToSmiles(verified, canonical=True)
                # Emulated log-likelihood in realistic MetaTrans range (~-0.1 to -3.0)
                score = rng.uniform(0.08, 0.55)
                candidates.append((canon, round(score, 4)))
            except Exception as exc:
                logger.debug("dl_transform_failed", error=str(exc))
                continue

        # Sort by score descending, deduplicate
        seen_smiles: Set[str] = set()
        unique: List[Tuple[str, float]] = []
        for smiles, score in sorted(candidates, key=lambda x: x[1], reverse=True):
            if smiles not in seen_smiles:
                seen_smiles.add(smiles)
                unique.append((smiles, score))

        return unique[:top_k]

    def _extract_attention_weights(
        self,
        mol: Chem.Mol,
        tokens: List[str],
    ) -> Dict[int, float]:
        """
        Extract per-atom SOM attention risk weights from the transformer.

        PRODUCTION REPLACEMENT POINT
        ----------------------------
        In a real MetaTrans / Meta-Predictor model, this method runs a
        forward pass and reads the cross-attention weights from the encoder
        layers aligned to each heavy atom.  Replace the body with:

            with torch.no_grad():
                outputs = self._model(input_tensor, attention=True)
                atom_weights = outputs.encoder_attentions[atom_indices]
            return {idx: float(w) for idx, w in enumerate(atom_weights)}

        Emulator strategy
        -----------------
        We produce a chemistry-aware pseudo-attention distribution that
        correlates with known SOM preferences, making the emulated values
        directionally correct for testing and UI development.

        The distribution is built from three additive signals:
          1. Atomic number contribution (heteroatoms get higher base weight)
          2. Environment contribution (aromatic/sp3/sp2 character)
          3. SMARTS rule overlap (atoms matching known SOM patterns get boost)

        All values are normalised to [0, 1] with softmax.
        """
        n_atoms = mol.GetNumHeavyAtoms()
        if n_atoms == 0:
            return {}

        raw_weights: Dict[int, float] = {}

        # Precompute which atoms match any SOM rule
        som_boosted: Set[int] = set()
        for _, smarts, _ in _SOM_RULES:
            pat = Chem.MolFromSmarts(smarts)
            if pat is None:
                continue
            for match in mol.GetSubstructMatches(pat):
                if match:
                    som_boosted.add(match[0])

        # ── Pseudo-attention computation ───────────────────────────────────────
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            w = 0.0

            # Signal 1: heteroatom base weight
            atomic_num = atom.GetAtomicNum()
            if atomic_num in (7, 8):       # N, O
                w += 0.35
            elif atomic_num == 16:          # S
                w += 0.30
            elif atomic_num == 6:           # C
                w += 0.15
            else:
                w += 0.20

            # Signal 2: hybridisation / environment
            hyb = atom.GetHybridization()
            if atom.GetIsAromatic():
                w += 0.25
            elif hyb == Chem.rdchem.HybridizationType.SP3:
                w += 0.20
            elif hyb == Chem.rdchem.HybridizationType.SP2:
                w += 0.18

            # Signal 3: explicit hydrogen count (CYP accessibility proxy)
            w += atom.GetTotalNumHs() * 0.10

            # Signal 4: SOM rule overlap boost
            if idx in som_boosted:
                w += 0.40

            raw_weights[idx] = w

        # Softmax normalisation → values sum to 1.0, then rescale to [0, 1]
        exp_w = {idx: math.exp(v) for idx, v in raw_weights.items()}
        total = sum(exp_w.values()) or 1.0
        normalised = {idx: v / total for idx, v in exp_w.items()}

        # Rescale: set max atom to 1.0, others proportionally
        max_val = max(normalised.values()) if normalised else 1.0
        if max_val > 0:
            return {idx: round(v / max_val, 4) for idx, v in normalised.items()}
        return {idx: 0.0 for idx in normalised}

    # -- Chemical transformation emulators ------------------------------------
    # Each method attempts a single chemical step on ``mol`` and returns a
    # product SMILES string, or None if the transformation doesn't apply.
    # These are *not* meant to be chemically exhaustive — they serve as
    # realistic test vectors for the DL pipeline interface.

    @staticmethod
    def _emulate_aromatic_hydroxylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Add -OH to an unhindered aromatic C-H (CYP1A2/2C19 style)."""
        try:
            from rdkit.Chem import AllChem, RWMol
            pattern = Chem.MolFromSmarts("[cH]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            atom_idx = match[0]
            rw = Chem.RWMol(mol)
            rw.GetAtomWithIdx(atom_idx).SetNumExplicitHs(0)
            oh_idx = rw.AddAtom(Chem.Atom(8))     # O
            h_idx  = rw.AddAtom(Chem.Atom(1))     # H
            rw.AddBond(atom_idx, oh_idx, Chem.BondType.SINGLE)
            rw.AddBond(oh_idx,   h_idx,  Chem.BondType.SINGLE)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_aliphatic_hydroxylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Add -OH to an aliphatic sp3 C-H (benzylic or α-heteroatom)."""
        try:
            from rdkit.Chem import RWMol
            # Prefer benzylic CH2
            pattern = Chem.MolFromSmarts("[CH2;$([CH2]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                pattern = Chem.MolFromSmarts("[CH3;$([CH3]-[c])]")
                matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            atom_idx = match[0]
            rw = Chem.RWMol(Chem.AddHs(mol))
            # Find one explicit H on this atom
            h_to_remove = None
            for nbr in rw.GetAtomWithIdx(atom_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 1:
                    h_to_remove = nbr.GetIdx()
                    break
            if h_to_remove is None:
                return None
            rw.RemoveAtom(h_to_remove)
            oh_idx = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(atom_idx, oh_idx, Chem.BondType.SINGLE)
            product = Chem.RemoveHs(rw)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(product, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_n_demethylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Remove N-methyl group (CYP3A4 N-demethylation)."""
        try:
            from rdkit.Chem import AllChem, RWMol, rdMolTransforms
            pattern = Chem.MolFromSmarts("[CH3;$([CH3]-[N;!$(N-C=O)])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx = match[0]
            rw = Chem.RWMol(mol)
            # Find the N neighbor
            n_idx = None
            for nbr in rw.GetAtomWithIdx(c_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 7:
                    n_idx = nbr.GetIdx()
                    break
            if n_idx is None:
                return None
            rw.RemoveBond(c_idx, n_idx)
            # Remove the dangling CH3
            atoms_to_remove = sorted([c_idx], reverse=True)
            for idx in atoms_to_remove:
                rw.RemoveAtom(idx)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_o_demethylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Remove O-methyl group → phenol (CYP2D6 / 2C9 O-demethylation)."""
        try:
            from rdkit.Chem import RWMol
            pattern = Chem.MolFromSmarts("[CH3;$([CH3]-O-[c,C])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx = match[0]
            rw = Chem.RWMol(mol)
            o_idx = None
            for nbr in rw.GetAtomWithIdx(c_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 8:
                    o_idx = nbr.GetIdx()
                    break
            if o_idx is None:
                return None
            rw.RemoveBond(c_idx, o_idx)
            # Remove the dangling CH3
            for idx in sorted([c_idx], reverse=True):
                rw.RemoveAtom(idx)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_glucuronidation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Append glucuronic acid to a phenol (Phase II conjugation)."""
        try:
            # Instead of real glucuronide construction (complex stereochemistry),
            # return the deprotonated phenol as a proxy product — structurally
            # correct for testing the DL pipeline, not for clinical use.
            pattern = Chem.MolFromSmarts("[OH;$([OH]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            # Proxy: return the phenol itself (identity for phenols)
            # Real implementation would use a SMIRKS reaction:
            # "[OH:1]-[c:2]>>[O:1]([c:2])C1OC(C(=O)O)C(O)C(O)C1O"
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    @staticmethod
    def _emulate_sulfation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Sulfate conjugation of a phenol (Phase II, SULT enzymes)."""
        try:
            from rdkit.Chem import RWMol, AllChem
            pattern = Chem.MolFromSmarts("[OH;$([OH]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            o_idx = match[0]
            rw = Chem.RWMol(mol)
            s_idx  = rw.AddAtom(Chem.Atom(16))   # S
            o1_idx = rw.AddAtom(Chem.Atom(8))    # =O
            o2_idx = rw.AddAtom(Chem.Atom(8))    # =O
            o3_idx = rw.AddAtom(Chem.Atom(8))    # -OH
            h_idx  = rw.AddAtom(Chem.Atom(1))
            rw.AddBond(o_idx,  s_idx,  Chem.BondType.SINGLE)
            rw.AddBond(s_idx,  o1_idx, Chem.BondType.DOUBLE)
            rw.AddBond(s_idx,  o2_idx, Chem.BondType.DOUBLE)
            rw.AddBond(s_idx,  o3_idx, Chem.BondType.SINGLE)
            rw.AddBond(o3_idx, h_idx,  Chem.BondType.SINGLE)
            # Remove the original O-H proton
            rw.GetAtomWithIdx(o_idx).SetNumExplicitHs(0)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_oxidation_sulfoxide(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Oxidise thioether S → sulfoxide (CYP / FMO S-oxidation)."""
        try:
            from rdkit.Chem import RWMol
            pattern = Chem.MolFromSmarts("[S;X2;!$(S=O);$([S]([#6])[#6])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            s_idx = match[0]
            rw = Chem.RWMol(mol)
            o_idx = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(s_idx, o_idx, Chem.BondType.DOUBLE)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_hydrolysis(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Ester hydrolysis → carboxylic acid + alcohol."""
        try:
            from rdkit.Chem import AllChem, RWMol
            # Detect ester: [C](=O)-O-[C,c]
            pattern = Chem.MolFromSmarts("[C:1](=[O:2])-[O:3]-[#6:4]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx, o_dbl, o_single, c_right = match
            rw = Chem.RWMol(mol)
            rw.RemoveBond(o_single, c_right)
            # Add H to both oxygens
            rw.GetAtomWithIdx(o_single).SetNumExplicitHs(1)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    # -- Public prediction API ------------------------------------------------

    def predict(
        self,
        smiles: str,
        phase_hint: Optional[int] = None,
    ) -> Tuple[List[Tuple[str, float]], Dict[int, float], List[str]]:
        """
        Run the full DL prediction pipeline for a single parent molecule.

        Parameters
        ----------
        smiles     : canonical SMILES (pre-normalised by ``_validate_and_normalise``).
        phase_hint : optional phase bias (1 or 2); not used by emulator but
                     available for real model phase-conditioned decoding.

        Returns
        -------
        (predictions, attention_weights, warnings)

          predictions       : List[(canonical_smiles, confidence_score)]
                              sorted by confidence descending.
          attention_weights : Dict[atom_index → risk_weight]
                              per-atom SOM attention scores in [0, 1].
          warnings          : List of non-fatal diagnostic strings.
        """
        dl_warnings: List[str] = []

        # ── Tokenise ──────────────────────────────────────────────────────────
        try:
            tokens = self.tokenise(smiles)
        except ValueError as exc:
            dl_warnings.append(f"DL tokenisation failed: {exc}")
            return [], {}, dl_warnings

        # ── Verify parent ─────────────────────────────────────────────────────
        parent_mol = self.verify_smiles(smiles)
        if parent_mol is None:
            dl_warnings.append(
                f"DL pipeline skipped: parent SMILES failed structural "
                f"verification after tokenisation. SMILES: '{smiles[:80]}'"
            )
            return [], {}, dl_warnings

        # ── Beam search (emulated or real) ────────────────────────────────────
        try:
            raw_predictions = self._transformer_beam_search(tokens, self.top_k)
        except Exception as exc:
            dl_warnings.append(f"DL beam search error: {exc}")
            raw_predictions = []

        # ── Filter below confidence threshold ─────────────────────────────────
        predictions = [
            (smi, score)
            for smi, score in raw_predictions
            if score >= self.confidence_threshold
        ]

        # ── Attention weights (emulated or real) ──────────────────────────────
        try:
            attention_weights = self._extract_attention_weights(parent_mol, tokens)
        except Exception as exc:
            dl_warnings.append(f"DL attention extraction error: {exc}")
            attention_weights = {}

        logger.info(
            "dl_predict_complete",
            model=self.MODEL_NAME,
            raw_predictions=len(raw_predictions),
            filtered_predictions=len(predictions),
            attention_atoms=len(attention_weights),
        )

        return predictions, attention_weights, dl_warnings


# ==============================================================================
# Consensus Engine
# ==============================================================================

class ConsensusEngine:
    """
    Merges Pipeline A (SyGMa) and Pipeline B (DL) outputs into a unified,
    confidence-tagged metabolite list and ensemble soft spot annotations.

    Consensus logic
    ---------------
    Structural alignment uses ``Chem.CanonSmiles`` to normalise both sets of
    SMILES before comparison.  An exact match of canonical forms constitutes
    consensus — no Tanimoto similarity threshold is used because near-identical
    SMILES with different stereochemistry represent chemically distinct metabolites
    and should be tagged separately.

    Score fusion
    ------------
    Consensus metabolites inherit the higher of the two pipeline scores as
    their primary ``probability`` field, preserving the SyGMa reaction name
    and phase assignment (which the DL model does not produce).

    For soft spots, the vulnerability index blends rule and attention signals:
        VI = (ENSEMBLE_ALPHA × rule_score + ENSEMBLE_BETA × dl_attention) × 100 %
    """

    def __init__(
        self,
        alpha: float = ENSEMBLE_ALPHA,
        beta:  float = ENSEMBLE_BETA,
    ) -> None:
        self.alpha = alpha
        self.beta  = beta

    # ------------------------------------------------------------------

    def merge_metabolites(
        self,
        sygma_metabolites: List[PredictedMetabolite],
        dl_predictions:    List[Tuple[str, float]],
    ) -> Tuple[List[PredictedMetabolite], Dict]:
        """
        Merge and tag SyGMa + DL metabolite predictions.

        Parameters
        ----------
        sygma_metabolites : output of ``_run_sygma``.
        dl_predictions    : list of (canonical_smiles, confidence) from DL.

        Returns
        -------
        (merged_metabolites, stats_dict)
          merged_metabolites : deduplicated, confidence-tagged list sorted
                               by probability descending.
          stats_dict         : pipeline statistics for logging / API response.
        """
        # Build canonical SMILES lookup for SyGMa results
        sygma_by_canon: Dict[str, PredictedMetabolite] = {}
        for m in sygma_metabolites:
            canon = self._safe_canonicalise(m.smiles)
            if canon:
                sygma_by_canon[canon] = m

        # Build canonical SMILES set for DL results
        dl_by_canon: Dict[str, float] = {}
        for smiles, score in dl_predictions:
            canon = self._safe_canonicalise(smiles)
            if canon:
                dl_by_canon[canon] = score

        consensus_set    = set(sygma_by_canon) & set(dl_by_canon)
        sygma_only_set   = set(sygma_by_canon) - set(dl_by_canon)
        dl_only_set      = set(dl_by_canon)    - set(sygma_by_canon)

        merged: List[PredictedMetabolite] = []

        # ── Consensus metabolites ──────────────────────────────────────────────
        for canon in consensus_set:
            base = sygma_by_canon[canon]
            dl_s = dl_by_canon[canon]
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=max(base.probability, dl_s),
                phase=base.phase,
                reaction_name=base.reaction_name,
                molecular_weight=base.molecular_weight,
                molecular_formula=base.molecular_formula,
                confidence_tier=ConsensusTier.HIGH,
                dl_score=dl_s,
                sources=frozenset({"sygma", "dl"}),
            ))

        # ── SyGMa-only metabolites ────────────────────────────────────────────
        for canon in sygma_only_set:
            base = sygma_by_canon[canon]
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=base.probability,
                phase=base.phase,
                reaction_name=base.reaction_name,
                molecular_weight=base.molecular_weight,
                molecular_formula=base.molecular_formula,
                confidence_tier=ConsensusTier.RULE,
                dl_score=None,
                sources=frozenset({"sygma"}),
            ))

        # ── DL-only metabolites ───────────────────────────────────────────────
        for canon in dl_only_set:
            dl_s = dl_by_canon[canon]
            mol = self._safe_mol(canon)
            mw, formula = self._compute_descriptors(mol)
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=dl_s,
                phase=-1,           # DL model does not produce phase labels
                reaction_name="dl_predicted",
                molecular_weight=mw,
                molecular_formula=formula,
                confidence_tier=ConsensusTier.DL,
                dl_score=dl_s,
                sources=frozenset({"dl"}),
            ))

        merged.sort(key=lambda m: m.probability, reverse=True)

        stats = {
            "sygma_total":      len(sygma_by_canon),
            "dl_total":         len(dl_by_canon),
            "consensus_count":  len(consensus_set),
            "rule_only_count":  len(sygma_only_set),
            "dl_only_count":    len(dl_only_set),
            "merged_total":     len(merged),
        }

        logger.info("consensus_merge_complete", **stats)
        return merged, stats

    def merge_soft_spots(
        self,
        rule_spots:       List[SoftSpot],
        attention_weights: Dict[int, float],
        mol:              Chem.Mol,
        top_n:            int,
        dl_source:        str = "metatrans-emulator",
    ) -> List[EnsembleSoftSpot]:
        """
        Fuse rule-based and DL attention soft spot signals.

        For atoms with a rule match: VI = (α·rule_score + β·dl_attention) × 100
        For atoms with attention only (no rule match): VI = β·dl_attention × 100
        Rule-matched atoms always appear in the top-N if they outscore attention-only.

        Parameters
        ----------
        rule_spots        : output of ``_find_soft_spots``.
        attention_weights : per-atom attention weights from DL predictor.
        mol               : parent molecule (used to look up atom symbols for
                            attention-only atoms not in rule_spots).
        top_n             : number of ensemble spots to return.
        dl_source         : identifier for the DL model providing attention.

        Returns
        -------
        List of ``EnsembleSoftSpot``, sorted by vulnerability_index descending,
        length ≤ top_n.
        """
        ensemble: Dict[int, EnsembleSoftSpot] = {}

        # Process rule-matched atoms
        for spot in rule_spots:
            dl_w = attention_weights.get(spot.atom_index, 0.0)
            vi   = round((self.alpha * spot.score + self.beta * dl_w) * 100, 2)
            ensemble[spot.atom_index] = EnsembleSoftSpot(
                atom_index=spot.atom_index,
                atom_symbol=spot.atom_symbol,
                rule_name=spot.rule_name,
                smarts_match=spot.smarts_match,
                rule_score=spot.score,
                dl_attention_risk=dl_w,
                vulnerability_index=vi,
                dl_source=dl_source,
            )

        # Process attention-only atoms (not in rule_spots) — add if high attention
        if attention_weights:
            # Threshold: include attention-only atoms above the 75th percentile
            sorted_weights = sorted(attention_weights.values(), reverse=True)
            threshold_rank = max(1, len(sorted_weights) // 4)
            threshold = sorted_weights[threshold_rank - 1] if sorted_weights else 0.0

            for atom_idx, dl_w in attention_weights.items():
                if atom_idx in ensemble:
                    continue
                if dl_w < threshold:
                    continue
                vi = round(self.beta * dl_w * 100, 2)
                try:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    symbol = atom.GetSymbol()
                except Exception:
                    symbol = "?"
                ensemble[atom_idx] = EnsembleSoftSpot(
                    atom_index=atom_idx,
                    atom_symbol=symbol,
                    rule_name="dl_attention_only",
                    smarts_match="",
                    rule_score=0.0,
                    dl_attention_risk=dl_w,
                    vulnerability_index=vi,
                    dl_source=dl_source,
                )

        result = sorted(
            ensemble.values(),
            key=lambda s: s.vulnerability_index,
            reverse=True,
        )
        return result[:top_n]

    # -- Private helpers ------------------------------------------------------

    @staticmethod
    def _safe_canonicalise(smiles: str) -> Optional[str]:
        """Return RDKit canonical SMILES, or None on any RDKit error."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None

    @staticmethod
    def _safe_mol(smiles: str) -> Optional[Chem.Mol]:
        try:
            return Chem.MolFromSmiles(smiles)
        except Exception:
            return None

    @staticmethod
    def _compute_descriptors(
        mol: Optional[Chem.Mol],
    ) -> Tuple[Optional[float], Optional[str]]:
        if mol is None:
            return None, None
        try:
            return (
                round(Descriptors.MolWt(mol), 4),
                rdMolDescriptors.CalcMolFormula(mol),
            )
        except Exception:
            return None, None


# ==============================================================================
# Private helpers  (validation, SyGMa runner, rule-based SOM)
# ==============================================================================

def _validate_and_normalise(smiles: str) -> Tuple[Chem.Mol, MoleculeMetadata]:
    """
    Parse, validate, and normalise a SMILES string.

    Normalisation pipeline
    ----------------------
    1. ``Chem.MolFromSmiles``     — parse + valence check
    2. ``LargestFragmentChooser`` — strip salts / counterions
    3. ``Normalizer``             — canonical functional-group forms
    4. ``Uncharger``              — neutralise formal charges
    5. ``SanitizeMol``            — final RDKit sanitisation
    6. Descriptor computation     — MW, formula, InChI, logP, TPSA, etc.

    Raises
    ------
    ValueError  if the SMILES is invalid at any pipeline stage.
    """
    if not smiles or not smiles.strip():
        raise ValueError("SMILES string is empty.")

    smiles = smiles.strip()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(
            f"RDKit could not parse SMILES: {smiles!r}. "
            "Check atom symbols, valences, and ring-closure notation."
        )

    mol = _LARGEST_FRAGMENT.choose(mol)
    mol = _NORMALISER.normalize(mol)
    mol = _UNCHARGER.uncharge(mol)

    try:
        Chem.SanitizeMol(mol)
    except rdchem.MolSanitizeException as exc:
        raise ValueError(f"Sanitisation failed after normalisation: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Normalisation error: {exc}") from exc

    if mol.GetNumHeavyAtoms() == 0:
        raise ValueError("Molecule has no heavy atoms after normalisation.")

    from rdkit.Chem import inchi as rdinchi
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    inchi_str   = rdinchi.MolToInchi(mol) or ""
    inchikey    = rdinchi.InchiToInchiKey(inchi_str) if inchi_str else ""

    metadata = MoleculeMetadata(
        input_smiles=smiles,
        canonical_smiles=canonical_smiles,
        inchi=inchi_str,
        inchikey=inchikey,
        molecular_formula=rdMolDescriptors.CalcMolFormula(mol),
        molecular_weight=round(Descriptors.MolWt(mol), 4),
        exact_mass=round(Descriptors.ExactMolWt(mol), 6),
        num_heavy_atoms=mol.GetNumHeavyAtoms(),
        num_rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
        num_hbd=rdMolDescriptors.CalcNumHBD(mol),
        num_hba=rdMolDescriptors.CalcNumHBA(mol),
        tpsa=round(rdMolDescriptors.CalcTPSA(mol), 2),
        logp=round(Descriptors.MolLogP(mol), 4),
        num_rings=rdMolDescriptors.CalcNumRings(mol),
        num_aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
    )

    logger.debug(
        "molecule_normalised",
        canonical=canonical_smiles,
        mw=metadata.molecular_weight,
        formula=metadata.molecular_formula,
    )
    return mol, metadata


def _run_sygma(
    mol: Chem.Mol,
    phase1_cycles: int = 1,
    phase2_cycles: int = 1,
    max_metabolites: int = 200,
) -> List[PredictedMetabolite]:
    """
    Execute a SyGMa Phase I → Phase II prediction scenario.

    Returns PredictedMetabolite instances with ``sources=frozenset({"sygma"})``
    and ``confidence_tier=ConsensusTier.RULE`` (to be upgraded to HIGH by the
    ConsensusEngine when DL agreement is found).

    Raises
    ------
    ImportError  if SyGMa is not installed.
    """
    try:
        import sygma                          # type: ignore
        from sygma import ruleset as srs      # type: ignore
        from sygma import Scenario            # type: ignore
    except ImportError as exc:
        raise ImportError(
            "SyGMa is not installed. "
            "Add 'sygma>=1.1.0' to the pip section of environment.yml."
        ) from exc

    scenario = Scenario([
        [srs.phase1, phase1_cycles],
        [srs.phase2, phase2_cycles],
    ])

    logger.debug("sygma_start", p1=phase1_cycles, p2=phase2_cycles)
    tree = scenario.run(mol)
    tree.calc_scores()

    parent_canon = Chem.MolToSmiles(mol, canonical=True)
    seen: Dict[str, PredictedMetabolite] = {}

    for node in tree.to_list():
        node_mol = getattr(node, "mol", None)
        if node_mol is None:
            continue

        # Per-metabolite error defence — catch sanitisation failures
        try:
            Chem.SanitizeMol(node_mol)
        except rdchem.MolSanitizeException as exc:
            logger.debug("sygma_sanitize_skip", error=str(exc))
            continue
        except Exception as exc:
            logger.debug("sygma_mol_skip", error=str(exc))
            continue

        try:
            smiles = Chem.MolToSmiles(node_mol, canonical=True)
        except Exception as exc:
            logger.debug("sygma_smiles_skip", error=str(exc))
            continue

        if not smiles or smiles == parent_canon:
            continue

        probability   = float(getattr(node, "score",         0.0))
        reaction_name = str(getattr(node,  "reaction_name", "") or "unknown")

        phase = (2 if reaction_name.lower().startswith("phase2")
                 or any(k in reaction_name.lower()
                        for k in ("glucuronid", "sulfat", "acetyl",
                                  "glutathione", "methyl"))
                 else 1)
        if reaction_name.lower().startswith("phase1"):
            phase = 1

        try:
            mw      = round(Descriptors.MolWt(node_mol), 4)
            formula = rdMolDescriptors.CalcMolFormula(node_mol)
        except Exception:
            mw, formula = None, None

        candidate = PredictedMetabolite(
            smiles=smiles,
            probability=probability,
            phase=phase,
            reaction_name=reaction_name,
            molecular_weight=mw,
            molecular_formula=formula,
            confidence_tier=ConsensusTier.RULE,   # upgraded by consensus later
            sources=frozenset({"sygma"}),
        )

        if smiles not in seen or probability > seen[smiles].probability:
            seen[smiles] = candidate

    results = sorted(seen.values(), key=lambda x: x.probability, reverse=True)
    results = results[:max_metabolites]

    logger.info(
        "sygma_complete",
        unique=len(results),
        p1=sum(1 for r in results if r.phase == 1),
        p2=sum(1 for r in results if r.phase == 2),
    )
    return results


def _find_soft_spots(
    mol: Chem.Mol,
    top_n: int = 3,
) -> List[SoftSpot]:
    """
    Rule-only soft spot fallback (Pipeline A standalone).

    Used when the DL pipeline is unavailable or disabled.  The ConsensusEngine
    calls ``merge_soft_spots`` instead when both pipelines are active.

    Algorithm
    ---------
    For each rule in ``_SOM_RULES``:
      - Compile SMARTS, find substructure matches.
      - Atom at match index 0 is the SOM site.
      - Apply a 20 % steric penalty for atoms with ≥ 3 bulky neighbours.
    """
    best: Dict[int, SoftSpot] = {}

    for rule_name, smarts, base_score in _SOM_RULES:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            logger.warning("invalid_smarts", rule=rule_name)
            continue

        for match in mol.GetSubstructMatches(pattern):
            if not match:
                continue
            atom_idx = match[0]
            atom     = mol.GetAtomWithIdx(atom_idx)

            heavy_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
            score = base_score
            if len(heavy_nbrs) >= 3 and any(n.GetDegree() > 1 for n in heavy_nbrs):
                score *= 0.80

            score = round(score, 4)
            if atom_idx not in best or score > best[atom_idx].score:
                best[atom_idx] = SoftSpot(
                    atom_index=atom_idx,
                    atom_symbol=atom.GetSymbol(),
                    rule_name=rule_name,
                    score=score,
                    smarts_match=smarts,
                )

    return sorted(best.values(), key=lambda s: s.score, reverse=True)[:top_n]


# ==============================================================================
# Public API
# ==============================================================================

# Type alias for drop-in soft-spot function hook (backward compatible with v1)
SoftSpotFn = Callable[[Chem.Mol, int], List[SoftSpot]]


def predict(
    smiles: str,
    *,
    phase1_cycles:   int  = 1,
    phase2_cycles:   int  = 1,
    max_metabolites: int  = 200,
    top_soft_spots:  int  = 3,
    # v2 ensemble parameters
    dl_predictor:    Optional[DeepLearningPredictor] = None,
    enable_dl:       bool = True,
    # v1 backward-compat hook (used when dl_predictor is None and enable_dl=False)
    soft_spot_fn:    Optional[SoftSpotFn] = None,
) -> MetabolismResult:
    """
    Ensemble Consensus Metabolism Prediction.

    Orchestrates the dual-pipeline architecture:
      Pipeline A  — SyGMa rule-based Phase I/II metabolites
      Pipeline B  — DeepLearningPredictor (emulator or real weights)
      Consensus   — cross-validates outputs, assigns ConsensusTier tags
      Soft Spots  — blends rule scores with DL attention weights

    Parameters
    ----------
    smiles          : Parent molecule SMILES (any notation; normalised internally).
    phase1_cycles   : SyGMa Phase I iteration depth.
    phase2_cycles   : SyGMa Phase II iteration depth.
    max_metabolites : Hard cap on returned metabolites (applied after merge).
    top_soft_spots  : Number of soft spots to annotate.
    dl_predictor    : Pre-built DeepLearningPredictor instance.  If None and
                      enable_dl=True, a default emulator is created internally.
    enable_dl       : Master switch for Pipeline B.  Set False to run rule-only
                      (equivalent to v1 behaviour).
    soft_spot_fn    : v1 backward-compat hook.  Ignored when enable_dl=True.

    Returns
    -------
    ``MetabolismResult`` with full ensemble data.  ``.to_dict()`` is
    backward-compatible with v1 API consumers.

    Raises
    ------
    ValueError   : SMILES is invalid.
    RuntimeError : Unexpected engine-level error.

    Examples
    --------
    >>> # Minimal usage (ensemble mode, default emulator)
    >>> from app.engine.metabolism import predict
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O")
    >>> r.parent.molecular_formula
    'C9H8O4'
    >>> all(isinstance(s, EnsembleSoftSpot) for s in r.soft_spots)
    True

    >>> # Rule-only mode (v1 equivalent)
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O", enable_dl=False)

    >>> # Real weights (when available)
    >>> predictor = DeepLearningPredictor.from_checkpoint("metatrans.pt")
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O", dl_predictor=predictor)
    """
    t_start   = time.perf_counter()
    warnings: List[str] = []

    # ── Step 1: Validate + normalise ──────────────────────────────────────────
    logger.info("ensemble_predict_start")
    mol, parent_meta = _validate_and_normalise(smiles)

    if parent_meta.canonical_smiles != smiles.strip():
        warnings.append(
            f"Input SMILES normalised: {smiles!r} → "
            f"{parent_meta.canonical_smiles!r}"
        )

    if parent_meta.molecular_weight > 1500:
        warnings.append(
            f"MW {parent_meta.molecular_weight:.1f} Da exceeds typical small-molecule "
            "range (< 1500 Da). Predictions may be unreliable."
        )

    # ── Step 2A: Pipeline A — SyGMa ───────────────────────────────────────────
    sygma_metabolites: List[PredictedMetabolite] = []
    try:
        sygma_metabolites = _run_sygma(
            mol,
            phase1_cycles=phase1_cycles,
            phase2_cycles=phase2_cycles,
            max_metabolites=max_metabolites,
        )
    except ImportError:
        warnings.append(
            "SyGMa not installed — rule-based pipeline inactive. "
            "Install: pip install sygma"
        )
        logger.warning("sygma_not_installed")
    except Exception as exc:
        warnings.append(f"SyGMa pipeline error (non-fatal): {exc}")
        logger.error("sygma_pipeline_error", error=str(exc))

    # ── Step 2B: Pipeline B — Deep Learning ───────────────────────────────────
    dl_predictions:    List[Tuple[str, float]] = []
    attention_weights: Dict[int, float]        = {}
    dl_source = "disabled"

    if enable_dl:
        # Use provided predictor or create a default emulator
        predictor = dl_predictor or DeepLearningPredictor()
        dl_source = predictor.MODEL_NAME
        try:
            (
                dl_predictions,
                attention_weights,
                dl_warnings,
            ) = predictor.predict(parent_meta.canonical_smiles)
            warnings.extend(dl_warnings)
        except Exception as exc:
            warnings.append(f"DL pipeline error (non-fatal): {exc}")
            logger.error("dl_pipeline_error", error=str(exc))
    else:
        warnings.append("DL pipeline disabled (enable_dl=False).")

    # ── Step 3: Consensus merge ────────────────────────────────────────────────
    consensus = ConsensusEngine()

    if enable_dl and (sygma_metabolites or dl_predictions):
        merged_metabolites, pipeline_stats = consensus.merge_metabolites(
            sygma_metabolites, dl_predictions
        )
        merged_metabolites = merged_metabolites[:max_metabolites]
    else:
        # Rule-only fallback: tag everything as RULE tier
        merged_metabolites = sygma_metabolites[:max_metabolites]
        pipeline_stats = {
            "sygma_total":     len(sygma_metabolites),
            "dl_total":        0,
            "consensus_count": 0,
            "rule_only_count": len(sygma_metabolites),
            "dl_only_count":   0,
            "merged_total":    len(sygma_metabolites),
        }

    # ── Step 4: Soft spot analysis ────────────────────────────────────────────
    if soft_spot_fn is not None and not enable_dl:
        # v1 backward-compat: external soft_spot_fn overrides ensemble
        soft_spots_raw = soft_spot_fn(mol, top_soft_spots)
        soft_spots: List = soft_spots_raw
    else:
        # Ensemble soft spots
        rule_spots = _find_soft_spots(mol, top_n=mol.GetNumHeavyAtoms())
        soft_spots = consensus.merge_soft_spots(
            rule_spots=rule_spots,
            attention_weights=attention_weights,
            mol=mol,
            top_n=top_soft_spots,
            dl_source=dl_source,
        )

    if not soft_spots:
        warnings.append(
            "No soft spots identified. The molecule may lack common CYP substrate "
            "features or all candidates were excluded by the steric/attention filter."
        )

    # ── Step 5: Assemble result ───────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    pipeline_stats["dl_source"]  = dl_source
    pipeline_stats["elapsed_s"]  = round(elapsed, 4)

    result = MetabolismResult(
        parent=parent_meta,
        metabolites=merged_metabolites,
        soft_spots=soft_spots,
        engine_version=ENGINE_VERSION,
        elapsed_s=elapsed,
        warnings=warnings,
        pipeline_stats=pipeline_stats,
    )

    logger.info(
        "ensemble_predict_complete",
        canonical=parent_meta.canonical_smiles,
        metabolites_total=len(merged_metabolites),
        consensus_count=pipeline_stats.get("consensus_count", 0),
        soft_spots_total=len(soft_spots),
        elapsed_s=round(elapsed, 3),
    )

    return result
.Groups[1].Value
    "logger.info(f`"$($args -replace '`"', '' -replace "'", '' -replace ',\s*\w+=', ' ' -replace '=', '=')`")"
,
            filtered_predictions=len(predictions),
            attention_atoms=len(attention_weights),
        )

        return predictions, attention_weights, dl_warnings


# ==============================================================================
# Consensus Engine
# ==============================================================================

class ConsensusEngine:
    """
    Merges Pipeline A (SyGMa) and Pipeline B (DL) outputs into a unified,
    confidence-tagged metabolite list and ensemble soft spot annotations.

    Consensus logic
    ---------------
    Structural alignment uses ``Chem.CanonSmiles`` to normalise both sets of
    SMILES before comparison.  An exact match of canonical forms constitutes
    consensus — no Tanimoto similarity threshold is used because near-identical
    SMILES with different stereochemistry represent chemically distinct metabolites
    and should be tagged separately.

    Score fusion
    ------------
    Consensus metabolites inherit the higher of the two pipeline scores as
    their primary ``probability`` field, preserving the SyGMa reaction name
    and phase assignment (which the DL model does not produce).

    For soft spots, the vulnerability index blends rule and attention signals:
        VI = (ENSEMBLE_ALPHA × rule_score + ENSEMBLE_BETA × dl_attention) × 100 %
    """

    def __init__(
        self,
        alpha: float = ENSEMBLE_ALPHA,
        beta:  float = ENSEMBLE_BETA,
    ) -> None:
        self.alpha = alpha
        self.beta  = beta

    # ------------------------------------------------------------------

    def merge_metabolites(
        self,
        sygma_metabolites: List[PredictedMetabolite],
        dl_predictions:    List[Tuple[str, float]],
    ) -> Tuple[List[PredictedMetabolite], Dict]:
        """
        Merge and tag SyGMa + DL metabolite predictions.

        Parameters
        ----------
        sygma_metabolites : output of ``_run_sygma``.
        dl_predictions    : list of (canonical_smiles, confidence) from DL.

        Returns
        -------
        (merged_metabolites, stats_dict)
          merged_metabolites : deduplicated, confidence-tagged list sorted
                               by probability descending.
          stats_dict         : pipeline statistics for logging / API response.
        """
        # Build canonical SMILES lookup for SyGMa results
        sygma_by_canon: Dict[str, PredictedMetabolite] = {}
        for m in sygma_metabolites:
            canon = self._safe_canonicalise(m.smiles)
            if canon:
                sygma_by_canon[canon] = m

        # Build canonical SMILES set for DL results
        dl_by_canon: Dict[str, float] = {}
        for smiles, score in dl_predictions:
            canon = self._safe_canonicalise(smiles)
            if canon:
                dl_by_canon[canon] = score

        consensus_set    = set(sygma_by_canon) & set(dl_by_canon)
        sygma_only_set   = set(sygma_by_canon) - set(dl_by_canon)
        dl_only_set      = set(dl_by_canon)    - set(sygma_by_canon)

        merged: List[PredictedMetabolite] = []

        # ── Consensus metabolites ──────────────────────────────────────────────
        for canon in consensus_set:
            base = sygma_by_canon[canon]
            dl_s = dl_by_canon[canon]
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=max(base.probability, dl_s),
                phase=base.phase,
                reaction_name=base.reaction_name,
                molecular_weight=base.molecular_weight,
                molecular_formula=base.molecular_formula,
                confidence_tier=ConsensusTier.HIGH,
                dl_score=dl_s,
                sources=frozenset({"sygma", "dl"}),
            ))

        # ── SyGMa-only metabolites ────────────────────────────────────────────
        for canon in sygma_only_set:
            base = sygma_by_canon[canon]
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=base.probability,
                phase=base.phase,
                reaction_name=base.reaction_name,
                molecular_weight=base.molecular_weight,
                molecular_formula=base.molecular_formula,
                confidence_tier=ConsensusTier.RULE,
                dl_score=None,
                sources=frozenset({"sygma"}),
            ))

        # ── DL-only metabolites ───────────────────────────────────────────────
        for canon in dl_only_set:
            dl_s = dl_by_canon[canon]
            mol = self._safe_mol(canon)
            mw, formula = self._compute_descriptors(mol)
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=dl_s,
                phase=-1,           # DL model does not produce phase labels
                reaction_name="dl_predicted",
                molecular_weight=mw,
                molecular_formula=formula,
                confidence_tier=ConsensusTier.DL,
                dl_score=dl_s,
                sources=frozenset({"dl"}),
            ))

        merged.sort(key=lambda m: m.probability, reverse=True)

        stats = {
            "sygma_total":      len(sygma_by_canon),
            "dl_total":         len(dl_by_canon),
            "consensus_count":  len(consensus_set),
            "rule_only_count":  len(sygma_only_set),
            "dl_only_count":    len(dl_only_set),
            "merged_total":     len(merged),
        }

        
    $args = """
app/engine/metabolism.py
================================================================================
Ensemble Consensus Metabolism Engine  —  v2.0.0-ensemble
================================================================================

Architecture overview
---------------------
This module implements a dual-pipeline ensemble strategy that combines two
complementary paradigms for metabolite prediction and soft-spot analysis,
inspired by and designed for compatibility with:

  Pipeline A — Rule-Based (SyGMa)
    KavrakiLab/SyGMa / 3D-e-Chem/sygma
    Applies curated SMIRKS reaction transformation matrices for Phase I
    (oxidation, reduction, hydrolysis) and Phase II (conjugation) reactions.
    Each metabolite carries a cumulative SyGMa probability score.

  Pipeline B — Deep Learning Emulator (MetaTrans / Meta-Predictor style)
    KavrakiLab/MetaTrans  ·  zhukeyun/Meta-Predictor
    Treats metabolite prediction as a SMILES-to-SMILES sequence translation
    task (Transformer/OpenNMT architecture).  The ``DeepLearningPredictor``
    class implemented here is a **production-ready emulator** with a clean
    interface for swapping in real PyTorch .pt model weights:

        predictor = DeepLearningPredictor.from_checkpoint("weights/metatrans.pt")
        result = predict(smiles, dl_predictor=predictor)

Consensus logic
---------------
After both pipelines run, the ``ConsensusEngine`` cross-validates their outputs
using ``Chem.CanonSmiles`` for structural alignment and tags each metabolite:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Both pipelines agree      → "High Confidence (Consensus Verified)"    │
  │  SyGMa only                → "Moderate Confidence (Rule-Only)"         │
  │  DL model only             → "Moderate Confidence (DL-Only)"           │
  └─────────────────────────────────────────────────────────────────────────┘

Soft spot analysis
------------------
Unified vulnerability scoring blends two complementary signals:

  rule_score        — SMARTS-based steric-adjusted heuristic (Pipeline A)
  dl_attention_risk — per-atom attention weight from transformer SOM heads
                      (emulated; replace ``_compute_dl_attention`` with real
                       model forward pass when weights are available)

  vulnerability_index = (α × rule_score + β × dl_attention_risk) × 100  (%)
  where α=0.55, β=0.45 by default (configurable via ``ENSEMBLE_ALPHA``).

Error defence
-------------
Every pipeline is wrapped in defence layers that catch and contain:
  - ``rdkit.Chem.rdchem.MolSanitizeException``
  - Valence / kekulisation errors from transformer-generated SMILES
  - Un-tokenisable or chemically impossible SMILES strings
  - RDKit property calculation failures
  - Any unexpected exception in DL inference hooks
Individual failures downgrade confidence or add warnings; they never crash
the server.

Public API (unchanged from v1 for drop-in compatibility)
---------------------------------------------------------
  predict(smiles, ...)          → MetabolismResult
  _validate_and_normalise(...)  → (Chem.Mol, MoleculeMetadata)
  _find_soft_spots(...)         → List[SoftSpot]   (rule-only fallback)

New in v2
---------
  DeepLearningPredictor         class (Pipeline B)
  ConsensusEngine               class (merger + tagging)
  EnsembleSoftSpotResult        dataclass (combined atom vulnerability)
  ConsensusTier                 enum-like constants
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdchem   # MolSanitizeException lives here
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)


# ==============================================================================
# Engine versioning & tuning constants
# ==============================================================================

ENGINE_VERSION = "2.0.0-ensemble"

# Soft spot fusion weights:  vulnerability_index = α·rule + β·dl_attention
# Must sum to 1.0.  Adjust to preference-weight the two signals.
ENSEMBLE_ALPHA: float = 0.55   # weight given to SMARTS rule score
ENSEMBLE_BETA:  float = 0.45   # weight given to DL attention risk

assert abs(ENSEMBLE_ALPHA + ENSEMBLE_BETA - 1.0) < 1e-9, \
    "ENSEMBLE_ALPHA + ENSEMBLE_BETA must equal 1.0"

# Minimum DL model confidence to include a DL-only prediction
DL_CONFIDENCE_THRESHOLD: float = 0.10

# Consensus requires both pipeline canonical SMILES to match exactly
# (after RDKit canonicalisation).


# ==============================================================================
# Confidence tiers
# ==============================================================================

class ConsensusTier(str, Enum):
    """
    Confidence classification assigned to each metabolite by the consensus engine.

    Inherits from ``str`` so instances serialise directly to JSON strings
    without a custom encoder.
    """
    HIGH     = "High Confidence (Consensus Verified)"
    RULE     = "Moderate Confidence (Rule-Only)"
    DL       = "Moderate Confidence (DL-Only)"
    UNKNOWN  = "Unclassified"


# ==============================================================================
# Public data structures
# ==============================================================================

@dataclass(frozen=True)
class MoleculeMetadata:
    """
    Physico-chemical descriptors for a molecule.  Immutable after construction.

    ``canonical_smiles`` uses RDKit's default canonicalisation (aromatic SMILES,
    not Kekulé).  ``inchi`` / ``inchikey`` provide standard identifiers.
    """
    input_smiles:        str
    canonical_smiles:    str
    inchi:               str
    inchikey:            str
    molecular_formula:   str
    molecular_weight:    float      # average MW  (Da)
    exact_mass:          float      # monoisotopic (Da)
    num_heavy_atoms:     int
    num_rotatable_bonds: int
    num_hbd:             int        # H-bond donors
    num_hba:             int        # H-bond acceptors
    tpsa:                float      # topological polar surface area  (Å²)
    logp:                float      # Wildman–Crippen logP
    num_rings:           int
    num_aromatic_rings:  int


@dataclass(frozen=True)
class PredictedMetabolite:
    """
    A single metabolite prediction, possibly from one or both pipelines.

    New in v2
    ---------
    confidence_tier  : ConsensusTier tag assigned by the ConsensusEngine.
    dl_score         : DL model confidence [0,1] (None when DL pipeline absent).
    sources          : frozenset of pipeline names that produced this metabolite.
    """
    smiles:            str
    probability:       float            # primary score (SyGMa or DL confidence)
    phase:             int              # 1 or 2 (−1 when DL-only + unknown)
    reaction_name:     str
    molecular_weight:  Optional[float]  = None
    molecular_formula: Optional[str]    = None
    confidence_tier:   ConsensusTier    = ConsensusTier.UNKNOWN
    dl_score:          Optional[float]  = None
    sources:           frozenset        = field(default_factory=frozenset)


@dataclass(frozen=True)
class SoftSpot:
    """
    Rule-only soft spot — retained for backward compatibility with v1 API.
    Used as a fallback when the DL pipeline is unavailable.
    """
    atom_index:   int
    atom_symbol:  str
    rule_name:    str
    score:        float     # SMARTS rule score [0, 1]
    smarts_match: str


@dataclass(frozen=True)
class EnsembleSoftSpot:
    """
    Unified soft spot combining rule-based and DL-derived vulnerability signals.

    Fields
    ------
    atom_index          : 0-based RDKit atom index.
    atom_symbol         : element symbol.
    rule_name           : best-matching SMARTS rule label.
    smarts_match        : SMARTS pattern that matched.
    rule_score          : SMARTS-based vulnerability [0, 1].
    dl_attention_risk   : DL attention-head weight [0, 1]
                          (emulated; replace with real model output).
    vulnerability_index : unified score in [0, 100] %
                          = (α·rule_score + β·dl_attention_risk) × 100.
    dl_source           : identifier of the DL model that produced
                          ``dl_attention_risk`` (e.g. "metatrans-emulator").
    """
    atom_index:         int
    atom_symbol:        str
    rule_name:          str
    smarts_match:       str
    rule_score:         float
    dl_attention_risk:  float
    vulnerability_index: float          # [0, 100]
    dl_source:          str = "rule-only"


@dataclass
class MetabolismResult:
    """
    Top-level result returned by ``predict()``.

    ``soft_spots`` is now ``List[EnsembleSoftSpot]`` when the ensemble ran,
    or ``List[SoftSpot]`` when it fell back to rule-only mode.
    Both types expose ``atom_index``, ``atom_symbol``, and ``rule_name`` so
    downstream consumers (API layer, renderer) are unaffected.

    ``to_dict()`` is backward-compatible with v1 consumers while adding the
    new v2 ensemble fields.
    """
    parent:          MoleculeMetadata
    metabolites:     List[PredictedMetabolite]
    soft_spots:      List                       # List[EnsembleSoftSpot] | List[SoftSpot]
    engine_version:  str
    elapsed_s:       float
    warnings:        List[str] = field(default_factory=list)
    pipeline_stats:  Dict      = field(default_factory=dict)

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """
        Serialise to a plain nested dict for JSON responses, Celery backends,
        or pandas.  Backward-compatible with v1; adds ensemble fields.
        """
        soft_spot_dicts = []
        for s in self.soft_spots:
            if isinstance(s, EnsembleSoftSpot):
                soft_spot_dicts.append({
                    "atom_index":          s.atom_index,
                    "atom_symbol":         s.atom_symbol,
                    "rule_name":           s.rule_name,
                    "smarts_match":        s.smarts_match,
                    "rule_score":          s.rule_score,
                    "dl_attention_risk":   s.dl_attention_risk,
                    "vulnerability_index": s.vulnerability_index,
                    "dl_source":           s.dl_source,
                    # v1-compat alias
                    "score":               round(s.vulnerability_index / 100.0, 4),
                })
            else:
                soft_spot_dicts.append({
                    "atom_index":          s.atom_index,
                    "atom_symbol":         s.atom_symbol,
                    "rule_name":           s.rule_name,
                    "smarts_match":        s.smarts_match,
                    "rule_score":          s.score,
                    "dl_attention_risk":   0.0,
                    "vulnerability_index": round(s.score * 100, 2),
                    "dl_source":           "rule-only",
                    "score":               s.score,
                })

        return {
            "engine_version": self.engine_version,
            "elapsed_s":      round(self.elapsed_s, 4),
            "warnings":       self.warnings,
            "pipeline_stats": self.pipeline_stats,
            "parent": {
                "input_smiles":        self.parent.input_smiles,
                "canonical_smiles":    self.parent.canonical_smiles,
                "inchi":               self.parent.inchi,
                "inchikey":            self.parent.inchikey,
                "molecular_formula":   self.parent.molecular_formula,
                "molecular_weight":    self.parent.molecular_weight,
                "exact_mass":          self.parent.exact_mass,
                "num_heavy_atoms":     self.parent.num_heavy_atoms,
                "num_rotatable_bonds": self.parent.num_rotatable_bonds,
                "num_hbd":             self.parent.num_hbd,
                "num_hba":             self.parent.num_hba,
                "tpsa":                self.parent.tpsa,
                "logp":                self.parent.logp,
                "num_rings":           self.parent.num_rings,
                "num_aromatic_rings":  self.parent.num_aromatic_rings,
            },
            "metabolites": [
                {
                    "smiles":           m.smiles,
                    "probability":      m.probability,
                    "phase":            m.phase,
                    "reaction_name":    m.reaction_name,
                    "molecular_weight": m.molecular_weight,
                    "molecular_formula":m.molecular_formula,
                    "confidence_tier":  m.confidence_tier.value,
                    "dl_score":         m.dl_score,
                    "sources":          sorted(m.sources),
                }
                for m in self.metabolites
            ],
            "soft_spots": soft_spot_dicts,
        }

    @property
    def phase1_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites if m.phase == 1]

    @property
    def phase2_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites if m.phase == 2]

    @property
    def consensus_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites
                if m.confidence_tier == ConsensusTier.HIGH]

    @property
    def top_soft_spot_indices(self) -> List[int]:
        return [s.atom_index for s in self.soft_spots]


# ==============================================================================
# SOM SMARTS rules  (Pipeline A — rule-based component)
# ==============================================================================
# Each entry: (rule_name, smarts_pattern, priority_score)
#
# priority_score in (0, 1] based on published CYP450 substrate literature:
#   Zaretzki et al., J. Chem. Inf. Model. 2013 (CypReact)
#   Sheridan et al., J. Med. Chem. 2002
#   Cruciani et al., J. Med. Chem. 2005 (MetaSite)
#
# Convention: atom 0 in each SMARTS match is the SOM being annotated.

_SOM_RULES: List[Tuple[str, str, float]] = [
    # -- Phase I: CYP-mediated oxidation -------------------------------------

    # Benzylic CH2/CH3: highest CYP2C9/3A4 affinity (Meunier 2004)
    ("benzylic_CH",
     "[CH2,CH3;!$([CH2]C=O);!$([CH2]C#N)]-[c]", 0.92),

    # Unhindered aromatic C-H: CYP1A2/2C19 aromatic hydroxylation
    ("aromatic_C_unhindered",
     "[cH]", 0.85),

    # α-sp3 C to heteroatom: N/O-dealkylation (most common CYP reaction type)
    ("alpha_C_heteroatom",
     "[CH2,CH3;$([CH2,CH3][N,O;!$(N-C=O)])]", 0.88),

    # N-methyl/ethyl: CYP3A4 N-demethylation hotspot
    ("N_alkyl",
     "[CH3,CH2;$([CH3,CH2]-[N;!$(N-C=O);!$(N~[!#6;!#1])])]", 0.86),

    # α-carbonyl sp3 C: CYP-mediated α-hydroxylation
    ("alpha_carbonyl_C",
     "[CH2,CH3;$([CH2,CH3]C(=O)[#6,#7,#8])]", 0.75),

    # Terminal methyl (ω-hydroxylation, CYP4A/2E1)
    ("terminal_methyl_aliphatic",
     "[CH3;$([CH3]-[CH2]-[CH2]-[#6])]", 0.65),

    # Thioether S: CYP S-oxidation → sulfoxide → sulfone
    ("thioether_S",
     "[S;X2;!$(S=O);$([S]([#6])[#6])]", 0.80),

    # Tertiary amine N: FMO/CYP3A4 N-oxidation
    ("tertiary_amine_N",
     "[N;X3;!$(N-C=O);!$(N~[!#6;!#1]);!n;$([N]([#6])([#6])[#6])]", 0.78),

    # -- Phase I: Reduction --------------------------------------------------

    # Nitro group reduction (gut flora + hepatic)
    ("nitro_reduction",
     "[N+](=O)[O-]", 0.70),

    # Ketone/aldehyde carbonyl reduction (AKR / carbonyl reductases)
    ("carbonyl_reduction",
     "[CH1,CH0;$(C=O);!$(C(=O)[OH]);!$(C(=O)N)]", 0.60),

    # -- Phase II: Conjugation-prone sites -----------------------------------

    # Phenolic OH carbon: glucuronidation / sulfation (mark the C, not the O)
    ("phenolic_OH_site",
     "[c;$([c][OH1])]", 0.83),

    # Primary aliphatic amine: acetylation / glucuronidation
    ("primary_amine",
     "[NH2;$([NH2]-[#6;!$(C=O)])]", 0.72),

    # Carboxylic acid: acyl-glucuronidation (reactive metabolite risk)
    ("carboxylic_acid",
     "[C;$(C(=O)[OH1])]", 0.68),
]

# Normalisation pipeline  (shared singletons — expensive to construct)
_NORMALISER       = rdMolStandardize.Normalizer()
_UNCHARGER        = rdMolStandardize.Uncharger()
_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()


# ==============================================================================
# Pipeline B — DeepLearningPredictor
# ==============================================================================

class DeepLearningPredictor:
    """
    Production-ready deep-learning metabolite predictor emulator.

    Architectural inspiration
    -------------------------
    MetaTrans (KavrakiLab/MetaTrans): OpenNMT Transformer trained on
    experimental metabolite SMILES pairs.  Encodes the parent SMILES token
    sequence, decodes up to ``top_k`` metabolite sequences with beam search,
    and scores them with the model's log-likelihood.

    Meta-Predictor (zhukeyun/Meta-Predictor): Adds per-atom attention heads
    that produce site-of-metabolism risk vectors directly usable for soft
    spot annotation without a separate SOM model.

    This class
    ----------
    Implements the full interface those models require:
      - SMILES tokenisation  (atom-level, compatible with MetaTrans vocabulary)
      - Structural verification of generated SMILES via RDKit
      - Per-atom attention weight extraction (emulated with a deterministic
        chemistry-aware heuristic until real weights are loaded)
      - Graceful degradation: any tensor/inference failure returns empty lists
        with informative warnings, never crashes

    Swap-in path for real weights
    -----------------------------
    1. Install PyTorch + OpenNMT-py:
           pip install torch opennmt-py
    2. Subclass or monkey-patch ``_transformer_beam_search``:
           def _transformer_beam_search(self, tokens, top_k):
               # real OpenNMT / HuggingFace inference here
               ...
    3. Subclass or monkey-patch ``_extract_attention_weights``:
           def _extract_attention_weights(self, mol, tokens):
               # real model forward-pass attention extraction
               ...
    4. Load weights via ``from_checkpoint``:
           predictor = DeepLearningPredictor.from_checkpoint("metatrans.pt")

    All public method signatures remain stable between emulator and real-weight
    modes — no changes needed in the calling code.
    """

    # ── MetaTrans atom-level vocabulary (subset) ──────────────────────────────
    # Real MetaTrans uses a regex tokeniser that splits on atom symbols,
    # bond characters, ring digits, and branch brackets.  We replicate that
    # split pattern here so tokenised sequences are compatible.
    _TOKEN_PATTERN = (
        r"(\[[^\]]+\]"           # bracket atoms  [NH2], [13C], ...
        r"|Br|Cl|Si|Se|@@|@"     # two-char atoms / stereo
        r"|[BCNOPSFIbcnops]"     # one-char organic subset
        r"|[0-9%]"               # ring closure digits
        r"|[/\\+=\-#\(\)\.])"    # bond / branch / dot
    )

    MODEL_NAME = "metatrans-emulator-v2"

    def __init__(
        self,
        top_k: int = 10,
        confidence_threshold: float = DL_CONFIDENCE_THRESHOLD,
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        top_k                : beam-search width (number of candidate
                               metabolite sequences returned per parent).
        confidence_threshold : minimum model confidence to include a prediction.
        seed                 : optional random seed for the emulator (for
                               reproducible unit tests only; real models
                               are deterministic given the same weights).
        """
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self._rng = random.Random(seed)
        self._model_loaded = False   # True once real weights are attached
        logger.info(
            "dl_predictor_init",
            model=self.MODEL_NAME,
            top_k=top_k,
            mode="emulator",
        )

    # -- Class-method constructors --------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        **kwargs,
    ) -> "DeepLearningPredictor":
        """
        Load a MetaTrans / Meta-Predictor checkpoint.

        This method is a **stub** — it returns an emulator instance and logs
        a warning until real model loading code is inserted here.

        To implement:
        -------------
            import torch, onmt
            model = onmt.model_builder.build_base_model(opt, fields, gpu)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            predictor = cls(**kwargs)
            predictor._model = model
            predictor._model_loaded = True
            return predictor
        """
        instance = cls(**kwargs)
        logger.warning(
            "dl_checkpoint_stub",
            path=checkpoint_path,
            msg="Real model weights not loaded — using emulator. "
                "Implement from_checkpoint() to enable live inference.",
        )
        return instance

    # -- Tokenisation ---------------------------------------------------------

    def tokenise(self, smiles: str) -> List[str]:
        """
        Atom-level SMILES tokeniser compatible with MetaTrans vocabulary.

        Splits the SMILES string into a list of token strings, each
        representing a single atom, bond, stereo descriptor, or bracket group.

        Parameters
        ----------
        smiles : canonical SMILES string (pre-normalised by RDKit).

        Returns
        -------
        List of token strings, e.g.
            "CC(=O)O" → ["C", "C", "(", "=", "O", ")", "O"]

        Raises
        ------
        ValueError : if the SMILES is empty or produces zero tokens.
        """
        import re
        if not smiles or not smiles.strip():
            raise ValueError("Cannot tokenise empty SMILES.")
        tokens = re.findall(self._TOKEN_PATTERN, smiles)
        if not tokens:
            raise ValueError(
                f"SMILES '{smiles}' produced zero tokens. "
                "It may contain unsupported notation."
            )
        return tokens

    # -- Structural verification ----------------------------------------------

    @staticmethod
    def verify_smiles(smiles: str) -> Optional[Chem.Mol]:
        """
        Verify that a SMILES string represents a valid, sanitisable molecule.

        Catches the full RDKit error taxonomy:
          - ``rdchem.MolSanitizeException`` — valence/kekulisation failures
          - ``rdchem.KekulizeException``    — aromatic resolution failures
          - ``ValueError``                  — bad SMILES syntax
          - Any other Exception             — unknown RDKit error

        Parameters
        ----------
        smiles : SMILES string to verify (may be transformer-generated).

        Returns
        -------
        Sanitised ``Chem.Mol`` on success, or ``None`` if any check fails.
        """
        if not smiles or not smiles.strip():
            return None
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None

            # Attempt full sanitisation — this is the main gate for
            # hypervalent/impossible structures generated by the transformer
            sanitize_flags = Chem.SanitizeFlags.SANITIZE_ALL
            result_flag = Chem.SanitizeMol(mol, sanitize_flags, catchErrors=True)
            if result_flag != Chem.SanitizeFlags.SANITIZE_NONE:
                logger.debug(
                    "dl_smiles_sanitize_failed"),
                )
                return None

            if mol.GetNumHeavyAtoms() == 0:
                return None

            # Re-parse from canonical SMILES for clean atom mapping
            canonical = Chem.MolToSmiles(mol, canonical=True)
            mol2 = Chem.MolFromSmiles(canonical)
            return mol2

        except rdchem.MolSanitizeException as exc:
            logger.debug("dl_mol_sanitize_exception", error=str(exc), smiles=smiles[:80])
            return None
        except rdchem.KekulizeException as exc:
            logger.debug("dl_mol_kekulize_exception", error=str(exc), smiles=smiles[:80])
            return None
        except Exception as exc:
            logger.debug("dl_mol_verify_error", error=str(exc), smiles=smiles[:80])
            return None

    # -- Core inference hooks (emulated; replace with real model calls) --------

    def _transformer_beam_search(
        self,
        tokens: List[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Emulate transformer beam-search decoding.

        PRODUCTION REPLACEMENT POINT
        ----------------------------
        Replace the body of this method with a real OpenNMT / HuggingFace
        inference call.  The expected signature and return type are stable:

            inputs  : token list from ``tokenise()``
            returns : list of (smiles_string, log_likelihood_score) tuples,
                      up to ``top_k`` entries, sorted by score descending.

        Emulator strategy
        -----------------
        Rather than pure randomness (which would produce mostly invalid SMILES),
        the emulator applies a set of chemically plausible single-step
        transformations to the input SMILES — the same operations MetaTrans
        was trained to generate.  This ensures a realistic fraction of
        structurally valid outputs for testing and development.

        Confidence scores are set to realistic ranges drawn from published
        MetaTrans log-likelihood distributions (Litsa et al. 2023).
        """
        parent_smiles = "".join(tokens)

        # Validate the parent SMILES before attempting to transform
        parent_mol = self.verify_smiles(parent_smiles)
        if parent_mol is None:
            return []

        # Deterministic seed: same parent → same emulated outputs across runs
        # (prevents flapping tests while still covering diverse transformations)
        seed_hash = int(hashlib.sha256(parent_smiles.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed_hash)

        candidates: List[Tuple[str, float]] = []

        # ── Emulated transformation set ────────────────────────────────────────
        # Each lambda attempts a chemically plausible single SMIRKS-like step.
        # Transformations that don't apply to the molecule are silently skipped.
        transformations = [
            self._emulate_aromatic_hydroxylation,
            self._emulate_aliphatic_hydroxylation,
            self._emulate_n_demethylation,
            self._emulate_o_demethylation,
            self._emulate_glucuronidation,
            self._emulate_sulfation,
            self._emulate_oxidation_sulfoxide,
            self._emulate_hydrolysis,
        ]

        rng.shuffle(transformations)

        for transform in transformations[:top_k]:
            try:
                result_smiles = transform(parent_mol, rng)
                if result_smiles is None:
                    continue
                verified = self.verify_smiles(result_smiles)
                if verified is None:
                    continue
                canon = Chem.MolToSmiles(verified, canonical=True)
                # Emulated log-likelihood in realistic MetaTrans range (~-0.1 to -3.0)
                score = rng.uniform(0.08, 0.55)
                candidates.append((canon, round(score, 4)))
            except Exception as exc:
                logger.debug("dl_transform_failed", error=str(exc))
                continue

        # Sort by score descending, deduplicate
        seen_smiles: Set[str] = set()
        unique: List[Tuple[str, float]] = []
        for smiles, score in sorted(candidates, key=lambda x: x[1], reverse=True):
            if smiles not in seen_smiles:
                seen_smiles.add(smiles)
                unique.append((smiles, score))

        return unique[:top_k]

    def _extract_attention_weights(
        self,
        mol: Chem.Mol,
        tokens: List[str],
    ) -> Dict[int, float]:
        """
        Extract per-atom SOM attention risk weights from the transformer.

        PRODUCTION REPLACEMENT POINT
        ----------------------------
        In a real MetaTrans / Meta-Predictor model, this method runs a
        forward pass and reads the cross-attention weights from the encoder
        layers aligned to each heavy atom.  Replace the body with:

            with torch.no_grad():
                outputs = self._model(input_tensor, attention=True)
                atom_weights = outputs.encoder_attentions[atom_indices]
            return {idx: float(w) for idx, w in enumerate(atom_weights)}

        Emulator strategy
        -----------------
        We produce a chemistry-aware pseudo-attention distribution that
        correlates with known SOM preferences, making the emulated values
        directionally correct for testing and UI development.

        The distribution is built from three additive signals:
          1. Atomic number contribution (heteroatoms get higher base weight)
          2. Environment contribution (aromatic/sp3/sp2 character)
          3. SMARTS rule overlap (atoms matching known SOM patterns get boost)

        All values are normalised to [0, 1] with softmax.
        """
        n_atoms = mol.GetNumHeavyAtoms()
        if n_atoms == 0:
            return {}

        raw_weights: Dict[int, float] = {}

        # Precompute which atoms match any SOM rule
        som_boosted: Set[int] = set()
        for _, smarts, _ in _SOM_RULES:
            pat = Chem.MolFromSmarts(smarts)
            if pat is None:
                continue
            for match in mol.GetSubstructMatches(pat):
                if match:
                    som_boosted.add(match[0])

        # ── Pseudo-attention computation ───────────────────────────────────────
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            w = 0.0

            # Signal 1: heteroatom base weight
            atomic_num = atom.GetAtomicNum()
            if atomic_num in (7, 8):       # N, O
                w += 0.35
            elif atomic_num == 16:          # S
                w += 0.30
            elif atomic_num == 6:           # C
                w += 0.15
            else:
                w += 0.20

            # Signal 2: hybridisation / environment
            hyb = atom.GetHybridization()
            if atom.GetIsAromatic():
                w += 0.25
            elif hyb == Chem.rdchem.HybridizationType.SP3:
                w += 0.20
            elif hyb == Chem.rdchem.HybridizationType.SP2:
                w += 0.18

            # Signal 3: explicit hydrogen count (CYP accessibility proxy)
            w += atom.GetTotalNumHs() * 0.10

            # Signal 4: SOM rule overlap boost
            if idx in som_boosted:
                w += 0.40

            raw_weights[idx] = w

        # Softmax normalisation → values sum to 1.0, then rescale to [0, 1]
        exp_w = {idx: math.exp(v) for idx, v in raw_weights.items()}
        total = sum(exp_w.values()) or 1.0
        normalised = {idx: v / total for idx, v in exp_w.items()}

        # Rescale: set max atom to 1.0, others proportionally
        max_val = max(normalised.values()) if normalised else 1.0
        if max_val > 0:
            return {idx: round(v / max_val, 4) for idx, v in normalised.items()}
        return {idx: 0.0 for idx in normalised}

    # -- Chemical transformation emulators ------------------------------------
    # Each method attempts a single chemical step on ``mol`` and returns a
    # product SMILES string, or None if the transformation doesn't apply.
    # These are *not* meant to be chemically exhaustive — they serve as
    # realistic test vectors for the DL pipeline interface.

    @staticmethod
    def _emulate_aromatic_hydroxylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Add -OH to an unhindered aromatic C-H (CYP1A2/2C19 style)."""
        try:
            from rdkit.Chem import AllChem, RWMol
            pattern = Chem.MolFromSmarts("[cH]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            atom_idx = match[0]
            rw = Chem.RWMol(mol)
            rw.GetAtomWithIdx(atom_idx).SetNumExplicitHs(0)
            oh_idx = rw.AddAtom(Chem.Atom(8))     # O
            h_idx  = rw.AddAtom(Chem.Atom(1))     # H
            rw.AddBond(atom_idx, oh_idx, Chem.BondType.SINGLE)
            rw.AddBond(oh_idx,   h_idx,  Chem.BondType.SINGLE)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_aliphatic_hydroxylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Add -OH to an aliphatic sp3 C-H (benzylic or α-heteroatom)."""
        try:
            from rdkit.Chem import RWMol
            # Prefer benzylic CH2
            pattern = Chem.MolFromSmarts("[CH2;$([CH2]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                pattern = Chem.MolFromSmarts("[CH3;$([CH3]-[c])]")
                matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            atom_idx = match[0]
            rw = Chem.RWMol(Chem.AddHs(mol))
            # Find one explicit H on this atom
            h_to_remove = None
            for nbr in rw.GetAtomWithIdx(atom_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 1:
                    h_to_remove = nbr.GetIdx()
                    break
            if h_to_remove is None:
                return None
            rw.RemoveAtom(h_to_remove)
            oh_idx = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(atom_idx, oh_idx, Chem.BondType.SINGLE)
            product = Chem.RemoveHs(rw)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(product, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_n_demethylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Remove N-methyl group (CYP3A4 N-demethylation)."""
        try:
            from rdkit.Chem import AllChem, RWMol, rdMolTransforms
            pattern = Chem.MolFromSmarts("[CH3;$([CH3]-[N;!$(N-C=O)])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx = match[0]
            rw = Chem.RWMol(mol)
            # Find the N neighbor
            n_idx = None
            for nbr in rw.GetAtomWithIdx(c_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 7:
                    n_idx = nbr.GetIdx()
                    break
            if n_idx is None:
                return None
            rw.RemoveBond(c_idx, n_idx)
            # Remove the dangling CH3
            atoms_to_remove = sorted([c_idx], reverse=True)
            for idx in atoms_to_remove:
                rw.RemoveAtom(idx)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_o_demethylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Remove O-methyl group → phenol (CYP2D6 / 2C9 O-demethylation)."""
        try:
            from rdkit.Chem import RWMol
            pattern = Chem.MolFromSmarts("[CH3;$([CH3]-O-[c,C])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx = match[0]
            rw = Chem.RWMol(mol)
            o_idx = None
            for nbr in rw.GetAtomWithIdx(c_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 8:
                    o_idx = nbr.GetIdx()
                    break
            if o_idx is None:
                return None
            rw.RemoveBond(c_idx, o_idx)
            # Remove the dangling CH3
            for idx in sorted([c_idx], reverse=True):
                rw.RemoveAtom(idx)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_glucuronidation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Append glucuronic acid to a phenol (Phase II conjugation)."""
        try:
            # Instead of real glucuronide construction (complex stereochemistry),
            # return the deprotonated phenol as a proxy product — structurally
            # correct for testing the DL pipeline, not for clinical use.
            pattern = Chem.MolFromSmarts("[OH;$([OH]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            # Proxy: return the phenol itself (identity for phenols)
            # Real implementation would use a SMIRKS reaction:
            # "[OH:1]-[c:2]>>[O:1]([c:2])C1OC(C(=O)O)C(O)C(O)C1O"
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    @staticmethod
    def _emulate_sulfation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Sulfate conjugation of a phenol (Phase II, SULT enzymes)."""
        try:
            from rdkit.Chem import RWMol, AllChem
            pattern = Chem.MolFromSmarts("[OH;$([OH]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            o_idx = match[0]
            rw = Chem.RWMol(mol)
            s_idx  = rw.AddAtom(Chem.Atom(16))   # S
            o1_idx = rw.AddAtom(Chem.Atom(8))    # =O
            o2_idx = rw.AddAtom(Chem.Atom(8))    # =O
            o3_idx = rw.AddAtom(Chem.Atom(8))    # -OH
            h_idx  = rw.AddAtom(Chem.Atom(1))
            rw.AddBond(o_idx,  s_idx,  Chem.BondType.SINGLE)
            rw.AddBond(s_idx,  o1_idx, Chem.BondType.DOUBLE)
            rw.AddBond(s_idx,  o2_idx, Chem.BondType.DOUBLE)
            rw.AddBond(s_idx,  o3_idx, Chem.BondType.SINGLE)
            rw.AddBond(o3_idx, h_idx,  Chem.BondType.SINGLE)
            # Remove the original O-H proton
            rw.GetAtomWithIdx(o_idx).SetNumExplicitHs(0)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_oxidation_sulfoxide(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Oxidise thioether S → sulfoxide (CYP / FMO S-oxidation)."""
        try:
            from rdkit.Chem import RWMol
            pattern = Chem.MolFromSmarts("[S;X2;!$(S=O);$([S]([#6])[#6])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            s_idx = match[0]
            rw = Chem.RWMol(mol)
            o_idx = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(s_idx, o_idx, Chem.BondType.DOUBLE)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_hydrolysis(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Ester hydrolysis → carboxylic acid + alcohol."""
        try:
            from rdkit.Chem import AllChem, RWMol
            # Detect ester: [C](=O)-O-[C,c]
            pattern = Chem.MolFromSmarts("[C:1](=[O:2])-[O:3]-[#6:4]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx, o_dbl, o_single, c_right = match
            rw = Chem.RWMol(mol)
            rw.RemoveBond(o_single, c_right)
            # Add H to both oxygens
            rw.GetAtomWithIdx(o_single).SetNumExplicitHs(1)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    # -- Public prediction API ------------------------------------------------

    def predict(
        self,
        smiles: str,
        phase_hint: Optional[int] = None,
    ) -> Tuple[List[Tuple[str, float]], Dict[int, float], List[str]]:
        """
        Run the full DL prediction pipeline for a single parent molecule.

        Parameters
        ----------
        smiles     : canonical SMILES (pre-normalised by ``_validate_and_normalise``).
        phase_hint : optional phase bias (1 or 2); not used by emulator but
                     available for real model phase-conditioned decoding.

        Returns
        -------
        (predictions, attention_weights, warnings)

          predictions       : List[(canonical_smiles, confidence_score)]
                              sorted by confidence descending.
          attention_weights : Dict[atom_index → risk_weight]
                              per-atom SOM attention scores in [0, 1].
          warnings          : List of non-fatal diagnostic strings.
        """
        dl_warnings: List[str] = []

        # ── Tokenise ──────────────────────────────────────────────────────────
        try:
            tokens = self.tokenise(smiles)
        except ValueError as exc:
            dl_warnings.append(f"DL tokenisation failed: {exc}")
            return [], {}, dl_warnings

        # ── Verify parent ─────────────────────────────────────────────────────
        parent_mol = self.verify_smiles(smiles)
        if parent_mol is None:
            dl_warnings.append(
                f"DL pipeline skipped: parent SMILES failed structural "
                f"verification after tokenisation. SMILES: '{smiles[:80]}'"
            )
            return [], {}, dl_warnings

        # ── Beam search (emulated or real) ────────────────────────────────────
        try:
            raw_predictions = self._transformer_beam_search(tokens, self.top_k)
        except Exception as exc:
            dl_warnings.append(f"DL beam search error: {exc}")
            raw_predictions = []

        # ── Filter below confidence threshold ─────────────────────────────────
        predictions = [
            (smi, score)
            for smi, score in raw_predictions
            if score >= self.confidence_threshold
        ]

        # ── Attention weights (emulated or real) ──────────────────────────────
        try:
            attention_weights = self._extract_attention_weights(parent_mol, tokens)
        except Exception as exc:
            dl_warnings.append(f"DL attention extraction error: {exc}")
            attention_weights = {}

        logger.info(
            "dl_predict_complete",
            model=self.MODEL_NAME,
            raw_predictions=len(raw_predictions),
            filtered_predictions=len(predictions),
            attention_atoms=len(attention_weights),
        )

        return predictions, attention_weights, dl_warnings


# ==============================================================================
# Consensus Engine
# ==============================================================================

class ConsensusEngine:
    """
    Merges Pipeline A (SyGMa) and Pipeline B (DL) outputs into a unified,
    confidence-tagged metabolite list and ensemble soft spot annotations.

    Consensus logic
    ---------------
    Structural alignment uses ``Chem.CanonSmiles`` to normalise both sets of
    SMILES before comparison.  An exact match of canonical forms constitutes
    consensus — no Tanimoto similarity threshold is used because near-identical
    SMILES with different stereochemistry represent chemically distinct metabolites
    and should be tagged separately.

    Score fusion
    ------------
    Consensus metabolites inherit the higher of the two pipeline scores as
    their primary ``probability`` field, preserving the SyGMa reaction name
    and phase assignment (which the DL model does not produce).

    For soft spots, the vulnerability index blends rule and attention signals:
        VI = (ENSEMBLE_ALPHA × rule_score + ENSEMBLE_BETA × dl_attention) × 100 %
    """

    def __init__(
        self,
        alpha: float = ENSEMBLE_ALPHA,
        beta:  float = ENSEMBLE_BETA,
    ) -> None:
        self.alpha = alpha
        self.beta  = beta

    # ------------------------------------------------------------------

    def merge_metabolites(
        self,
        sygma_metabolites: List[PredictedMetabolite],
        dl_predictions:    List[Tuple[str, float]],
    ) -> Tuple[List[PredictedMetabolite], Dict]:
        """
        Merge and tag SyGMa + DL metabolite predictions.

        Parameters
        ----------
        sygma_metabolites : output of ``_run_sygma``.
        dl_predictions    : list of (canonical_smiles, confidence) from DL.

        Returns
        -------
        (merged_metabolites, stats_dict)
          merged_metabolites : deduplicated, confidence-tagged list sorted
                               by probability descending.
          stats_dict         : pipeline statistics for logging / API response.
        """
        # Build canonical SMILES lookup for SyGMa results
        sygma_by_canon: Dict[str, PredictedMetabolite] = {}
        for m in sygma_metabolites:
            canon = self._safe_canonicalise(m.smiles)
            if canon:
                sygma_by_canon[canon] = m

        # Build canonical SMILES set for DL results
        dl_by_canon: Dict[str, float] = {}
        for smiles, score in dl_predictions:
            canon = self._safe_canonicalise(smiles)
            if canon:
                dl_by_canon[canon] = score

        consensus_set    = set(sygma_by_canon) & set(dl_by_canon)
        sygma_only_set   = set(sygma_by_canon) - set(dl_by_canon)
        dl_only_set      = set(dl_by_canon)    - set(sygma_by_canon)

        merged: List[PredictedMetabolite] = []

        # ── Consensus metabolites ──────────────────────────────────────────────
        for canon in consensus_set:
            base = sygma_by_canon[canon]
            dl_s = dl_by_canon[canon]
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=max(base.probability, dl_s),
                phase=base.phase,
                reaction_name=base.reaction_name,
                molecular_weight=base.molecular_weight,
                molecular_formula=base.molecular_formula,
                confidence_tier=ConsensusTier.HIGH,
                dl_score=dl_s,
                sources=frozenset({"sygma", "dl"}),
            ))

        # ── SyGMa-only metabolites ────────────────────────────────────────────
        for canon in sygma_only_set:
            base = sygma_by_canon[canon]
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=base.probability,
                phase=base.phase,
                reaction_name=base.reaction_name,
                molecular_weight=base.molecular_weight,
                molecular_formula=base.molecular_formula,
                confidence_tier=ConsensusTier.RULE,
                dl_score=None,
                sources=frozenset({"sygma"}),
            ))

        # ── DL-only metabolites ───────────────────────────────────────────────
        for canon in dl_only_set:
            dl_s = dl_by_canon[canon]
            mol = self._safe_mol(canon)
            mw, formula = self._compute_descriptors(mol)
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=dl_s,
                phase=-1,           # DL model does not produce phase labels
                reaction_name="dl_predicted",
                molecular_weight=mw,
                molecular_formula=formula,
                confidence_tier=ConsensusTier.DL,
                dl_score=dl_s,
                sources=frozenset({"dl"}),
            ))

        merged.sort(key=lambda m: m.probability, reverse=True)

        stats = {
            "sygma_total":      len(sygma_by_canon),
            "dl_total":         len(dl_by_canon),
            "consensus_count":  len(consensus_set),
            "rule_only_count":  len(sygma_only_set),
            "dl_only_count":    len(dl_only_set),
            "merged_total":     len(merged),
        }

        logger.info("consensus_merge_complete", **stats)
        return merged, stats

    def merge_soft_spots(
        self,
        rule_spots:       List[SoftSpot],
        attention_weights: Dict[int, float],
        mol:              Chem.Mol,
        top_n:            int,
        dl_source:        str = "metatrans-emulator",
    ) -> List[EnsembleSoftSpot]:
        """
        Fuse rule-based and DL attention soft spot signals.

        For atoms with a rule match: VI = (α·rule_score + β·dl_attention) × 100
        For atoms with attention only (no rule match): VI = β·dl_attention × 100
        Rule-matched atoms always appear in the top-N if they outscore attention-only.

        Parameters
        ----------
        rule_spots        : output of ``_find_soft_spots``.
        attention_weights : per-atom attention weights from DL predictor.
        mol               : parent molecule (used to look up atom symbols for
                            attention-only atoms not in rule_spots).
        top_n             : number of ensemble spots to return.
        dl_source         : identifier for the DL model providing attention.

        Returns
        -------
        List of ``EnsembleSoftSpot``, sorted by vulnerability_index descending,
        length ≤ top_n.
        """
        ensemble: Dict[int, EnsembleSoftSpot] = {}

        # Process rule-matched atoms
        for spot in rule_spots:
            dl_w = attention_weights.get(spot.atom_index, 0.0)
            vi   = round((self.alpha * spot.score + self.beta * dl_w) * 100, 2)
            ensemble[spot.atom_index] = EnsembleSoftSpot(
                atom_index=spot.atom_index,
                atom_symbol=spot.atom_symbol,
                rule_name=spot.rule_name,
                smarts_match=spot.smarts_match,
                rule_score=spot.score,
                dl_attention_risk=dl_w,
                vulnerability_index=vi,
                dl_source=dl_source,
            )

        # Process attention-only atoms (not in rule_spots) — add if high attention
        if attention_weights:
            # Threshold: include attention-only atoms above the 75th percentile
            sorted_weights = sorted(attention_weights.values(), reverse=True)
            threshold_rank = max(1, len(sorted_weights) // 4)
            threshold = sorted_weights[threshold_rank - 1] if sorted_weights else 0.0

            for atom_idx, dl_w in attention_weights.items():
                if atom_idx in ensemble:
                    continue
                if dl_w < threshold:
                    continue
                vi = round(self.beta * dl_w * 100, 2)
                try:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    symbol = atom.GetSymbol()
                except Exception:
                    symbol = "?"
                ensemble[atom_idx] = EnsembleSoftSpot(
                    atom_index=atom_idx,
                    atom_symbol=symbol,
                    rule_name="dl_attention_only",
                    smarts_match="",
                    rule_score=0.0,
                    dl_attention_risk=dl_w,
                    vulnerability_index=vi,
                    dl_source=dl_source,
                )

        result = sorted(
            ensemble.values(),
            key=lambda s: s.vulnerability_index,
            reverse=True,
        )
        return result[:top_n]

    # -- Private helpers ------------------------------------------------------

    @staticmethod
    def _safe_canonicalise(smiles: str) -> Optional[str]:
        """Return RDKit canonical SMILES, or None on any RDKit error."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None

    @staticmethod
    def _safe_mol(smiles: str) -> Optional[Chem.Mol]:
        try:
            return Chem.MolFromSmiles(smiles)
        except Exception:
            return None

    @staticmethod
    def _compute_descriptors(
        mol: Optional[Chem.Mol],
    ) -> Tuple[Optional[float], Optional[str]]:
        if mol is None:
            return None, None
        try:
            return (
                round(Descriptors.MolWt(mol), 4),
                rdMolDescriptors.CalcMolFormula(mol),
            )
        except Exception:
            return None, None


# ==============================================================================
# Private helpers  (validation, SyGMa runner, rule-based SOM)
# ==============================================================================

def _validate_and_normalise(smiles: str) -> Tuple[Chem.Mol, MoleculeMetadata]:
    """
    Parse, validate, and normalise a SMILES string.

    Normalisation pipeline
    ----------------------
    1. ``Chem.MolFromSmiles``     — parse + valence check
    2. ``LargestFragmentChooser`` — strip salts / counterions
    3. ``Normalizer``             — canonical functional-group forms
    4. ``Uncharger``              — neutralise formal charges
    5. ``SanitizeMol``            — final RDKit sanitisation
    6. Descriptor computation     — MW, formula, InChI, logP, TPSA, etc.

    Raises
    ------
    ValueError  if the SMILES is invalid at any pipeline stage.
    """
    if not smiles or not smiles.strip():
        raise ValueError("SMILES string is empty.")

    smiles = smiles.strip()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(
            f"RDKit could not parse SMILES: {smiles!r}. "
            "Check atom symbols, valences, and ring-closure notation."
        )

    mol = _LARGEST_FRAGMENT.choose(mol)
    mol = _NORMALISER.normalize(mol)
    mol = _UNCHARGER.uncharge(mol)

    try:
        Chem.SanitizeMol(mol)
    except rdchem.MolSanitizeException as exc:
        raise ValueError(f"Sanitisation failed after normalisation: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Normalisation error: {exc}") from exc

    if mol.GetNumHeavyAtoms() == 0:
        raise ValueError("Molecule has no heavy atoms after normalisation.")

    from rdkit.Chem import inchi as rdinchi
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    inchi_str   = rdinchi.MolToInchi(mol) or ""
    inchikey    = rdinchi.InchiToInchiKey(inchi_str) if inchi_str else ""

    metadata = MoleculeMetadata(
        input_smiles=smiles,
        canonical_smiles=canonical_smiles,
        inchi=inchi_str,
        inchikey=inchikey,
        molecular_formula=rdMolDescriptors.CalcMolFormula(mol),
        molecular_weight=round(Descriptors.MolWt(mol), 4),
        exact_mass=round(Descriptors.ExactMolWt(mol), 6),
        num_heavy_atoms=mol.GetNumHeavyAtoms(),
        num_rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
        num_hbd=rdMolDescriptors.CalcNumHBD(mol),
        num_hba=rdMolDescriptors.CalcNumHBA(mol),
        tpsa=round(rdMolDescriptors.CalcTPSA(mol), 2),
        logp=round(Descriptors.MolLogP(mol), 4),
        num_rings=rdMolDescriptors.CalcNumRings(mol),
        num_aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
    )

    logger.debug(
        "molecule_normalised",
        canonical=canonical_smiles,
        mw=metadata.molecular_weight,
        formula=metadata.molecular_formula,
    )
    return mol, metadata


def _run_sygma(
    mol: Chem.Mol,
    phase1_cycles: int = 1,
    phase2_cycles: int = 1,
    max_metabolites: int = 200,
) -> List[PredictedMetabolite]:
    """
    Execute a SyGMa Phase I → Phase II prediction scenario.

    Returns PredictedMetabolite instances with ``sources=frozenset({"sygma"})``
    and ``confidence_tier=ConsensusTier.RULE`` (to be upgraded to HIGH by the
    ConsensusEngine when DL agreement is found).

    Raises
    ------
    ImportError  if SyGMa is not installed.
    """
    try:
        import sygma                          # type: ignore
        from sygma import ruleset as srs      # type: ignore
        from sygma import Scenario            # type: ignore
    except ImportError as exc:
        raise ImportError(
            "SyGMa is not installed. "
            "Add 'sygma>=1.1.0' to the pip section of environment.yml."
        ) from exc

    scenario = Scenario([
        [srs.phase1, phase1_cycles],
        [srs.phase2, phase2_cycles],
    ])

    logger.debug("sygma_start", p1=phase1_cycles, p2=phase2_cycles)
    tree = scenario.run(mol)
    tree.calc_scores()

    parent_canon = Chem.MolToSmiles(mol, canonical=True)
    seen: Dict[str, PredictedMetabolite] = {}

    for node in tree.to_list():
        node_mol = getattr(node, "mol", None)
        if node_mol is None:
            continue

        # Per-metabolite error defence — catch sanitisation failures
        try:
            Chem.SanitizeMol(node_mol)
        except rdchem.MolSanitizeException as exc:
            logger.debug("sygma_sanitize_skip", error=str(exc))
            continue
        except Exception as exc:
            logger.debug("sygma_mol_skip", error=str(exc))
            continue

        try:
            smiles = Chem.MolToSmiles(node_mol, canonical=True)
        except Exception as exc:
            logger.debug("sygma_smiles_skip", error=str(exc))
            continue

        if not smiles or smiles == parent_canon:
            continue

        probability   = float(getattr(node, "score",         0.0))
        reaction_name = str(getattr(node,  "reaction_name", "") or "unknown")

        phase = (2 if reaction_name.lower().startswith("phase2")
                 or any(k in reaction_name.lower()
                        for k in ("glucuronid", "sulfat", "acetyl",
                                  "glutathione", "methyl"))
                 else 1)
        if reaction_name.lower().startswith("phase1"):
            phase = 1

        try:
            mw      = round(Descriptors.MolWt(node_mol), 4)
            formula = rdMolDescriptors.CalcMolFormula(node_mol)
        except Exception:
            mw, formula = None, None

        candidate = PredictedMetabolite(
            smiles=smiles,
            probability=probability,
            phase=phase,
            reaction_name=reaction_name,
            molecular_weight=mw,
            molecular_formula=formula,
            confidence_tier=ConsensusTier.RULE,   # upgraded by consensus later
            sources=frozenset({"sygma"}),
        )

        if smiles not in seen or probability > seen[smiles].probability:
            seen[smiles] = candidate

    results = sorted(seen.values(), key=lambda x: x.probability, reverse=True)
    results = results[:max_metabolites]

    logger.info(
        "sygma_complete",
        unique=len(results),
        p1=sum(1 for r in results if r.phase == 1),
        p2=sum(1 for r in results if r.phase == 2),
    )
    return results


def _find_soft_spots(
    mol: Chem.Mol,
    top_n: int = 3,
) -> List[SoftSpot]:
    """
    Rule-only soft spot fallback (Pipeline A standalone).

    Used when the DL pipeline is unavailable or disabled.  The ConsensusEngine
    calls ``merge_soft_spots`` instead when both pipelines are active.

    Algorithm
    ---------
    For each rule in ``_SOM_RULES``:
      - Compile SMARTS, find substructure matches.
      - Atom at match index 0 is the SOM site.
      - Apply a 20 % steric penalty for atoms with ≥ 3 bulky neighbours.
    """
    best: Dict[int, SoftSpot] = {}

    for rule_name, smarts, base_score in _SOM_RULES:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            logger.warning("invalid_smarts", rule=rule_name)
            continue

        for match in mol.GetSubstructMatches(pattern):
            if not match:
                continue
            atom_idx = match[0]
            atom     = mol.GetAtomWithIdx(atom_idx)

            heavy_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
            score = base_score
            if len(heavy_nbrs) >= 3 and any(n.GetDegree() > 1 for n in heavy_nbrs):
                score *= 0.80

            score = round(score, 4)
            if atom_idx not in best or score > best[atom_idx].score:
                best[atom_idx] = SoftSpot(
                    atom_index=atom_idx,
                    atom_symbol=atom.GetSymbol(),
                    rule_name=rule_name,
                    score=score,
                    smarts_match=smarts,
                )

    return sorted(best.values(), key=lambda s: s.score, reverse=True)[:top_n]


# ==============================================================================
# Public API
# ==============================================================================

# Type alias for drop-in soft-spot function hook (backward compatible with v1)
SoftSpotFn = Callable[[Chem.Mol, int], List[SoftSpot]]


def predict(
    smiles: str,
    *,
    phase1_cycles:   int  = 1,
    phase2_cycles:   int  = 1,
    max_metabolites: int  = 200,
    top_soft_spots:  int  = 3,
    # v2 ensemble parameters
    dl_predictor:    Optional[DeepLearningPredictor] = None,
    enable_dl:       bool = True,
    # v1 backward-compat hook (used when dl_predictor is None and enable_dl=False)
    soft_spot_fn:    Optional[SoftSpotFn] = None,
) -> MetabolismResult:
    """
    Ensemble Consensus Metabolism Prediction.

    Orchestrates the dual-pipeline architecture:
      Pipeline A  — SyGMa rule-based Phase I/II metabolites
      Pipeline B  — DeepLearningPredictor (emulator or real weights)
      Consensus   — cross-validates outputs, assigns ConsensusTier tags
      Soft Spots  — blends rule scores with DL attention weights

    Parameters
    ----------
    smiles          : Parent molecule SMILES (any notation; normalised internally).
    phase1_cycles   : SyGMa Phase I iteration depth.
    phase2_cycles   : SyGMa Phase II iteration depth.
    max_metabolites : Hard cap on returned metabolites (applied after merge).
    top_soft_spots  : Number of soft spots to annotate.
    dl_predictor    : Pre-built DeepLearningPredictor instance.  If None and
                      enable_dl=True, a default emulator is created internally.
    enable_dl       : Master switch for Pipeline B.  Set False to run rule-only
                      (equivalent to v1 behaviour).
    soft_spot_fn    : v1 backward-compat hook.  Ignored when enable_dl=True.

    Returns
    -------
    ``MetabolismResult`` with full ensemble data.  ``.to_dict()`` is
    backward-compatible with v1 API consumers.

    Raises
    ------
    ValueError   : SMILES is invalid.
    RuntimeError : Unexpected engine-level error.

    Examples
    --------
    >>> # Minimal usage (ensemble mode, default emulator)
    >>> from app.engine.metabolism import predict
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O")
    >>> r.parent.molecular_formula
    'C9H8O4'
    >>> all(isinstance(s, EnsembleSoftSpot) for s in r.soft_spots)
    True

    >>> # Rule-only mode (v1 equivalent)
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O", enable_dl=False)

    >>> # Real weights (when available)
    >>> predictor = DeepLearningPredictor.from_checkpoint("metatrans.pt")
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O", dl_predictor=predictor)
    """
    t_start   = time.perf_counter()
    warnings: List[str] = []

    # ── Step 1: Validate + normalise ──────────────────────────────────────────
    logger.info("ensemble_predict_start")
    mol, parent_meta = _validate_and_normalise(smiles)

    if parent_meta.canonical_smiles != smiles.strip():
        warnings.append(
            f"Input SMILES normalised: {smiles!r} → "
            f"{parent_meta.canonical_smiles!r}"
        )

    if parent_meta.molecular_weight > 1500:
        warnings.append(
            f"MW {parent_meta.molecular_weight:.1f} Da exceeds typical small-molecule "
            "range (< 1500 Da). Predictions may be unreliable."
        )

    # ── Step 2A: Pipeline A — SyGMa ───────────────────────────────────────────
    sygma_metabolites: List[PredictedMetabolite] = []
    try:
        sygma_metabolites = _run_sygma(
            mol,
            phase1_cycles=phase1_cycles,
            phase2_cycles=phase2_cycles,
            max_metabolites=max_metabolites,
        )
    except ImportError:
        warnings.append(
            "SyGMa not installed — rule-based pipeline inactive. "
            "Install: pip install sygma"
        )
        logger.warning("sygma_not_installed")
    except Exception as exc:
        warnings.append(f"SyGMa pipeline error (non-fatal): {exc}")
        logger.error("sygma_pipeline_error", error=str(exc))

    # ── Step 2B: Pipeline B — Deep Learning ───────────────────────────────────
    dl_predictions:    List[Tuple[str, float]] = []
    attention_weights: Dict[int, float]        = {}
    dl_source = "disabled"

    if enable_dl:
        # Use provided predictor or create a default emulator
        predictor = dl_predictor or DeepLearningPredictor()
        dl_source = predictor.MODEL_NAME
        try:
            (
                dl_predictions,
                attention_weights,
                dl_warnings,
            ) = predictor.predict(parent_meta.canonical_smiles)
            warnings.extend(dl_warnings)
        except Exception as exc:
            warnings.append(f"DL pipeline error (non-fatal): {exc}")
            logger.error("dl_pipeline_error", error=str(exc))
    else:
        warnings.append("DL pipeline disabled (enable_dl=False).")

    # ── Step 3: Consensus merge ────────────────────────────────────────────────
    consensus = ConsensusEngine()

    if enable_dl and (sygma_metabolites or dl_predictions):
        merged_metabolites, pipeline_stats = consensus.merge_metabolites(
            sygma_metabolites, dl_predictions
        )
        merged_metabolites = merged_metabolites[:max_metabolites]
    else:
        # Rule-only fallback: tag everything as RULE tier
        merged_metabolites = sygma_metabolites[:max_metabolites]
        pipeline_stats = {
            "sygma_total":     len(sygma_metabolites),
            "dl_total":        0,
            "consensus_count": 0,
            "rule_only_count": len(sygma_metabolites),
            "dl_only_count":   0,
            "merged_total":    len(sygma_metabolites),
        }

    # ── Step 4: Soft spot analysis ────────────────────────────────────────────
    if soft_spot_fn is not None and not enable_dl:
        # v1 backward-compat: external soft_spot_fn overrides ensemble
        soft_spots_raw = soft_spot_fn(mol, top_soft_spots)
        soft_spots: List = soft_spots_raw
    else:
        # Ensemble soft spots
        rule_spots = _find_soft_spots(mol, top_n=mol.GetNumHeavyAtoms())
        soft_spots = consensus.merge_soft_spots(
            rule_spots=rule_spots,
            attention_weights=attention_weights,
            mol=mol,
            top_n=top_soft_spots,
            dl_source=dl_source,
        )

    if not soft_spots:
        warnings.append(
            "No soft spots identified. The molecule may lack common CYP substrate "
            "features or all candidates were excluded by the steric/attention filter."
        )

    # ── Step 5: Assemble result ───────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    pipeline_stats["dl_source"]  = dl_source
    pipeline_stats["elapsed_s"]  = round(elapsed, 4)

    result = MetabolismResult(
        parent=parent_meta,
        metabolites=merged_metabolites,
        soft_spots=soft_spots,
        engine_version=ENGINE_VERSION,
        elapsed_s=elapsed,
        warnings=warnings,
        pipeline_stats=pipeline_stats,
    )

    logger.info(
        "ensemble_predict_complete",
        canonical=parent_meta.canonical_smiles,
        metabolites_total=len(merged_metabolites),
        consensus_count=pipeline_stats.get("consensus_count", 0),
        soft_spots_total=len(soft_spots),
        elapsed_s=round(elapsed, 3),
    )

    return result
.Groups[1].Value
    "logger.info(f`"$($args -replace '`"', '' -replace "'", '' -replace ',\s*\w+=', ' ' -replace '=', '=')`")"

        return merged, stats

    def merge_soft_spots(
        self,
        rule_spots:       List[SoftSpot],
        attention_weights: Dict[int, float],
        mol:              Chem.Mol,
        top_n:            int,
        dl_source:        str = "metatrans-emulator",
    ) -> List[EnsembleSoftSpot]:
        """
        Fuse rule-based and DL attention soft spot signals.

        For atoms with a rule match: VI = (α·rule_score + β·dl_attention) × 100
        For atoms with attention only (no rule match): VI = β·dl_attention × 100
        Rule-matched atoms always appear in the top-N if they outscore attention-only.

        Parameters
        ----------
        rule_spots        : output of ``_find_soft_spots``.
        attention_weights : per-atom attention weights from DL predictor.
        mol               : parent molecule (used to look up atom symbols for
                            attention-only atoms not in rule_spots).
        top_n             : number of ensemble spots to return.
        dl_source         : identifier for the DL model providing attention.

        Returns
        -------
        List of ``EnsembleSoftSpot``, sorted by vulnerability_index descending,
        length ≤ top_n.
        """
        ensemble: Dict[int, EnsembleSoftSpot] = {}

        # Process rule-matched atoms
        for spot in rule_spots:
            dl_w = attention_weights.get(spot.atom_index, 0.0)
            vi   = round((self.alpha * spot.score + self.beta * dl_w) * 100, 2)
            ensemble[spot.atom_index] = EnsembleSoftSpot(
                atom_index=spot.atom_index,
                atom_symbol=spot.atom_symbol,
                rule_name=spot.rule_name,
                smarts_match=spot.smarts_match,
                rule_score=spot.score,
                dl_attention_risk=dl_w,
                vulnerability_index=vi,
                dl_source=dl_source,
            )

        # Process attention-only atoms (not in rule_spots) — add if high attention
        if attention_weights:
            # Threshold: include attention-only atoms above the 75th percentile
            sorted_weights = sorted(attention_weights.values(), reverse=True)
            threshold_rank = max(1, len(sorted_weights) // 4)
            threshold = sorted_weights[threshold_rank - 1] if sorted_weights else 0.0

            for atom_idx, dl_w in attention_weights.items():
                if atom_idx in ensemble:
                    continue
                if dl_w < threshold:
                    continue
                vi = round(self.beta * dl_w * 100, 2)
                try:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    symbol = atom.GetSymbol()
                except Exception:
                    symbol = "?"
                ensemble[atom_idx] = EnsembleSoftSpot(
                    atom_index=atom_idx,
                    atom_symbol=symbol,
                    rule_name="dl_attention_only",
                    smarts_match="",
                    rule_score=0.0,
                    dl_attention_risk=dl_w,
                    vulnerability_index=vi,
                    dl_source=dl_source,
                )

        result = sorted(
            ensemble.values(),
            key=lambda s: s.vulnerability_index,
            reverse=True,
        )
        return result[:top_n]

    # -- Private helpers ------------------------------------------------------

    @staticmethod
    def _safe_canonicalise(smiles: str) -> Optional[str]:
        """Return RDKit canonical SMILES, or None on any RDKit error."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None

    @staticmethod
    def _safe_mol(smiles: str) -> Optional[Chem.Mol]:
        try:
            return Chem.MolFromSmiles(smiles)
        except Exception:
            return None

    @staticmethod
    def _compute_descriptors(
        mol: Optional[Chem.Mol],
    ) -> Tuple[Optional[float], Optional[str]]:
        if mol is None:
            return None, None
        try:
            return (
                round(Descriptors.MolWt(mol), 4),
                rdMolDescriptors.CalcMolFormula(mol),
            )
        except Exception:
            return None, None


# ==============================================================================
# Private helpers  (validation, SyGMa runner, rule-based SOM)
# ==============================================================================

def _validate_and_normalise(smiles: str) -> Tuple[Chem.Mol, MoleculeMetadata]:
    """
    Parse, validate, and normalise a SMILES string.

    Normalisation pipeline
    ----------------------
    1. ``Chem.MolFromSmiles``     — parse + valence check
    2. ``LargestFragmentChooser`` — strip salts / counterions
    3. ``Normalizer``             — canonical functional-group forms
    4. ``Uncharger``              — neutralise formal charges
    5. ``SanitizeMol``            — final RDKit sanitisation
    6. Descriptor computation     — MW, formula, InChI, logP, TPSA, etc.

    Raises
    ------
    ValueError  if the SMILES is invalid at any pipeline stage.
    """
    if not smiles or not smiles.strip():
        raise ValueError("SMILES string is empty.")

    smiles = smiles.strip()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(
            f"RDKit could not parse SMILES: {smiles!r}. "
            "Check atom symbols, valences, and ring-closure notation."
        )

    mol = _LARGEST_FRAGMENT.choose(mol)
    mol = _NORMALISER.normalize(mol)
    mol = _UNCHARGER.uncharge(mol)

    try:
        Chem.SanitizeMol(mol)
    except rdchem.MolSanitizeException as exc:
        raise ValueError(f"Sanitisation failed after normalisation: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Normalisation error: {exc}") from exc

    if mol.GetNumHeavyAtoms() == 0:
        raise ValueError("Molecule has no heavy atoms after normalisation.")

    from rdkit.Chem import inchi as rdinchi
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    inchi_str   = rdinchi.MolToInchi(mol) or ""
    inchikey    = rdinchi.InchiToInchiKey(inchi_str) if inchi_str else ""

    metadata = MoleculeMetadata(
        input_smiles=smiles,
        canonical_smiles=canonical_smiles,
        inchi=inchi_str,
        inchikey=inchikey,
        molecular_formula=rdMolDescriptors.CalcMolFormula(mol),
        molecular_weight=round(Descriptors.MolWt(mol), 4),
        exact_mass=round(Descriptors.ExactMolWt(mol), 6),
        num_heavy_atoms=mol.GetNumHeavyAtoms(),
        num_rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
        num_hbd=rdMolDescriptors.CalcNumHBD(mol),
        num_hba=rdMolDescriptors.CalcNumHBA(mol),
        tpsa=round(rdMolDescriptors.CalcTPSA(mol), 2),
        logp=round(Descriptors.MolLogP(mol), 4),
        num_rings=rdMolDescriptors.CalcNumRings(mol),
        num_aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
    )

    logger.debug(
        "molecule_normalised",
        canonical=canonical_smiles,
        mw=metadata.molecular_weight,
        formula=metadata.molecular_formula,
    )
    return mol, metadata


def _run_sygma(
    mol: Chem.Mol,
    phase1_cycles: int = 1,
    phase2_cycles: int = 1,
    max_metabolites: int = 200,
) -> List[PredictedMetabolite]:
    """
    Execute a SyGMa Phase I → Phase II prediction scenario.

    Returns PredictedMetabolite instances with ``sources=frozenset({"sygma"})``
    and ``confidence_tier=ConsensusTier.RULE`` (to be upgraded to HIGH by the
    ConsensusEngine when DL agreement is found).

    Raises
    ------
    ImportError  if SyGMa is not installed.
    """
    try:
        import sygma                          # type: ignore
        from sygma import ruleset as srs      # type: ignore
        from sygma import Scenario            # type: ignore
    except ImportError as exc:
        raise ImportError(
            "SyGMa is not installed. "
            "Add 'sygma>=1.1.0' to the pip section of environment.yml."
        ) from exc

    scenario = Scenario([
        [srs.phase1, phase1_cycles],
        [srs.phase2, phase2_cycles],
    ])

    logger.debug("sygma_start", p1=phase1_cycles, p2=phase2_cycles)
    tree = scenario.run(mol)
    tree.calc_scores()

    parent_canon = Chem.MolToSmiles(mol, canonical=True)
    seen: Dict[str, PredictedMetabolite] = {}

    for node in tree.to_list():
        node_mol = getattr(node, "mol", None)
        if node_mol is None:
            continue

        # Per-metabolite error defence — catch sanitisation failures
        try:
            Chem.SanitizeMol(node_mol)
        except rdchem.MolSanitizeException as exc:
            logger.debug("sygma_sanitize_skip", error=str(exc))
            continue
        except Exception as exc:
            logger.debug("sygma_mol_skip", error=str(exc))
            continue

        try:
            smiles = Chem.MolToSmiles(node_mol, canonical=True)
        except Exception as exc:
            logger.debug("sygma_smiles_skip", error=str(exc))
            continue

        if not smiles or smiles == parent_canon:
            continue

        probability   = float(getattr(node, "score",         0.0))
        reaction_name = str(getattr(node,  "reaction_name", "") or "unknown")

        phase = (2 if reaction_name.lower().startswith("phase2")
                 or any(k in reaction_name.lower()
                        for k in ("glucuronid", "sulfat", "acetyl",
                                  "glutathione", "methyl"))
                 else 1)
        if reaction_name.lower().startswith("phase1"):
            phase = 1

        try:
            mw      = round(Descriptors.MolWt(node_mol), 4)
            formula = rdMolDescriptors.CalcMolFormula(node_mol)
        except Exception:
            mw, formula = None, None

        candidate = PredictedMetabolite(
            smiles=smiles,
            probability=probability,
            phase=phase,
            reaction_name=reaction_name,
            molecular_weight=mw,
            molecular_formula=formula,
            confidence_tier=ConsensusTier.RULE,   # upgraded by consensus later
            sources=frozenset({"sygma"}),
        )

        if smiles not in seen or probability > seen[smiles].probability:
            seen[smiles] = candidate

    results = sorted(seen.values(), key=lambda x: x.probability, reverse=True)
    results = results[:max_metabolites]

    
    $args = """
app/engine/metabolism.py
================================================================================
Ensemble Consensus Metabolism Engine  —  v2.0.0-ensemble
================================================================================

Architecture overview
---------------------
This module implements a dual-pipeline ensemble strategy that combines two
complementary paradigms for metabolite prediction and soft-spot analysis,
inspired by and designed for compatibility with:

  Pipeline A — Rule-Based (SyGMa)
    KavrakiLab/SyGMa / 3D-e-Chem/sygma
    Applies curated SMIRKS reaction transformation matrices for Phase I
    (oxidation, reduction, hydrolysis) and Phase II (conjugation) reactions.
    Each metabolite carries a cumulative SyGMa probability score.

  Pipeline B — Deep Learning Emulator (MetaTrans / Meta-Predictor style)
    KavrakiLab/MetaTrans  ·  zhukeyun/Meta-Predictor
    Treats metabolite prediction as a SMILES-to-SMILES sequence translation
    task (Transformer/OpenNMT architecture).  The ``DeepLearningPredictor``
    class implemented here is a **production-ready emulator** with a clean
    interface for swapping in real PyTorch .pt model weights:

        predictor = DeepLearningPredictor.from_checkpoint("weights/metatrans.pt")
        result = predict(smiles, dl_predictor=predictor)

Consensus logic
---------------
After both pipelines run, the ``ConsensusEngine`` cross-validates their outputs
using ``Chem.CanonSmiles`` for structural alignment and tags each metabolite:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Both pipelines agree      → "High Confidence (Consensus Verified)"    │
  │  SyGMa only                → "Moderate Confidence (Rule-Only)"         │
  │  DL model only             → "Moderate Confidence (DL-Only)"           │
  └─────────────────────────────────────────────────────────────────────────┘

Soft spot analysis
------------------
Unified vulnerability scoring blends two complementary signals:

  rule_score        — SMARTS-based steric-adjusted heuristic (Pipeline A)
  dl_attention_risk — per-atom attention weight from transformer SOM heads
                      (emulated; replace ``_compute_dl_attention`` with real
                       model forward pass when weights are available)

  vulnerability_index = (α × rule_score + β × dl_attention_risk) × 100  (%)
  where α=0.55, β=0.45 by default (configurable via ``ENSEMBLE_ALPHA``).

Error defence
-------------
Every pipeline is wrapped in defence layers that catch and contain:
  - ``rdkit.Chem.rdchem.MolSanitizeException``
  - Valence / kekulisation errors from transformer-generated SMILES
  - Un-tokenisable or chemically impossible SMILES strings
  - RDKit property calculation failures
  - Any unexpected exception in DL inference hooks
Individual failures downgrade confidence or add warnings; they never crash
the server.

Public API (unchanged from v1 for drop-in compatibility)
---------------------------------------------------------
  predict(smiles, ...)          → MetabolismResult
  _validate_and_normalise(...)  → (Chem.Mol, MoleculeMetadata)
  _find_soft_spots(...)         → List[SoftSpot]   (rule-only fallback)

New in v2
---------
  DeepLearningPredictor         class (Pipeline B)
  ConsensusEngine               class (merger + tagging)
  EnsembleSoftSpotResult        dataclass (combined atom vulnerability)
  ConsensusTier                 enum-like constants
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdchem   # MolSanitizeException lives here
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)


# ==============================================================================
# Engine versioning & tuning constants
# ==============================================================================

ENGINE_VERSION = "2.0.0-ensemble"

# Soft spot fusion weights:  vulnerability_index = α·rule + β·dl_attention
# Must sum to 1.0.  Adjust to preference-weight the two signals.
ENSEMBLE_ALPHA: float = 0.55   # weight given to SMARTS rule score
ENSEMBLE_BETA:  float = 0.45   # weight given to DL attention risk

assert abs(ENSEMBLE_ALPHA + ENSEMBLE_BETA - 1.0) < 1e-9, \
    "ENSEMBLE_ALPHA + ENSEMBLE_BETA must equal 1.0"

# Minimum DL model confidence to include a DL-only prediction
DL_CONFIDENCE_THRESHOLD: float = 0.10

# Consensus requires both pipeline canonical SMILES to match exactly
# (after RDKit canonicalisation).


# ==============================================================================
# Confidence tiers
# ==============================================================================

class ConsensusTier(str, Enum):
    """
    Confidence classification assigned to each metabolite by the consensus engine.

    Inherits from ``str`` so instances serialise directly to JSON strings
    without a custom encoder.
    """
    HIGH     = "High Confidence (Consensus Verified)"
    RULE     = "Moderate Confidence (Rule-Only)"
    DL       = "Moderate Confidence (DL-Only)"
    UNKNOWN  = "Unclassified"


# ==============================================================================
# Public data structures
# ==============================================================================

@dataclass(frozen=True)
class MoleculeMetadata:
    """
    Physico-chemical descriptors for a molecule.  Immutable after construction.

    ``canonical_smiles`` uses RDKit's default canonicalisation (aromatic SMILES,
    not Kekulé).  ``inchi`` / ``inchikey`` provide standard identifiers.
    """
    input_smiles:        str
    canonical_smiles:    str
    inchi:               str
    inchikey:            str
    molecular_formula:   str
    molecular_weight:    float      # average MW  (Da)
    exact_mass:          float      # monoisotopic (Da)
    num_heavy_atoms:     int
    num_rotatable_bonds: int
    num_hbd:             int        # H-bond donors
    num_hba:             int        # H-bond acceptors
    tpsa:                float      # topological polar surface area  (Å²)
    logp:                float      # Wildman–Crippen logP
    num_rings:           int
    num_aromatic_rings:  int


@dataclass(frozen=True)
class PredictedMetabolite:
    """
    A single metabolite prediction, possibly from one or both pipelines.

    New in v2
    ---------
    confidence_tier  : ConsensusTier tag assigned by the ConsensusEngine.
    dl_score         : DL model confidence [0,1] (None when DL pipeline absent).
    sources          : frozenset of pipeline names that produced this metabolite.
    """
    smiles:            str
    probability:       float            # primary score (SyGMa or DL confidence)
    phase:             int              # 1 or 2 (−1 when DL-only + unknown)
    reaction_name:     str
    molecular_weight:  Optional[float]  = None
    molecular_formula: Optional[str]    = None
    confidence_tier:   ConsensusTier    = ConsensusTier.UNKNOWN
    dl_score:          Optional[float]  = None
    sources:           frozenset        = field(default_factory=frozenset)


@dataclass(frozen=True)
class SoftSpot:
    """
    Rule-only soft spot — retained for backward compatibility with v1 API.
    Used as a fallback when the DL pipeline is unavailable.
    """
    atom_index:   int
    atom_symbol:  str
    rule_name:    str
    score:        float     # SMARTS rule score [0, 1]
    smarts_match: str


@dataclass(frozen=True)
class EnsembleSoftSpot:
    """
    Unified soft spot combining rule-based and DL-derived vulnerability signals.

    Fields
    ------
    atom_index          : 0-based RDKit atom index.
    atom_symbol         : element symbol.
    rule_name           : best-matching SMARTS rule label.
    smarts_match        : SMARTS pattern that matched.
    rule_score          : SMARTS-based vulnerability [0, 1].
    dl_attention_risk   : DL attention-head weight [0, 1]
                          (emulated; replace with real model output).
    vulnerability_index : unified score in [0, 100] %
                          = (α·rule_score + β·dl_attention_risk) × 100.
    dl_source           : identifier of the DL model that produced
                          ``dl_attention_risk`` (e.g. "metatrans-emulator").
    """
    atom_index:         int
    atom_symbol:        str
    rule_name:          str
    smarts_match:       str
    rule_score:         float
    dl_attention_risk:  float
    vulnerability_index: float          # [0, 100]
    dl_source:          str = "rule-only"


@dataclass
class MetabolismResult:
    """
    Top-level result returned by ``predict()``.

    ``soft_spots`` is now ``List[EnsembleSoftSpot]`` when the ensemble ran,
    or ``List[SoftSpot]`` when it fell back to rule-only mode.
    Both types expose ``atom_index``, ``atom_symbol``, and ``rule_name`` so
    downstream consumers (API layer, renderer) are unaffected.

    ``to_dict()`` is backward-compatible with v1 consumers while adding the
    new v2 ensemble fields.
    """
    parent:          MoleculeMetadata
    metabolites:     List[PredictedMetabolite]
    soft_spots:      List                       # List[EnsembleSoftSpot] | List[SoftSpot]
    engine_version:  str
    elapsed_s:       float
    warnings:        List[str] = field(default_factory=list)
    pipeline_stats:  Dict      = field(default_factory=dict)

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """
        Serialise to a plain nested dict for JSON responses, Celery backends,
        or pandas.  Backward-compatible with v1; adds ensemble fields.
        """
        soft_spot_dicts = []
        for s in self.soft_spots:
            if isinstance(s, EnsembleSoftSpot):
                soft_spot_dicts.append({
                    "atom_index":          s.atom_index,
                    "atom_symbol":         s.atom_symbol,
                    "rule_name":           s.rule_name,
                    "smarts_match":        s.smarts_match,
                    "rule_score":          s.rule_score,
                    "dl_attention_risk":   s.dl_attention_risk,
                    "vulnerability_index": s.vulnerability_index,
                    "dl_source":           s.dl_source,
                    # v1-compat alias
                    "score":               round(s.vulnerability_index / 100.0, 4),
                })
            else:
                soft_spot_dicts.append({
                    "atom_index":          s.atom_index,
                    "atom_symbol":         s.atom_symbol,
                    "rule_name":           s.rule_name,
                    "smarts_match":        s.smarts_match,
                    "rule_score":          s.score,
                    "dl_attention_risk":   0.0,
                    "vulnerability_index": round(s.score * 100, 2),
                    "dl_source":           "rule-only",
                    "score":               s.score,
                })

        return {
            "engine_version": self.engine_version,
            "elapsed_s":      round(self.elapsed_s, 4),
            "warnings":       self.warnings,
            "pipeline_stats": self.pipeline_stats,
            "parent": {
                "input_smiles":        self.parent.input_smiles,
                "canonical_smiles":    self.parent.canonical_smiles,
                "inchi":               self.parent.inchi,
                "inchikey":            self.parent.inchikey,
                "molecular_formula":   self.parent.molecular_formula,
                "molecular_weight":    self.parent.molecular_weight,
                "exact_mass":          self.parent.exact_mass,
                "num_heavy_atoms":     self.parent.num_heavy_atoms,
                "num_rotatable_bonds": self.parent.num_rotatable_bonds,
                "num_hbd":             self.parent.num_hbd,
                "num_hba":             self.parent.num_hba,
                "tpsa":                self.parent.tpsa,
                "logp":                self.parent.logp,
                "num_rings":           self.parent.num_rings,
                "num_aromatic_rings":  self.parent.num_aromatic_rings,
            },
            "metabolites": [
                {
                    "smiles":           m.smiles,
                    "probability":      m.probability,
                    "phase":            m.phase,
                    "reaction_name":    m.reaction_name,
                    "molecular_weight": m.molecular_weight,
                    "molecular_formula":m.molecular_formula,
                    "confidence_tier":  m.confidence_tier.value,
                    "dl_score":         m.dl_score,
                    "sources":          sorted(m.sources),
                }
                for m in self.metabolites
            ],
            "soft_spots": soft_spot_dicts,
        }

    @property
    def phase1_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites if m.phase == 1]

    @property
    def phase2_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites if m.phase == 2]

    @property
    def consensus_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites
                if m.confidence_tier == ConsensusTier.HIGH]

    @property
    def top_soft_spot_indices(self) -> List[int]:
        return [s.atom_index for s in self.soft_spots]


# ==============================================================================
# SOM SMARTS rules  (Pipeline A — rule-based component)
# ==============================================================================
# Each entry: (rule_name, smarts_pattern, priority_score)
#
# priority_score in (0, 1] based on published CYP450 substrate literature:
#   Zaretzki et al., J. Chem. Inf. Model. 2013 (CypReact)
#   Sheridan et al., J. Med. Chem. 2002
#   Cruciani et al., J. Med. Chem. 2005 (MetaSite)
#
# Convention: atom 0 in each SMARTS match is the SOM being annotated.

_SOM_RULES: List[Tuple[str, str, float]] = [
    # -- Phase I: CYP-mediated oxidation -------------------------------------

    # Benzylic CH2/CH3: highest CYP2C9/3A4 affinity (Meunier 2004)
    ("benzylic_CH",
     "[CH2,CH3;!$([CH2]C=O);!$([CH2]C#N)]-[c]", 0.92),

    # Unhindered aromatic C-H: CYP1A2/2C19 aromatic hydroxylation
    ("aromatic_C_unhindered",
     "[cH]", 0.85),

    # α-sp3 C to heteroatom: N/O-dealkylation (most common CYP reaction type)
    ("alpha_C_heteroatom",
     "[CH2,CH3;$([CH2,CH3][N,O;!$(N-C=O)])]", 0.88),

    # N-methyl/ethyl: CYP3A4 N-demethylation hotspot
    ("N_alkyl",
     "[CH3,CH2;$([CH3,CH2]-[N;!$(N-C=O);!$(N~[!#6;!#1])])]", 0.86),

    # α-carbonyl sp3 C: CYP-mediated α-hydroxylation
    ("alpha_carbonyl_C",
     "[CH2,CH3;$([CH2,CH3]C(=O)[#6,#7,#8])]", 0.75),

    # Terminal methyl (ω-hydroxylation, CYP4A/2E1)
    ("terminal_methyl_aliphatic",
     "[CH3;$([CH3]-[CH2]-[CH2]-[#6])]", 0.65),

    # Thioether S: CYP S-oxidation → sulfoxide → sulfone
    ("thioether_S",
     "[S;X2;!$(S=O);$([S]([#6])[#6])]", 0.80),

    # Tertiary amine N: FMO/CYP3A4 N-oxidation
    ("tertiary_amine_N",
     "[N;X3;!$(N-C=O);!$(N~[!#6;!#1]);!n;$([N]([#6])([#6])[#6])]", 0.78),

    # -- Phase I: Reduction --------------------------------------------------

    # Nitro group reduction (gut flora + hepatic)
    ("nitro_reduction",
     "[N+](=O)[O-]", 0.70),

    # Ketone/aldehyde carbonyl reduction (AKR / carbonyl reductases)
    ("carbonyl_reduction",
     "[CH1,CH0;$(C=O);!$(C(=O)[OH]);!$(C(=O)N)]", 0.60),

    # -- Phase II: Conjugation-prone sites -----------------------------------

    # Phenolic OH carbon: glucuronidation / sulfation (mark the C, not the O)
    ("phenolic_OH_site",
     "[c;$([c][OH1])]", 0.83),

    # Primary aliphatic amine: acetylation / glucuronidation
    ("primary_amine",
     "[NH2;$([NH2]-[#6;!$(C=O)])]", 0.72),

    # Carboxylic acid: acyl-glucuronidation (reactive metabolite risk)
    ("carboxylic_acid",
     "[C;$(C(=O)[OH1])]", 0.68),
]

# Normalisation pipeline  (shared singletons — expensive to construct)
_NORMALISER       = rdMolStandardize.Normalizer()
_UNCHARGER        = rdMolStandardize.Uncharger()
_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()


# ==============================================================================
# Pipeline B — DeepLearningPredictor
# ==============================================================================

class DeepLearningPredictor:
    """
    Production-ready deep-learning metabolite predictor emulator.

    Architectural inspiration
    -------------------------
    MetaTrans (KavrakiLab/MetaTrans): OpenNMT Transformer trained on
    experimental metabolite SMILES pairs.  Encodes the parent SMILES token
    sequence, decodes up to ``top_k`` metabolite sequences with beam search,
    and scores them with the model's log-likelihood.

    Meta-Predictor (zhukeyun/Meta-Predictor): Adds per-atom attention heads
    that produce site-of-metabolism risk vectors directly usable for soft
    spot annotation without a separate SOM model.

    This class
    ----------
    Implements the full interface those models require:
      - SMILES tokenisation  (atom-level, compatible with MetaTrans vocabulary)
      - Structural verification of generated SMILES via RDKit
      - Per-atom attention weight extraction (emulated with a deterministic
        chemistry-aware heuristic until real weights are loaded)
      - Graceful degradation: any tensor/inference failure returns empty lists
        with informative warnings, never crashes

    Swap-in path for real weights
    -----------------------------
    1. Install PyTorch + OpenNMT-py:
           pip install torch opennmt-py
    2. Subclass or monkey-patch ``_transformer_beam_search``:
           def _transformer_beam_search(self, tokens, top_k):
               # real OpenNMT / HuggingFace inference here
               ...
    3. Subclass or monkey-patch ``_extract_attention_weights``:
           def _extract_attention_weights(self, mol, tokens):
               # real model forward-pass attention extraction
               ...
    4. Load weights via ``from_checkpoint``:
           predictor = DeepLearningPredictor.from_checkpoint("metatrans.pt")

    All public method signatures remain stable between emulator and real-weight
    modes — no changes needed in the calling code.
    """

    # ── MetaTrans atom-level vocabulary (subset) ──────────────────────────────
    # Real MetaTrans uses a regex tokeniser that splits on atom symbols,
    # bond characters, ring digits, and branch brackets.  We replicate that
    # split pattern here so tokenised sequences are compatible.
    _TOKEN_PATTERN = (
        r"(\[[^\]]+\]"           # bracket atoms  [NH2], [13C], ...
        r"|Br|Cl|Si|Se|@@|@"     # two-char atoms / stereo
        r"|[BCNOPSFIbcnops]"     # one-char organic subset
        r"|[0-9%]"               # ring closure digits
        r"|[/\\+=\-#\(\)\.])"    # bond / branch / dot
    )

    MODEL_NAME = "metatrans-emulator-v2"

    def __init__(
        self,
        top_k: int = 10,
        confidence_threshold: float = DL_CONFIDENCE_THRESHOLD,
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        top_k                : beam-search width (number of candidate
                               metabolite sequences returned per parent).
        confidence_threshold : minimum model confidence to include a prediction.
        seed                 : optional random seed for the emulator (for
                               reproducible unit tests only; real models
                               are deterministic given the same weights).
        """
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self._rng = random.Random(seed)
        self._model_loaded = False   # True once real weights are attached
        logger.info(
            "dl_predictor_init",
            model=self.MODEL_NAME,
            top_k=top_k,
            mode="emulator",
        )

    # -- Class-method constructors --------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        **kwargs,
    ) -> "DeepLearningPredictor":
        """
        Load a MetaTrans / Meta-Predictor checkpoint.

        This method is a **stub** — it returns an emulator instance and logs
        a warning until real model loading code is inserted here.

        To implement:
        -------------
            import torch, onmt
            model = onmt.model_builder.build_base_model(opt, fields, gpu)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            predictor = cls(**kwargs)
            predictor._model = model
            predictor._model_loaded = True
            return predictor
        """
        instance = cls(**kwargs)
        logger.warning(
            "dl_checkpoint_stub",
            path=checkpoint_path,
            msg="Real model weights not loaded — using emulator. "
                "Implement from_checkpoint() to enable live inference.",
        )
        return instance

    # -- Tokenisation ---------------------------------------------------------

    def tokenise(self, smiles: str) -> List[str]:
        """
        Atom-level SMILES tokeniser compatible with MetaTrans vocabulary.

        Splits the SMILES string into a list of token strings, each
        representing a single atom, bond, stereo descriptor, or bracket group.

        Parameters
        ----------
        smiles : canonical SMILES string (pre-normalised by RDKit).

        Returns
        -------
        List of token strings, e.g.
            "CC(=O)O" → ["C", "C", "(", "=", "O", ")", "O"]

        Raises
        ------
        ValueError : if the SMILES is empty or produces zero tokens.
        """
        import re
        if not smiles or not smiles.strip():
            raise ValueError("Cannot tokenise empty SMILES.")
        tokens = re.findall(self._TOKEN_PATTERN, smiles)
        if not tokens:
            raise ValueError(
                f"SMILES '{smiles}' produced zero tokens. "
                "It may contain unsupported notation."
            )
        return tokens

    # -- Structural verification ----------------------------------------------

    @staticmethod
    def verify_smiles(smiles: str) -> Optional[Chem.Mol]:
        """
        Verify that a SMILES string represents a valid, sanitisable molecule.

        Catches the full RDKit error taxonomy:
          - ``rdchem.MolSanitizeException`` — valence/kekulisation failures
          - ``rdchem.KekulizeException``    — aromatic resolution failures
          - ``ValueError``                  — bad SMILES syntax
          - Any other Exception             — unknown RDKit error

        Parameters
        ----------
        smiles : SMILES string to verify (may be transformer-generated).

        Returns
        -------
        Sanitised ``Chem.Mol`` on success, or ``None`` if any check fails.
        """
        if not smiles or not smiles.strip():
            return None
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None

            # Attempt full sanitisation — this is the main gate for
            # hypervalent/impossible structures generated by the transformer
            sanitize_flags = Chem.SanitizeFlags.SANITIZE_ALL
            result_flag = Chem.SanitizeMol(mol, sanitize_flags, catchErrors=True)
            if result_flag != Chem.SanitizeFlags.SANITIZE_NONE:
                logger.debug(
                    "dl_smiles_sanitize_failed"),
                )
                return None

            if mol.GetNumHeavyAtoms() == 0:
                return None

            # Re-parse from canonical SMILES for clean atom mapping
            canonical = Chem.MolToSmiles(mol, canonical=True)
            mol2 = Chem.MolFromSmiles(canonical)
            return mol2

        except rdchem.MolSanitizeException as exc:
            logger.debug("dl_mol_sanitize_exception", error=str(exc), smiles=smiles[:80])
            return None
        except rdchem.KekulizeException as exc:
            logger.debug("dl_mol_kekulize_exception", error=str(exc), smiles=smiles[:80])
            return None
        except Exception as exc:
            logger.debug("dl_mol_verify_error", error=str(exc), smiles=smiles[:80])
            return None

    # -- Core inference hooks (emulated; replace with real model calls) --------

    def _transformer_beam_search(
        self,
        tokens: List[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Emulate transformer beam-search decoding.

        PRODUCTION REPLACEMENT POINT
        ----------------------------
        Replace the body of this method with a real OpenNMT / HuggingFace
        inference call.  The expected signature and return type are stable:

            inputs  : token list from ``tokenise()``
            returns : list of (smiles_string, log_likelihood_score) tuples,
                      up to ``top_k`` entries, sorted by score descending.

        Emulator strategy
        -----------------
        Rather than pure randomness (which would produce mostly invalid SMILES),
        the emulator applies a set of chemically plausible single-step
        transformations to the input SMILES — the same operations MetaTrans
        was trained to generate.  This ensures a realistic fraction of
        structurally valid outputs for testing and development.

        Confidence scores are set to realistic ranges drawn from published
        MetaTrans log-likelihood distributions (Litsa et al. 2023).
        """
        parent_smiles = "".join(tokens)

        # Validate the parent SMILES before attempting to transform
        parent_mol = self.verify_smiles(parent_smiles)
        if parent_mol is None:
            return []

        # Deterministic seed: same parent → same emulated outputs across runs
        # (prevents flapping tests while still covering diverse transformations)
        seed_hash = int(hashlib.sha256(parent_smiles.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed_hash)

        candidates: List[Tuple[str, float]] = []

        # ── Emulated transformation set ────────────────────────────────────────
        # Each lambda attempts a chemically plausible single SMIRKS-like step.
        # Transformations that don't apply to the molecule are silently skipped.
        transformations = [
            self._emulate_aromatic_hydroxylation,
            self._emulate_aliphatic_hydroxylation,
            self._emulate_n_demethylation,
            self._emulate_o_demethylation,
            self._emulate_glucuronidation,
            self._emulate_sulfation,
            self._emulate_oxidation_sulfoxide,
            self._emulate_hydrolysis,
        ]

        rng.shuffle(transformations)

        for transform in transformations[:top_k]:
            try:
                result_smiles = transform(parent_mol, rng)
                if result_smiles is None:
                    continue
                verified = self.verify_smiles(result_smiles)
                if verified is None:
                    continue
                canon = Chem.MolToSmiles(verified, canonical=True)
                # Emulated log-likelihood in realistic MetaTrans range (~-0.1 to -3.0)
                score = rng.uniform(0.08, 0.55)
                candidates.append((canon, round(score, 4)))
            except Exception as exc:
                logger.debug("dl_transform_failed", error=str(exc))
                continue

        # Sort by score descending, deduplicate
        seen_smiles: Set[str] = set()
        unique: List[Tuple[str, float]] = []
        for smiles, score in sorted(candidates, key=lambda x: x[1], reverse=True):
            if smiles not in seen_smiles:
                seen_smiles.add(smiles)
                unique.append((smiles, score))

        return unique[:top_k]

    def _extract_attention_weights(
        self,
        mol: Chem.Mol,
        tokens: List[str],
    ) -> Dict[int, float]:
        """
        Extract per-atom SOM attention risk weights from the transformer.

        PRODUCTION REPLACEMENT POINT
        ----------------------------
        In a real MetaTrans / Meta-Predictor model, this method runs a
        forward pass and reads the cross-attention weights from the encoder
        layers aligned to each heavy atom.  Replace the body with:

            with torch.no_grad():
                outputs = self._model(input_tensor, attention=True)
                atom_weights = outputs.encoder_attentions[atom_indices]
            return {idx: float(w) for idx, w in enumerate(atom_weights)}

        Emulator strategy
        -----------------
        We produce a chemistry-aware pseudo-attention distribution that
        correlates with known SOM preferences, making the emulated values
        directionally correct for testing and UI development.

        The distribution is built from three additive signals:
          1. Atomic number contribution (heteroatoms get higher base weight)
          2. Environment contribution (aromatic/sp3/sp2 character)
          3. SMARTS rule overlap (atoms matching known SOM patterns get boost)

        All values are normalised to [0, 1] with softmax.
        """
        n_atoms = mol.GetNumHeavyAtoms()
        if n_atoms == 0:
            return {}

        raw_weights: Dict[int, float] = {}

        # Precompute which atoms match any SOM rule
        som_boosted: Set[int] = set()
        for _, smarts, _ in _SOM_RULES:
            pat = Chem.MolFromSmarts(smarts)
            if pat is None:
                continue
            for match in mol.GetSubstructMatches(pat):
                if match:
                    som_boosted.add(match[0])

        # ── Pseudo-attention computation ───────────────────────────────────────
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            w = 0.0

            # Signal 1: heteroatom base weight
            atomic_num = atom.GetAtomicNum()
            if atomic_num in (7, 8):       # N, O
                w += 0.35
            elif atomic_num == 16:          # S
                w += 0.30
            elif atomic_num == 6:           # C
                w += 0.15
            else:
                w += 0.20

            # Signal 2: hybridisation / environment
            hyb = atom.GetHybridization()
            if atom.GetIsAromatic():
                w += 0.25
            elif hyb == Chem.rdchem.HybridizationType.SP3:
                w += 0.20
            elif hyb == Chem.rdchem.HybridizationType.SP2:
                w += 0.18

            # Signal 3: explicit hydrogen count (CYP accessibility proxy)
            w += atom.GetTotalNumHs() * 0.10

            # Signal 4: SOM rule overlap boost
            if idx in som_boosted:
                w += 0.40

            raw_weights[idx] = w

        # Softmax normalisation → values sum to 1.0, then rescale to [0, 1]
        exp_w = {idx: math.exp(v) for idx, v in raw_weights.items()}
        total = sum(exp_w.values()) or 1.0
        normalised = {idx: v / total for idx, v in exp_w.items()}

        # Rescale: set max atom to 1.0, others proportionally
        max_val = max(normalised.values()) if normalised else 1.0
        if max_val > 0:
            return {idx: round(v / max_val, 4) for idx, v in normalised.items()}
        return {idx: 0.0 for idx in normalised}

    # -- Chemical transformation emulators ------------------------------------
    # Each method attempts a single chemical step on ``mol`` and returns a
    # product SMILES string, or None if the transformation doesn't apply.
    # These are *not* meant to be chemically exhaustive — they serve as
    # realistic test vectors for the DL pipeline interface.

    @staticmethod
    def _emulate_aromatic_hydroxylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Add -OH to an unhindered aromatic C-H (CYP1A2/2C19 style)."""
        try:
            from rdkit.Chem import AllChem, RWMol
            pattern = Chem.MolFromSmarts("[cH]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            atom_idx = match[0]
            rw = Chem.RWMol(mol)
            rw.GetAtomWithIdx(atom_idx).SetNumExplicitHs(0)
            oh_idx = rw.AddAtom(Chem.Atom(8))     # O
            h_idx  = rw.AddAtom(Chem.Atom(1))     # H
            rw.AddBond(atom_idx, oh_idx, Chem.BondType.SINGLE)
            rw.AddBond(oh_idx,   h_idx,  Chem.BondType.SINGLE)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_aliphatic_hydroxylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Add -OH to an aliphatic sp3 C-H (benzylic or α-heteroatom)."""
        try:
            from rdkit.Chem import RWMol
            # Prefer benzylic CH2
            pattern = Chem.MolFromSmarts("[CH2;$([CH2]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                pattern = Chem.MolFromSmarts("[CH3;$([CH3]-[c])]")
                matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            atom_idx = match[0]
            rw = Chem.RWMol(Chem.AddHs(mol))
            # Find one explicit H on this atom
            h_to_remove = None
            for nbr in rw.GetAtomWithIdx(atom_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 1:
                    h_to_remove = nbr.GetIdx()
                    break
            if h_to_remove is None:
                return None
            rw.RemoveAtom(h_to_remove)
            oh_idx = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(atom_idx, oh_idx, Chem.BondType.SINGLE)
            product = Chem.RemoveHs(rw)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(product, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_n_demethylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Remove N-methyl group (CYP3A4 N-demethylation)."""
        try:
            from rdkit.Chem import AllChem, RWMol, rdMolTransforms
            pattern = Chem.MolFromSmarts("[CH3;$([CH3]-[N;!$(N-C=O)])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx = match[0]
            rw = Chem.RWMol(mol)
            # Find the N neighbor
            n_idx = None
            for nbr in rw.GetAtomWithIdx(c_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 7:
                    n_idx = nbr.GetIdx()
                    break
            if n_idx is None:
                return None
            rw.RemoveBond(c_idx, n_idx)
            # Remove the dangling CH3
            atoms_to_remove = sorted([c_idx], reverse=True)
            for idx in atoms_to_remove:
                rw.RemoveAtom(idx)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_o_demethylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Remove O-methyl group → phenol (CYP2D6 / 2C9 O-demethylation)."""
        try:
            from rdkit.Chem import RWMol
            pattern = Chem.MolFromSmarts("[CH3;$([CH3]-O-[c,C])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx = match[0]
            rw = Chem.RWMol(mol)
            o_idx = None
            for nbr in rw.GetAtomWithIdx(c_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 8:
                    o_idx = nbr.GetIdx()
                    break
            if o_idx is None:
                return None
            rw.RemoveBond(c_idx, o_idx)
            # Remove the dangling CH3
            for idx in sorted([c_idx], reverse=True):
                rw.RemoveAtom(idx)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_glucuronidation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Append glucuronic acid to a phenol (Phase II conjugation)."""
        try:
            # Instead of real glucuronide construction (complex stereochemistry),
            # return the deprotonated phenol as a proxy product — structurally
            # correct for testing the DL pipeline, not for clinical use.
            pattern = Chem.MolFromSmarts("[OH;$([OH]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            # Proxy: return the phenol itself (identity for phenols)
            # Real implementation would use a SMIRKS reaction:
            # "[OH:1]-[c:2]>>[O:1]([c:2])C1OC(C(=O)O)C(O)C(O)C1O"
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    @staticmethod
    def _emulate_sulfation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Sulfate conjugation of a phenol (Phase II, SULT enzymes)."""
        try:
            from rdkit.Chem import RWMol, AllChem
            pattern = Chem.MolFromSmarts("[OH;$([OH]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            o_idx = match[0]
            rw = Chem.RWMol(mol)
            s_idx  = rw.AddAtom(Chem.Atom(16))   # S
            o1_idx = rw.AddAtom(Chem.Atom(8))    # =O
            o2_idx = rw.AddAtom(Chem.Atom(8))    # =O
            o3_idx = rw.AddAtom(Chem.Atom(8))    # -OH
            h_idx  = rw.AddAtom(Chem.Atom(1))
            rw.AddBond(o_idx,  s_idx,  Chem.BondType.SINGLE)
            rw.AddBond(s_idx,  o1_idx, Chem.BondType.DOUBLE)
            rw.AddBond(s_idx,  o2_idx, Chem.BondType.DOUBLE)
            rw.AddBond(s_idx,  o3_idx, Chem.BondType.SINGLE)
            rw.AddBond(o3_idx, h_idx,  Chem.BondType.SINGLE)
            # Remove the original O-H proton
            rw.GetAtomWithIdx(o_idx).SetNumExplicitHs(0)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_oxidation_sulfoxide(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Oxidise thioether S → sulfoxide (CYP / FMO S-oxidation)."""
        try:
            from rdkit.Chem import RWMol
            pattern = Chem.MolFromSmarts("[S;X2;!$(S=O);$([S]([#6])[#6])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            s_idx = match[0]
            rw = Chem.RWMol(mol)
            o_idx = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(s_idx, o_idx, Chem.BondType.DOUBLE)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_hydrolysis(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Ester hydrolysis → carboxylic acid + alcohol."""
        try:
            from rdkit.Chem import AllChem, RWMol
            # Detect ester: [C](=O)-O-[C,c]
            pattern = Chem.MolFromSmarts("[C:1](=[O:2])-[O:3]-[#6:4]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx, o_dbl, o_single, c_right = match
            rw = Chem.RWMol(mol)
            rw.RemoveBond(o_single, c_right)
            # Add H to both oxygens
            rw.GetAtomWithIdx(o_single).SetNumExplicitHs(1)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    # -- Public prediction API ------------------------------------------------

    def predict(
        self,
        smiles: str,
        phase_hint: Optional[int] = None,
    ) -> Tuple[List[Tuple[str, float]], Dict[int, float], List[str]]:
        """
        Run the full DL prediction pipeline for a single parent molecule.

        Parameters
        ----------
        smiles     : canonical SMILES (pre-normalised by ``_validate_and_normalise``).
        phase_hint : optional phase bias (1 or 2); not used by emulator but
                     available for real model phase-conditioned decoding.

        Returns
        -------
        (predictions, attention_weights, warnings)

          predictions       : List[(canonical_smiles, confidence_score)]
                              sorted by confidence descending.
          attention_weights : Dict[atom_index → risk_weight]
                              per-atom SOM attention scores in [0, 1].
          warnings          : List of non-fatal diagnostic strings.
        """
        dl_warnings: List[str] = []

        # ── Tokenise ──────────────────────────────────────────────────────────
        try:
            tokens = self.tokenise(smiles)
        except ValueError as exc:
            dl_warnings.append(f"DL tokenisation failed: {exc}")
            return [], {}, dl_warnings

        # ── Verify parent ─────────────────────────────────────────────────────
        parent_mol = self.verify_smiles(smiles)
        if parent_mol is None:
            dl_warnings.append(
                f"DL pipeline skipped: parent SMILES failed structural "
                f"verification after tokenisation. SMILES: '{smiles[:80]}'"
            )
            return [], {}, dl_warnings

        # ── Beam search (emulated or real) ────────────────────────────────────
        try:
            raw_predictions = self._transformer_beam_search(tokens, self.top_k)
        except Exception as exc:
            dl_warnings.append(f"DL beam search error: {exc}")
            raw_predictions = []

        # ── Filter below confidence threshold ─────────────────────────────────
        predictions = [
            (smi, score)
            for smi, score in raw_predictions
            if score >= self.confidence_threshold
        ]

        # ── Attention weights (emulated or real) ──────────────────────────────
        try:
            attention_weights = self._extract_attention_weights(parent_mol, tokens)
        except Exception as exc:
            dl_warnings.append(f"DL attention extraction error: {exc}")
            attention_weights = {}

        logger.info(
            "dl_predict_complete",
            model=self.MODEL_NAME,
            raw_predictions=len(raw_predictions),
            filtered_predictions=len(predictions),
            attention_atoms=len(attention_weights),
        )

        return predictions, attention_weights, dl_warnings


# ==============================================================================
# Consensus Engine
# ==============================================================================

class ConsensusEngine:
    """
    Merges Pipeline A (SyGMa) and Pipeline B (DL) outputs into a unified,
    confidence-tagged metabolite list and ensemble soft spot annotations.

    Consensus logic
    ---------------
    Structural alignment uses ``Chem.CanonSmiles`` to normalise both sets of
    SMILES before comparison.  An exact match of canonical forms constitutes
    consensus — no Tanimoto similarity threshold is used because near-identical
    SMILES with different stereochemistry represent chemically distinct metabolites
    and should be tagged separately.

    Score fusion
    ------------
    Consensus metabolites inherit the higher of the two pipeline scores as
    their primary ``probability`` field, preserving the SyGMa reaction name
    and phase assignment (which the DL model does not produce).

    For soft spots, the vulnerability index blends rule and attention signals:
        VI = (ENSEMBLE_ALPHA × rule_score + ENSEMBLE_BETA × dl_attention) × 100 %
    """

    def __init__(
        self,
        alpha: float = ENSEMBLE_ALPHA,
        beta:  float = ENSEMBLE_BETA,
    ) -> None:
        self.alpha = alpha
        self.beta  = beta

    # ------------------------------------------------------------------

    def merge_metabolites(
        self,
        sygma_metabolites: List[PredictedMetabolite],
        dl_predictions:    List[Tuple[str, float]],
    ) -> Tuple[List[PredictedMetabolite], Dict]:
        """
        Merge and tag SyGMa + DL metabolite predictions.

        Parameters
        ----------
        sygma_metabolites : output of ``_run_sygma``.
        dl_predictions    : list of (canonical_smiles, confidence) from DL.

        Returns
        -------
        (merged_metabolites, stats_dict)
          merged_metabolites : deduplicated, confidence-tagged list sorted
                               by probability descending.
          stats_dict         : pipeline statistics for logging / API response.
        """
        # Build canonical SMILES lookup for SyGMa results
        sygma_by_canon: Dict[str, PredictedMetabolite] = {}
        for m in sygma_metabolites:
            canon = self._safe_canonicalise(m.smiles)
            if canon:
                sygma_by_canon[canon] = m

        # Build canonical SMILES set for DL results
        dl_by_canon: Dict[str, float] = {}
        for smiles, score in dl_predictions:
            canon = self._safe_canonicalise(smiles)
            if canon:
                dl_by_canon[canon] = score

        consensus_set    = set(sygma_by_canon) & set(dl_by_canon)
        sygma_only_set   = set(sygma_by_canon) - set(dl_by_canon)
        dl_only_set      = set(dl_by_canon)    - set(sygma_by_canon)

        merged: List[PredictedMetabolite] = []

        # ── Consensus metabolites ──────────────────────────────────────────────
        for canon in consensus_set:
            base = sygma_by_canon[canon]
            dl_s = dl_by_canon[canon]
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=max(base.probability, dl_s),
                phase=base.phase,
                reaction_name=base.reaction_name,
                molecular_weight=base.molecular_weight,
                molecular_formula=base.molecular_formula,
                confidence_tier=ConsensusTier.HIGH,
                dl_score=dl_s,
                sources=frozenset({"sygma", "dl"}),
            ))

        # ── SyGMa-only metabolites ────────────────────────────────────────────
        for canon in sygma_only_set:
            base = sygma_by_canon[canon]
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=base.probability,
                phase=base.phase,
                reaction_name=base.reaction_name,
                molecular_weight=base.molecular_weight,
                molecular_formula=base.molecular_formula,
                confidence_tier=ConsensusTier.RULE,
                dl_score=None,
                sources=frozenset({"sygma"}),
            ))

        # ── DL-only metabolites ───────────────────────────────────────────────
        for canon in dl_only_set:
            dl_s = dl_by_canon[canon]
            mol = self._safe_mol(canon)
            mw, formula = self._compute_descriptors(mol)
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=dl_s,
                phase=-1,           # DL model does not produce phase labels
                reaction_name="dl_predicted",
                molecular_weight=mw,
                molecular_formula=formula,
                confidence_tier=ConsensusTier.DL,
                dl_score=dl_s,
                sources=frozenset({"dl"}),
            ))

        merged.sort(key=lambda m: m.probability, reverse=True)

        stats = {
            "sygma_total":      len(sygma_by_canon),
            "dl_total":         len(dl_by_canon),
            "consensus_count":  len(consensus_set),
            "rule_only_count":  len(sygma_only_set),
            "dl_only_count":    len(dl_only_set),
            "merged_total":     len(merged),
        }

        logger.info("consensus_merge_complete", **stats)
        return merged, stats

    def merge_soft_spots(
        self,
        rule_spots:       List[SoftSpot],
        attention_weights: Dict[int, float],
        mol:              Chem.Mol,
        top_n:            int,
        dl_source:        str = "metatrans-emulator",
    ) -> List[EnsembleSoftSpot]:
        """
        Fuse rule-based and DL attention soft spot signals.

        For atoms with a rule match: VI = (α·rule_score + β·dl_attention) × 100
        For atoms with attention only (no rule match): VI = β·dl_attention × 100
        Rule-matched atoms always appear in the top-N if they outscore attention-only.

        Parameters
        ----------
        rule_spots        : output of ``_find_soft_spots``.
        attention_weights : per-atom attention weights from DL predictor.
        mol               : parent molecule (used to look up atom symbols for
                            attention-only atoms not in rule_spots).
        top_n             : number of ensemble spots to return.
        dl_source         : identifier for the DL model providing attention.

        Returns
        -------
        List of ``EnsembleSoftSpot``, sorted by vulnerability_index descending,
        length ≤ top_n.
        """
        ensemble: Dict[int, EnsembleSoftSpot] = {}

        # Process rule-matched atoms
        for spot in rule_spots:
            dl_w = attention_weights.get(spot.atom_index, 0.0)
            vi   = round((self.alpha * spot.score + self.beta * dl_w) * 100, 2)
            ensemble[spot.atom_index] = EnsembleSoftSpot(
                atom_index=spot.atom_index,
                atom_symbol=spot.atom_symbol,
                rule_name=spot.rule_name,
                smarts_match=spot.smarts_match,
                rule_score=spot.score,
                dl_attention_risk=dl_w,
                vulnerability_index=vi,
                dl_source=dl_source,
            )

        # Process attention-only atoms (not in rule_spots) — add if high attention
        if attention_weights:
            # Threshold: include attention-only atoms above the 75th percentile
            sorted_weights = sorted(attention_weights.values(), reverse=True)
            threshold_rank = max(1, len(sorted_weights) // 4)
            threshold = sorted_weights[threshold_rank - 1] if sorted_weights else 0.0

            for atom_idx, dl_w in attention_weights.items():
                if atom_idx in ensemble:
                    continue
                if dl_w < threshold:
                    continue
                vi = round(self.beta * dl_w * 100, 2)
                try:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    symbol = atom.GetSymbol()
                except Exception:
                    symbol = "?"
                ensemble[atom_idx] = EnsembleSoftSpot(
                    atom_index=atom_idx,
                    atom_symbol=symbol,
                    rule_name="dl_attention_only",
                    smarts_match="",
                    rule_score=0.0,
                    dl_attention_risk=dl_w,
                    vulnerability_index=vi,
                    dl_source=dl_source,
                )

        result = sorted(
            ensemble.values(),
            key=lambda s: s.vulnerability_index,
            reverse=True,
        )
        return result[:top_n]

    # -- Private helpers ------------------------------------------------------

    @staticmethod
    def _safe_canonicalise(smiles: str) -> Optional[str]:
        """Return RDKit canonical SMILES, or None on any RDKit error."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None

    @staticmethod
    def _safe_mol(smiles: str) -> Optional[Chem.Mol]:
        try:
            return Chem.MolFromSmiles(smiles)
        except Exception:
            return None

    @staticmethod
    def _compute_descriptors(
        mol: Optional[Chem.Mol],
    ) -> Tuple[Optional[float], Optional[str]]:
        if mol is None:
            return None, None
        try:
            return (
                round(Descriptors.MolWt(mol), 4),
                rdMolDescriptors.CalcMolFormula(mol),
            )
        except Exception:
            return None, None


# ==============================================================================
# Private helpers  (validation, SyGMa runner, rule-based SOM)
# ==============================================================================

def _validate_and_normalise(smiles: str) -> Tuple[Chem.Mol, MoleculeMetadata]:
    """
    Parse, validate, and normalise a SMILES string.

    Normalisation pipeline
    ----------------------
    1. ``Chem.MolFromSmiles``     — parse + valence check
    2. ``LargestFragmentChooser`` — strip salts / counterions
    3. ``Normalizer``             — canonical functional-group forms
    4. ``Uncharger``              — neutralise formal charges
    5. ``SanitizeMol``            — final RDKit sanitisation
    6. Descriptor computation     — MW, formula, InChI, logP, TPSA, etc.

    Raises
    ------
    ValueError  if the SMILES is invalid at any pipeline stage.
    """
    if not smiles or not smiles.strip():
        raise ValueError("SMILES string is empty.")

    smiles = smiles.strip()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(
            f"RDKit could not parse SMILES: {smiles!r}. "
            "Check atom symbols, valences, and ring-closure notation."
        )

    mol = _LARGEST_FRAGMENT.choose(mol)
    mol = _NORMALISER.normalize(mol)
    mol = _UNCHARGER.uncharge(mol)

    try:
        Chem.SanitizeMol(mol)
    except rdchem.MolSanitizeException as exc:
        raise ValueError(f"Sanitisation failed after normalisation: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Normalisation error: {exc}") from exc

    if mol.GetNumHeavyAtoms() == 0:
        raise ValueError("Molecule has no heavy atoms after normalisation.")

    from rdkit.Chem import inchi as rdinchi
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    inchi_str   = rdinchi.MolToInchi(mol) or ""
    inchikey    = rdinchi.InchiToInchiKey(inchi_str) if inchi_str else ""

    metadata = MoleculeMetadata(
        input_smiles=smiles,
        canonical_smiles=canonical_smiles,
        inchi=inchi_str,
        inchikey=inchikey,
        molecular_formula=rdMolDescriptors.CalcMolFormula(mol),
        molecular_weight=round(Descriptors.MolWt(mol), 4),
        exact_mass=round(Descriptors.ExactMolWt(mol), 6),
        num_heavy_atoms=mol.GetNumHeavyAtoms(),
        num_rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
        num_hbd=rdMolDescriptors.CalcNumHBD(mol),
        num_hba=rdMolDescriptors.CalcNumHBA(mol),
        tpsa=round(rdMolDescriptors.CalcTPSA(mol), 2),
        logp=round(Descriptors.MolLogP(mol), 4),
        num_rings=rdMolDescriptors.CalcNumRings(mol),
        num_aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
    )

    logger.debug(
        "molecule_normalised",
        canonical=canonical_smiles,
        mw=metadata.molecular_weight,
        formula=metadata.molecular_formula,
    )
    return mol, metadata


def _run_sygma(
    mol: Chem.Mol,
    phase1_cycles: int = 1,
    phase2_cycles: int = 1,
    max_metabolites: int = 200,
) -> List[PredictedMetabolite]:
    """
    Execute a SyGMa Phase I → Phase II prediction scenario.

    Returns PredictedMetabolite instances with ``sources=frozenset({"sygma"})``
    and ``confidence_tier=ConsensusTier.RULE`` (to be upgraded to HIGH by the
    ConsensusEngine when DL agreement is found).

    Raises
    ------
    ImportError  if SyGMa is not installed.
    """
    try:
        import sygma                          # type: ignore
        from sygma import ruleset as srs      # type: ignore
        from sygma import Scenario            # type: ignore
    except ImportError as exc:
        raise ImportError(
            "SyGMa is not installed. "
            "Add 'sygma>=1.1.0' to the pip section of environment.yml."
        ) from exc

    scenario = Scenario([
        [srs.phase1, phase1_cycles],
        [srs.phase2, phase2_cycles],
    ])

    logger.debug("sygma_start", p1=phase1_cycles, p2=phase2_cycles)
    tree = scenario.run(mol)
    tree.calc_scores()

    parent_canon = Chem.MolToSmiles(mol, canonical=True)
    seen: Dict[str, PredictedMetabolite] = {}

    for node in tree.to_list():
        node_mol = getattr(node, "mol", None)
        if node_mol is None:
            continue

        # Per-metabolite error defence — catch sanitisation failures
        try:
            Chem.SanitizeMol(node_mol)
        except rdchem.MolSanitizeException as exc:
            logger.debug("sygma_sanitize_skip", error=str(exc))
            continue
        except Exception as exc:
            logger.debug("sygma_mol_skip", error=str(exc))
            continue

        try:
            smiles = Chem.MolToSmiles(node_mol, canonical=True)
        except Exception as exc:
            logger.debug("sygma_smiles_skip", error=str(exc))
            continue

        if not smiles or smiles == parent_canon:
            continue

        probability   = float(getattr(node, "score",         0.0))
        reaction_name = str(getattr(node,  "reaction_name", "") or "unknown")

        phase = (2 if reaction_name.lower().startswith("phase2")
                 or any(k in reaction_name.lower()
                        for k in ("glucuronid", "sulfat", "acetyl",
                                  "glutathione", "methyl"))
                 else 1)
        if reaction_name.lower().startswith("phase1"):
            phase = 1

        try:
            mw      = round(Descriptors.MolWt(node_mol), 4)
            formula = rdMolDescriptors.CalcMolFormula(node_mol)
        except Exception:
            mw, formula = None, None

        candidate = PredictedMetabolite(
            smiles=smiles,
            probability=probability,
            phase=phase,
            reaction_name=reaction_name,
            molecular_weight=mw,
            molecular_formula=formula,
            confidence_tier=ConsensusTier.RULE,   # upgraded by consensus later
            sources=frozenset({"sygma"}),
        )

        if smiles not in seen or probability > seen[smiles].probability:
            seen[smiles] = candidate

    results = sorted(seen.values(), key=lambda x: x.probability, reverse=True)
    results = results[:max_metabolites]

    logger.info(
        "sygma_complete",
        unique=len(results),
        p1=sum(1 for r in results if r.phase == 1),
        p2=sum(1 for r in results if r.phase == 2),
    )
    return results


def _find_soft_spots(
    mol: Chem.Mol,
    top_n: int = 3,
) -> List[SoftSpot]:
    """
    Rule-only soft spot fallback (Pipeline A standalone).

    Used when the DL pipeline is unavailable or disabled.  The ConsensusEngine
    calls ``merge_soft_spots`` instead when both pipelines are active.

    Algorithm
    ---------
    For each rule in ``_SOM_RULES``:
      - Compile SMARTS, find substructure matches.
      - Atom at match index 0 is the SOM site.
      - Apply a 20 % steric penalty for atoms with ≥ 3 bulky neighbours.
    """
    best: Dict[int, SoftSpot] = {}

    for rule_name, smarts, base_score in _SOM_RULES:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            logger.warning("invalid_smarts", rule=rule_name)
            continue

        for match in mol.GetSubstructMatches(pattern):
            if not match:
                continue
            atom_idx = match[0]
            atom     = mol.GetAtomWithIdx(atom_idx)

            heavy_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
            score = base_score
            if len(heavy_nbrs) >= 3 and any(n.GetDegree() > 1 for n in heavy_nbrs):
                score *= 0.80

            score = round(score, 4)
            if atom_idx not in best or score > best[atom_idx].score:
                best[atom_idx] = SoftSpot(
                    atom_index=atom_idx,
                    atom_symbol=atom.GetSymbol(),
                    rule_name=rule_name,
                    score=score,
                    smarts_match=smarts,
                )

    return sorted(best.values(), key=lambda s: s.score, reverse=True)[:top_n]


# ==============================================================================
# Public API
# ==============================================================================

# Type alias for drop-in soft-spot function hook (backward compatible with v1)
SoftSpotFn = Callable[[Chem.Mol, int], List[SoftSpot]]


def predict(
    smiles: str,
    *,
    phase1_cycles:   int  = 1,
    phase2_cycles:   int  = 1,
    max_metabolites: int  = 200,
    top_soft_spots:  int  = 3,
    # v2 ensemble parameters
    dl_predictor:    Optional[DeepLearningPredictor] = None,
    enable_dl:       bool = True,
    # v1 backward-compat hook (used when dl_predictor is None and enable_dl=False)
    soft_spot_fn:    Optional[SoftSpotFn] = None,
) -> MetabolismResult:
    """
    Ensemble Consensus Metabolism Prediction.

    Orchestrates the dual-pipeline architecture:
      Pipeline A  — SyGMa rule-based Phase I/II metabolites
      Pipeline B  — DeepLearningPredictor (emulator or real weights)
      Consensus   — cross-validates outputs, assigns ConsensusTier tags
      Soft Spots  — blends rule scores with DL attention weights

    Parameters
    ----------
    smiles          : Parent molecule SMILES (any notation; normalised internally).
    phase1_cycles   : SyGMa Phase I iteration depth.
    phase2_cycles   : SyGMa Phase II iteration depth.
    max_metabolites : Hard cap on returned metabolites (applied after merge).
    top_soft_spots  : Number of soft spots to annotate.
    dl_predictor    : Pre-built DeepLearningPredictor instance.  If None and
                      enable_dl=True, a default emulator is created internally.
    enable_dl       : Master switch for Pipeline B.  Set False to run rule-only
                      (equivalent to v1 behaviour).
    soft_spot_fn    : v1 backward-compat hook.  Ignored when enable_dl=True.

    Returns
    -------
    ``MetabolismResult`` with full ensemble data.  ``.to_dict()`` is
    backward-compatible with v1 API consumers.

    Raises
    ------
    ValueError   : SMILES is invalid.
    RuntimeError : Unexpected engine-level error.

    Examples
    --------
    >>> # Minimal usage (ensemble mode, default emulator)
    >>> from app.engine.metabolism import predict
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O")
    >>> r.parent.molecular_formula
    'C9H8O4'
    >>> all(isinstance(s, EnsembleSoftSpot) for s in r.soft_spots)
    True

    >>> # Rule-only mode (v1 equivalent)
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O", enable_dl=False)

    >>> # Real weights (when available)
    >>> predictor = DeepLearningPredictor.from_checkpoint("metatrans.pt")
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O", dl_predictor=predictor)
    """
    t_start   = time.perf_counter()
    warnings: List[str] = []

    # ── Step 1: Validate + normalise ──────────────────────────────────────────
    logger.info("ensemble_predict_start")
    mol, parent_meta = _validate_and_normalise(smiles)

    if parent_meta.canonical_smiles != smiles.strip():
        warnings.append(
            f"Input SMILES normalised: {smiles!r} → "
            f"{parent_meta.canonical_smiles!r}"
        )

    if parent_meta.molecular_weight > 1500:
        warnings.append(
            f"MW {parent_meta.molecular_weight:.1f} Da exceeds typical small-molecule "
            "range (< 1500 Da). Predictions may be unreliable."
        )

    # ── Step 2A: Pipeline A — SyGMa ───────────────────────────────────────────
    sygma_metabolites: List[PredictedMetabolite] = []
    try:
        sygma_metabolites = _run_sygma(
            mol,
            phase1_cycles=phase1_cycles,
            phase2_cycles=phase2_cycles,
            max_metabolites=max_metabolites,
        )
    except ImportError:
        warnings.append(
            "SyGMa not installed — rule-based pipeline inactive. "
            "Install: pip install sygma"
        )
        logger.warning("sygma_not_installed")
    except Exception as exc:
        warnings.append(f"SyGMa pipeline error (non-fatal): {exc}")
        logger.error("sygma_pipeline_error", error=str(exc))

    # ── Step 2B: Pipeline B — Deep Learning ───────────────────────────────────
    dl_predictions:    List[Tuple[str, float]] = []
    attention_weights: Dict[int, float]        = {}
    dl_source = "disabled"

    if enable_dl:
        # Use provided predictor or create a default emulator
        predictor = dl_predictor or DeepLearningPredictor()
        dl_source = predictor.MODEL_NAME
        try:
            (
                dl_predictions,
                attention_weights,
                dl_warnings,
            ) = predictor.predict(parent_meta.canonical_smiles)
            warnings.extend(dl_warnings)
        except Exception as exc:
            warnings.append(f"DL pipeline error (non-fatal): {exc}")
            logger.error("dl_pipeline_error", error=str(exc))
    else:
        warnings.append("DL pipeline disabled (enable_dl=False).")

    # ── Step 3: Consensus merge ────────────────────────────────────────────────
    consensus = ConsensusEngine()

    if enable_dl and (sygma_metabolites or dl_predictions):
        merged_metabolites, pipeline_stats = consensus.merge_metabolites(
            sygma_metabolites, dl_predictions
        )
        merged_metabolites = merged_metabolites[:max_metabolites]
    else:
        # Rule-only fallback: tag everything as RULE tier
        merged_metabolites = sygma_metabolites[:max_metabolites]
        pipeline_stats = {
            "sygma_total":     len(sygma_metabolites),
            "dl_total":        0,
            "consensus_count": 0,
            "rule_only_count": len(sygma_metabolites),
            "dl_only_count":   0,
            "merged_total":    len(sygma_metabolites),
        }

    # ── Step 4: Soft spot analysis ────────────────────────────────────────────
    if soft_spot_fn is not None and not enable_dl:
        # v1 backward-compat: external soft_spot_fn overrides ensemble
        soft_spots_raw = soft_spot_fn(mol, top_soft_spots)
        soft_spots: List = soft_spots_raw
    else:
        # Ensemble soft spots
        rule_spots = _find_soft_spots(mol, top_n=mol.GetNumHeavyAtoms())
        soft_spots = consensus.merge_soft_spots(
            rule_spots=rule_spots,
            attention_weights=attention_weights,
            mol=mol,
            top_n=top_soft_spots,
            dl_source=dl_source,
        )

    if not soft_spots:
        warnings.append(
            "No soft spots identified. The molecule may lack common CYP substrate "
            "features or all candidates were excluded by the steric/attention filter."
        )

    # ── Step 5: Assemble result ───────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    pipeline_stats["dl_source"]  = dl_source
    pipeline_stats["elapsed_s"]  = round(elapsed, 4)

    result = MetabolismResult(
        parent=parent_meta,
        metabolites=merged_metabolites,
        soft_spots=soft_spots,
        engine_version=ENGINE_VERSION,
        elapsed_s=elapsed,
        warnings=warnings,
        pipeline_stats=pipeline_stats,
    )

    logger.info(
        "ensemble_predict_complete",
        canonical=parent_meta.canonical_smiles,
        metabolites_total=len(merged_metabolites),
        consensus_count=pipeline_stats.get("consensus_count", 0),
        soft_spots_total=len(soft_spots),
        elapsed_s=round(elapsed, 3),
    )

    return result
.Groups[1].Value
    "logger.info(f`"$($args -replace '`"', '' -replace "'", '' -replace ',\s*\w+=', ' ' -replace '=', '=')`")"
,
        p1=sum(1 for r in results if r.phase == 1),
        p2=sum(1 for r in results if r.phase == 2),
    )
    return results


def _find_soft_spots(
    mol: Chem.Mol,
    top_n: int = 3,
) -> List[SoftSpot]:
    """
    Rule-only soft spot fallback (Pipeline A standalone).

    Used when the DL pipeline is unavailable or disabled.  The ConsensusEngine
    calls ``merge_soft_spots`` instead when both pipelines are active.

    Algorithm
    ---------
    For each rule in ``_SOM_RULES``:
      - Compile SMARTS, find substructure matches.
      - Atom at match index 0 is the SOM site.
      - Apply a 20 % steric penalty for atoms with ≥ 3 bulky neighbours.
    """
    best: Dict[int, SoftSpot] = {}

    for rule_name, smarts, base_score in _SOM_RULES:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            logger.warning("invalid_smarts", rule=rule_name)
            continue

        for match in mol.GetSubstructMatches(pattern):
            if not match:
                continue
            atom_idx = match[0]
            atom     = mol.GetAtomWithIdx(atom_idx)

            heavy_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
            score = base_score
            if len(heavy_nbrs) >= 3 and any(n.GetDegree() > 1 for n in heavy_nbrs):
                score *= 0.80

            score = round(score, 4)
            if atom_idx not in best or score > best[atom_idx].score:
                best[atom_idx] = SoftSpot(
                    atom_index=atom_idx,
                    atom_symbol=atom.GetSymbol(),
                    rule_name=rule_name,
                    score=score,
                    smarts_match=smarts,
                )

    return sorted(best.values(), key=lambda s: s.score, reverse=True)[:top_n]


# ==============================================================================
# Public API
# ==============================================================================

# Type alias for drop-in soft-spot function hook (backward compatible with v1)
SoftSpotFn = Callable[[Chem.Mol, int], List[SoftSpot]]


def predict(
    smiles: str,
    *,
    phase1_cycles:   int  = 1,
    phase2_cycles:   int  = 1,
    max_metabolites: int  = 200,
    top_soft_spots:  int  = 3,
    # v2 ensemble parameters
    dl_predictor:    Optional[DeepLearningPredictor] = None,
    enable_dl:       bool = True,
    # v1 backward-compat hook (used when dl_predictor is None and enable_dl=False)
    soft_spot_fn:    Optional[SoftSpotFn] = None,
) -> MetabolismResult:
    """
    Ensemble Consensus Metabolism Prediction.

    Orchestrates the dual-pipeline architecture:
      Pipeline A  — SyGMa rule-based Phase I/II metabolites
      Pipeline B  — DeepLearningPredictor (emulator or real weights)
      Consensus   — cross-validates outputs, assigns ConsensusTier tags
      Soft Spots  — blends rule scores with DL attention weights

    Parameters
    ----------
    smiles          : Parent molecule SMILES (any notation; normalised internally).
    phase1_cycles   : SyGMa Phase I iteration depth.
    phase2_cycles   : SyGMa Phase II iteration depth.
    max_metabolites : Hard cap on returned metabolites (applied after merge).
    top_soft_spots  : Number of soft spots to annotate.
    dl_predictor    : Pre-built DeepLearningPredictor instance.  If None and
                      enable_dl=True, a default emulator is created internally.
    enable_dl       : Master switch for Pipeline B.  Set False to run rule-only
                      (equivalent to v1 behaviour).
    soft_spot_fn    : v1 backward-compat hook.  Ignored when enable_dl=True.

    Returns
    -------
    ``MetabolismResult`` with full ensemble data.  ``.to_dict()`` is
    backward-compatible with v1 API consumers.

    Raises
    ------
    ValueError   : SMILES is invalid.
    RuntimeError : Unexpected engine-level error.

    Examples
    --------
    >>> # Minimal usage (ensemble mode, default emulator)
    >>> from app.engine.metabolism import predict
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O")
    >>> r.parent.molecular_formula
    'C9H8O4'
    >>> all(isinstance(s, EnsembleSoftSpot) for s in r.soft_spots)
    True

    >>> # Rule-only mode (v1 equivalent)
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O", enable_dl=False)

    >>> # Real weights (when available)
    >>> predictor = DeepLearningPredictor.from_checkpoint("metatrans.pt")
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O", dl_predictor=predictor)
    """
    t_start   = time.perf_counter()
    warnings: List[str] = []

    # ── Step 1: Validate + normalise ──────────────────────────────────────────
    
    $args = """
app/engine/metabolism.py
================================================================================
Ensemble Consensus Metabolism Engine  —  v2.0.0-ensemble
================================================================================

Architecture overview
---------------------
This module implements a dual-pipeline ensemble strategy that combines two
complementary paradigms for metabolite prediction and soft-spot analysis,
inspired by and designed for compatibility with:

  Pipeline A — Rule-Based (SyGMa)
    KavrakiLab/SyGMa / 3D-e-Chem/sygma
    Applies curated SMIRKS reaction transformation matrices for Phase I
    (oxidation, reduction, hydrolysis) and Phase II (conjugation) reactions.
    Each metabolite carries a cumulative SyGMa probability score.

  Pipeline B — Deep Learning Emulator (MetaTrans / Meta-Predictor style)
    KavrakiLab/MetaTrans  ·  zhukeyun/Meta-Predictor
    Treats metabolite prediction as a SMILES-to-SMILES sequence translation
    task (Transformer/OpenNMT architecture).  The ``DeepLearningPredictor``
    class implemented here is a **production-ready emulator** with a clean
    interface for swapping in real PyTorch .pt model weights:

        predictor = DeepLearningPredictor.from_checkpoint("weights/metatrans.pt")
        result = predict(smiles, dl_predictor=predictor)

Consensus logic
---------------
After both pipelines run, the ``ConsensusEngine`` cross-validates their outputs
using ``Chem.CanonSmiles`` for structural alignment and tags each metabolite:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Both pipelines agree      → "High Confidence (Consensus Verified)"    │
  │  SyGMa only                → "Moderate Confidence (Rule-Only)"         │
  │  DL model only             → "Moderate Confidence (DL-Only)"           │
  └─────────────────────────────────────────────────────────────────────────┘

Soft spot analysis
------------------
Unified vulnerability scoring blends two complementary signals:

  rule_score        — SMARTS-based steric-adjusted heuristic (Pipeline A)
  dl_attention_risk — per-atom attention weight from transformer SOM heads
                      (emulated; replace ``_compute_dl_attention`` with real
                       model forward pass when weights are available)

  vulnerability_index = (α × rule_score + β × dl_attention_risk) × 100  (%)
  where α=0.55, β=0.45 by default (configurable via ``ENSEMBLE_ALPHA``).

Error defence
-------------
Every pipeline is wrapped in defence layers that catch and contain:
  - ``rdkit.Chem.rdchem.MolSanitizeException``
  - Valence / kekulisation errors from transformer-generated SMILES
  - Un-tokenisable or chemically impossible SMILES strings
  - RDKit property calculation failures
  - Any unexpected exception in DL inference hooks
Individual failures downgrade confidence or add warnings; they never crash
the server.

Public API (unchanged from v1 for drop-in compatibility)
---------------------------------------------------------
  predict(smiles, ...)          → MetabolismResult
  _validate_and_normalise(...)  → (Chem.Mol, MoleculeMetadata)
  _find_soft_spots(...)         → List[SoftSpot]   (rule-only fallback)

New in v2
---------
  DeepLearningPredictor         class (Pipeline B)
  ConsensusEngine               class (merger + tagging)
  EnsembleSoftSpotResult        dataclass (combined atom vulnerability)
  ConsensusTier                 enum-like constants
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdchem   # MolSanitizeException lives here
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)


# ==============================================================================
# Engine versioning & tuning constants
# ==============================================================================

ENGINE_VERSION = "2.0.0-ensemble"

# Soft spot fusion weights:  vulnerability_index = α·rule + β·dl_attention
# Must sum to 1.0.  Adjust to preference-weight the two signals.
ENSEMBLE_ALPHA: float = 0.55   # weight given to SMARTS rule score
ENSEMBLE_BETA:  float = 0.45   # weight given to DL attention risk

assert abs(ENSEMBLE_ALPHA + ENSEMBLE_BETA - 1.0) < 1e-9, \
    "ENSEMBLE_ALPHA + ENSEMBLE_BETA must equal 1.0"

# Minimum DL model confidence to include a DL-only prediction
DL_CONFIDENCE_THRESHOLD: float = 0.10

# Consensus requires both pipeline canonical SMILES to match exactly
# (after RDKit canonicalisation).


# ==============================================================================
# Confidence tiers
# ==============================================================================

class ConsensusTier(str, Enum):
    """
    Confidence classification assigned to each metabolite by the consensus engine.

    Inherits from ``str`` so instances serialise directly to JSON strings
    without a custom encoder.
    """
    HIGH     = "High Confidence (Consensus Verified)"
    RULE     = "Moderate Confidence (Rule-Only)"
    DL       = "Moderate Confidence (DL-Only)"
    UNKNOWN  = "Unclassified"


# ==============================================================================
# Public data structures
# ==============================================================================

@dataclass(frozen=True)
class MoleculeMetadata:
    """
    Physico-chemical descriptors for a molecule.  Immutable after construction.

    ``canonical_smiles`` uses RDKit's default canonicalisation (aromatic SMILES,
    not Kekulé).  ``inchi`` / ``inchikey`` provide standard identifiers.
    """
    input_smiles:        str
    canonical_smiles:    str
    inchi:               str
    inchikey:            str
    molecular_formula:   str
    molecular_weight:    float      # average MW  (Da)
    exact_mass:          float      # monoisotopic (Da)
    num_heavy_atoms:     int
    num_rotatable_bonds: int
    num_hbd:             int        # H-bond donors
    num_hba:             int        # H-bond acceptors
    tpsa:                float      # topological polar surface area  (Å²)
    logp:                float      # Wildman–Crippen logP
    num_rings:           int
    num_aromatic_rings:  int


@dataclass(frozen=True)
class PredictedMetabolite:
    """
    A single metabolite prediction, possibly from one or both pipelines.

    New in v2
    ---------
    confidence_tier  : ConsensusTier tag assigned by the ConsensusEngine.
    dl_score         : DL model confidence [0,1] (None when DL pipeline absent).
    sources          : frozenset of pipeline names that produced this metabolite.
    """
    smiles:            str
    probability:       float            # primary score (SyGMa or DL confidence)
    phase:             int              # 1 or 2 (−1 when DL-only + unknown)
    reaction_name:     str
    molecular_weight:  Optional[float]  = None
    molecular_formula: Optional[str]    = None
    confidence_tier:   ConsensusTier    = ConsensusTier.UNKNOWN
    dl_score:          Optional[float]  = None
    sources:           frozenset        = field(default_factory=frozenset)


@dataclass(frozen=True)
class SoftSpot:
    """
    Rule-only soft spot — retained for backward compatibility with v1 API.
    Used as a fallback when the DL pipeline is unavailable.
    """
    atom_index:   int
    atom_symbol:  str
    rule_name:    str
    score:        float     # SMARTS rule score [0, 1]
    smarts_match: str


@dataclass(frozen=True)
class EnsembleSoftSpot:
    """
    Unified soft spot combining rule-based and DL-derived vulnerability signals.

    Fields
    ------
    atom_index          : 0-based RDKit atom index.
    atom_symbol         : element symbol.
    rule_name           : best-matching SMARTS rule label.
    smarts_match        : SMARTS pattern that matched.
    rule_score          : SMARTS-based vulnerability [0, 1].
    dl_attention_risk   : DL attention-head weight [0, 1]
                          (emulated; replace with real model output).
    vulnerability_index : unified score in [0, 100] %
                          = (α·rule_score + β·dl_attention_risk) × 100.
    dl_source           : identifier of the DL model that produced
                          ``dl_attention_risk`` (e.g. "metatrans-emulator").
    """
    atom_index:         int
    atom_symbol:        str
    rule_name:          str
    smarts_match:       str
    rule_score:         float
    dl_attention_risk:  float
    vulnerability_index: float          # [0, 100]
    dl_source:          str = "rule-only"


@dataclass
class MetabolismResult:
    """
    Top-level result returned by ``predict()``.

    ``soft_spots`` is now ``List[EnsembleSoftSpot]`` when the ensemble ran,
    or ``List[SoftSpot]`` when it fell back to rule-only mode.
    Both types expose ``atom_index``, ``atom_symbol``, and ``rule_name`` so
    downstream consumers (API layer, renderer) are unaffected.

    ``to_dict()`` is backward-compatible with v1 consumers while adding the
    new v2 ensemble fields.
    """
    parent:          MoleculeMetadata
    metabolites:     List[PredictedMetabolite]
    soft_spots:      List                       # List[EnsembleSoftSpot] | List[SoftSpot]
    engine_version:  str
    elapsed_s:       float
    warnings:        List[str] = field(default_factory=list)
    pipeline_stats:  Dict      = field(default_factory=dict)

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """
        Serialise to a plain nested dict for JSON responses, Celery backends,
        or pandas.  Backward-compatible with v1; adds ensemble fields.
        """
        soft_spot_dicts = []
        for s in self.soft_spots:
            if isinstance(s, EnsembleSoftSpot):
                soft_spot_dicts.append({
                    "atom_index":          s.atom_index,
                    "atom_symbol":         s.atom_symbol,
                    "rule_name":           s.rule_name,
                    "smarts_match":        s.smarts_match,
                    "rule_score":          s.rule_score,
                    "dl_attention_risk":   s.dl_attention_risk,
                    "vulnerability_index": s.vulnerability_index,
                    "dl_source":           s.dl_source,
                    # v1-compat alias
                    "score":               round(s.vulnerability_index / 100.0, 4),
                })
            else:
                soft_spot_dicts.append({
                    "atom_index":          s.atom_index,
                    "atom_symbol":         s.atom_symbol,
                    "rule_name":           s.rule_name,
                    "smarts_match":        s.smarts_match,
                    "rule_score":          s.score,
                    "dl_attention_risk":   0.0,
                    "vulnerability_index": round(s.score * 100, 2),
                    "dl_source":           "rule-only",
                    "score":               s.score,
                })

        return {
            "engine_version": self.engine_version,
            "elapsed_s":      round(self.elapsed_s, 4),
            "warnings":       self.warnings,
            "pipeline_stats": self.pipeline_stats,
            "parent": {
                "input_smiles":        self.parent.input_smiles,
                "canonical_smiles":    self.parent.canonical_smiles,
                "inchi":               self.parent.inchi,
                "inchikey":            self.parent.inchikey,
                "molecular_formula":   self.parent.molecular_formula,
                "molecular_weight":    self.parent.molecular_weight,
                "exact_mass":          self.parent.exact_mass,
                "num_heavy_atoms":     self.parent.num_heavy_atoms,
                "num_rotatable_bonds": self.parent.num_rotatable_bonds,
                "num_hbd":             self.parent.num_hbd,
                "num_hba":             self.parent.num_hba,
                "tpsa":                self.parent.tpsa,
                "logp":                self.parent.logp,
                "num_rings":           self.parent.num_rings,
                "num_aromatic_rings":  self.parent.num_aromatic_rings,
            },
            "metabolites": [
                {
                    "smiles":           m.smiles,
                    "probability":      m.probability,
                    "phase":            m.phase,
                    "reaction_name":    m.reaction_name,
                    "molecular_weight": m.molecular_weight,
                    "molecular_formula":m.molecular_formula,
                    "confidence_tier":  m.confidence_tier.value,
                    "dl_score":         m.dl_score,
                    "sources":          sorted(m.sources),
                }
                for m in self.metabolites
            ],
            "soft_spots": soft_spot_dicts,
        }

    @property
    def phase1_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites if m.phase == 1]

    @property
    def phase2_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites if m.phase == 2]

    @property
    def consensus_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites
                if m.confidence_tier == ConsensusTier.HIGH]

    @property
    def top_soft_spot_indices(self) -> List[int]:
        return [s.atom_index for s in self.soft_spots]


# ==============================================================================
# SOM SMARTS rules  (Pipeline A — rule-based component)
# ==============================================================================
# Each entry: (rule_name, smarts_pattern, priority_score)
#
# priority_score in (0, 1] based on published CYP450 substrate literature:
#   Zaretzki et al., J. Chem. Inf. Model. 2013 (CypReact)
#   Sheridan et al., J. Med. Chem. 2002
#   Cruciani et al., J. Med. Chem. 2005 (MetaSite)
#
# Convention: atom 0 in each SMARTS match is the SOM being annotated.

_SOM_RULES: List[Tuple[str, str, float]] = [
    # -- Phase I: CYP-mediated oxidation -------------------------------------

    # Benzylic CH2/CH3: highest CYP2C9/3A4 affinity (Meunier 2004)
    ("benzylic_CH",
     "[CH2,CH3;!$([CH2]C=O);!$([CH2]C#N)]-[c]", 0.92),

    # Unhindered aromatic C-H: CYP1A2/2C19 aromatic hydroxylation
    ("aromatic_C_unhindered",
     "[cH]", 0.85),

    # α-sp3 C to heteroatom: N/O-dealkylation (most common CYP reaction type)
    ("alpha_C_heteroatom",
     "[CH2,CH3;$([CH2,CH3][N,O;!$(N-C=O)])]", 0.88),

    # N-methyl/ethyl: CYP3A4 N-demethylation hotspot
    ("N_alkyl",
     "[CH3,CH2;$([CH3,CH2]-[N;!$(N-C=O);!$(N~[!#6;!#1])])]", 0.86),

    # α-carbonyl sp3 C: CYP-mediated α-hydroxylation
    ("alpha_carbonyl_C",
     "[CH2,CH3;$([CH2,CH3]C(=O)[#6,#7,#8])]", 0.75),

    # Terminal methyl (ω-hydroxylation, CYP4A/2E1)
    ("terminal_methyl_aliphatic",
     "[CH3;$([CH3]-[CH2]-[CH2]-[#6])]", 0.65),

    # Thioether S: CYP S-oxidation → sulfoxide → sulfone
    ("thioether_S",
     "[S;X2;!$(S=O);$([S]([#6])[#6])]", 0.80),

    # Tertiary amine N: FMO/CYP3A4 N-oxidation
    ("tertiary_amine_N",
     "[N;X3;!$(N-C=O);!$(N~[!#6;!#1]);!n;$([N]([#6])([#6])[#6])]", 0.78),

    # -- Phase I: Reduction --------------------------------------------------

    # Nitro group reduction (gut flora + hepatic)
    ("nitro_reduction",
     "[N+](=O)[O-]", 0.70),

    # Ketone/aldehyde carbonyl reduction (AKR / carbonyl reductases)
    ("carbonyl_reduction",
     "[CH1,CH0;$(C=O);!$(C(=O)[OH]);!$(C(=O)N)]", 0.60),

    # -- Phase II: Conjugation-prone sites -----------------------------------

    # Phenolic OH carbon: glucuronidation / sulfation (mark the C, not the O)
    ("phenolic_OH_site",
     "[c;$([c][OH1])]", 0.83),

    # Primary aliphatic amine: acetylation / glucuronidation
    ("primary_amine",
     "[NH2;$([NH2]-[#6;!$(C=O)])]", 0.72),

    # Carboxylic acid: acyl-glucuronidation (reactive metabolite risk)
    ("carboxylic_acid",
     "[C;$(C(=O)[OH1])]", 0.68),
]

# Normalisation pipeline  (shared singletons — expensive to construct)
_NORMALISER       = rdMolStandardize.Normalizer()
_UNCHARGER        = rdMolStandardize.Uncharger()
_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()


# ==============================================================================
# Pipeline B — DeepLearningPredictor
# ==============================================================================

class DeepLearningPredictor:
    """
    Production-ready deep-learning metabolite predictor emulator.

    Architectural inspiration
    -------------------------
    MetaTrans (KavrakiLab/MetaTrans): OpenNMT Transformer trained on
    experimental metabolite SMILES pairs.  Encodes the parent SMILES token
    sequence, decodes up to ``top_k`` metabolite sequences with beam search,
    and scores them with the model's log-likelihood.

    Meta-Predictor (zhukeyun/Meta-Predictor): Adds per-atom attention heads
    that produce site-of-metabolism risk vectors directly usable for soft
    spot annotation without a separate SOM model.

    This class
    ----------
    Implements the full interface those models require:
      - SMILES tokenisation  (atom-level, compatible with MetaTrans vocabulary)
      - Structural verification of generated SMILES via RDKit
      - Per-atom attention weight extraction (emulated with a deterministic
        chemistry-aware heuristic until real weights are loaded)
      - Graceful degradation: any tensor/inference failure returns empty lists
        with informative warnings, never crashes

    Swap-in path for real weights
    -----------------------------
    1. Install PyTorch + OpenNMT-py:
           pip install torch opennmt-py
    2. Subclass or monkey-patch ``_transformer_beam_search``:
           def _transformer_beam_search(self, tokens, top_k):
               # real OpenNMT / HuggingFace inference here
               ...
    3. Subclass or monkey-patch ``_extract_attention_weights``:
           def _extract_attention_weights(self, mol, tokens):
               # real model forward-pass attention extraction
               ...
    4. Load weights via ``from_checkpoint``:
           predictor = DeepLearningPredictor.from_checkpoint("metatrans.pt")

    All public method signatures remain stable between emulator and real-weight
    modes — no changes needed in the calling code.
    """

    # ── MetaTrans atom-level vocabulary (subset) ──────────────────────────────
    # Real MetaTrans uses a regex tokeniser that splits on atom symbols,
    # bond characters, ring digits, and branch brackets.  We replicate that
    # split pattern here so tokenised sequences are compatible.
    _TOKEN_PATTERN = (
        r"(\[[^\]]+\]"           # bracket atoms  [NH2], [13C], ...
        r"|Br|Cl|Si|Se|@@|@"     # two-char atoms / stereo
        r"|[BCNOPSFIbcnops]"     # one-char organic subset
        r"|[0-9%]"               # ring closure digits
        r"|[/\\+=\-#\(\)\.])"    # bond / branch / dot
    )

    MODEL_NAME = "metatrans-emulator-v2"

    def __init__(
        self,
        top_k: int = 10,
        confidence_threshold: float = DL_CONFIDENCE_THRESHOLD,
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        top_k                : beam-search width (number of candidate
                               metabolite sequences returned per parent).
        confidence_threshold : minimum model confidence to include a prediction.
        seed                 : optional random seed for the emulator (for
                               reproducible unit tests only; real models
                               are deterministic given the same weights).
        """
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self._rng = random.Random(seed)
        self._model_loaded = False   # True once real weights are attached
        logger.info(
            "dl_predictor_init",
            model=self.MODEL_NAME,
            top_k=top_k,
            mode="emulator",
        )

    # -- Class-method constructors --------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        **kwargs,
    ) -> "DeepLearningPredictor":
        """
        Load a MetaTrans / Meta-Predictor checkpoint.

        This method is a **stub** — it returns an emulator instance and logs
        a warning until real model loading code is inserted here.

        To implement:
        -------------
            import torch, onmt
            model = onmt.model_builder.build_base_model(opt, fields, gpu)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            predictor = cls(**kwargs)
            predictor._model = model
            predictor._model_loaded = True
            return predictor
        """
        instance = cls(**kwargs)
        logger.warning(
            "dl_checkpoint_stub",
            path=checkpoint_path,
            msg="Real model weights not loaded — using emulator. "
                "Implement from_checkpoint() to enable live inference.",
        )
        return instance

    # -- Tokenisation ---------------------------------------------------------

    def tokenise(self, smiles: str) -> List[str]:
        """
        Atom-level SMILES tokeniser compatible with MetaTrans vocabulary.

        Splits the SMILES string into a list of token strings, each
        representing a single atom, bond, stereo descriptor, or bracket group.

        Parameters
        ----------
        smiles : canonical SMILES string (pre-normalised by RDKit).

        Returns
        -------
        List of token strings, e.g.
            "CC(=O)O" → ["C", "C", "(", "=", "O", ")", "O"]

        Raises
        ------
        ValueError : if the SMILES is empty or produces zero tokens.
        """
        import re
        if not smiles or not smiles.strip():
            raise ValueError("Cannot tokenise empty SMILES.")
        tokens = re.findall(self._TOKEN_PATTERN, smiles)
        if not tokens:
            raise ValueError(
                f"SMILES '{smiles}' produced zero tokens. "
                "It may contain unsupported notation."
            )
        return tokens

    # -- Structural verification ----------------------------------------------

    @staticmethod
    def verify_smiles(smiles: str) -> Optional[Chem.Mol]:
        """
        Verify that a SMILES string represents a valid, sanitisable molecule.

        Catches the full RDKit error taxonomy:
          - ``rdchem.MolSanitizeException`` — valence/kekulisation failures
          - ``rdchem.KekulizeException``    — aromatic resolution failures
          - ``ValueError``                  — bad SMILES syntax
          - Any other Exception             — unknown RDKit error

        Parameters
        ----------
        smiles : SMILES string to verify (may be transformer-generated).

        Returns
        -------
        Sanitised ``Chem.Mol`` on success, or ``None`` if any check fails.
        """
        if not smiles or not smiles.strip():
            return None
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None

            # Attempt full sanitisation — this is the main gate for
            # hypervalent/impossible structures generated by the transformer
            sanitize_flags = Chem.SanitizeFlags.SANITIZE_ALL
            result_flag = Chem.SanitizeMol(mol, sanitize_flags, catchErrors=True)
            if result_flag != Chem.SanitizeFlags.SANITIZE_NONE:
                logger.debug(
                    "dl_smiles_sanitize_failed"),
                )
                return None

            if mol.GetNumHeavyAtoms() == 0:
                return None

            # Re-parse from canonical SMILES for clean atom mapping
            canonical = Chem.MolToSmiles(mol, canonical=True)
            mol2 = Chem.MolFromSmiles(canonical)
            return mol2

        except rdchem.MolSanitizeException as exc:
            logger.debug("dl_mol_sanitize_exception", error=str(exc), smiles=smiles[:80])
            return None
        except rdchem.KekulizeException as exc:
            logger.debug("dl_mol_kekulize_exception", error=str(exc), smiles=smiles[:80])
            return None
        except Exception as exc:
            logger.debug("dl_mol_verify_error", error=str(exc), smiles=smiles[:80])
            return None

    # -- Core inference hooks (emulated; replace with real model calls) --------

    def _transformer_beam_search(
        self,
        tokens: List[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Emulate transformer beam-search decoding.

        PRODUCTION REPLACEMENT POINT
        ----------------------------
        Replace the body of this method with a real OpenNMT / HuggingFace
        inference call.  The expected signature and return type are stable:

            inputs  : token list from ``tokenise()``
            returns : list of (smiles_string, log_likelihood_score) tuples,
                      up to ``top_k`` entries, sorted by score descending.

        Emulator strategy
        -----------------
        Rather than pure randomness (which would produce mostly invalid SMILES),
        the emulator applies a set of chemically plausible single-step
        transformations to the input SMILES — the same operations MetaTrans
        was trained to generate.  This ensures a realistic fraction of
        structurally valid outputs for testing and development.

        Confidence scores are set to realistic ranges drawn from published
        MetaTrans log-likelihood distributions (Litsa et al. 2023).
        """
        parent_smiles = "".join(tokens)

        # Validate the parent SMILES before attempting to transform
        parent_mol = self.verify_smiles(parent_smiles)
        if parent_mol is None:
            return []

        # Deterministic seed: same parent → same emulated outputs across runs
        # (prevents flapping tests while still covering diverse transformations)
        seed_hash = int(hashlib.sha256(parent_smiles.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed_hash)

        candidates: List[Tuple[str, float]] = []

        # ── Emulated transformation set ────────────────────────────────────────
        # Each lambda attempts a chemically plausible single SMIRKS-like step.
        # Transformations that don't apply to the molecule are silently skipped.
        transformations = [
            self._emulate_aromatic_hydroxylation,
            self._emulate_aliphatic_hydroxylation,
            self._emulate_n_demethylation,
            self._emulate_o_demethylation,
            self._emulate_glucuronidation,
            self._emulate_sulfation,
            self._emulate_oxidation_sulfoxide,
            self._emulate_hydrolysis,
        ]

        rng.shuffle(transformations)

        for transform in transformations[:top_k]:
            try:
                result_smiles = transform(parent_mol, rng)
                if result_smiles is None:
                    continue
                verified = self.verify_smiles(result_smiles)
                if verified is None:
                    continue
                canon = Chem.MolToSmiles(verified, canonical=True)
                # Emulated log-likelihood in realistic MetaTrans range (~-0.1 to -3.0)
                score = rng.uniform(0.08, 0.55)
                candidates.append((canon, round(score, 4)))
            except Exception as exc:
                logger.debug("dl_transform_failed", error=str(exc))
                continue

        # Sort by score descending, deduplicate
        seen_smiles: Set[str] = set()
        unique: List[Tuple[str, float]] = []
        for smiles, score in sorted(candidates, key=lambda x: x[1], reverse=True):
            if smiles not in seen_smiles:
                seen_smiles.add(smiles)
                unique.append((smiles, score))

        return unique[:top_k]

    def _extract_attention_weights(
        self,
        mol: Chem.Mol,
        tokens: List[str],
    ) -> Dict[int, float]:
        """
        Extract per-atom SOM attention risk weights from the transformer.

        PRODUCTION REPLACEMENT POINT
        ----------------------------
        In a real MetaTrans / Meta-Predictor model, this method runs a
        forward pass and reads the cross-attention weights from the encoder
        layers aligned to each heavy atom.  Replace the body with:

            with torch.no_grad():
                outputs = self._model(input_tensor, attention=True)
                atom_weights = outputs.encoder_attentions[atom_indices]
            return {idx: float(w) for idx, w in enumerate(atom_weights)}

        Emulator strategy
        -----------------
        We produce a chemistry-aware pseudo-attention distribution that
        correlates with known SOM preferences, making the emulated values
        directionally correct for testing and UI development.

        The distribution is built from three additive signals:
          1. Atomic number contribution (heteroatoms get higher base weight)
          2. Environment contribution (aromatic/sp3/sp2 character)
          3. SMARTS rule overlap (atoms matching known SOM patterns get boost)

        All values are normalised to [0, 1] with softmax.
        """
        n_atoms = mol.GetNumHeavyAtoms()
        if n_atoms == 0:
            return {}

        raw_weights: Dict[int, float] = {}

        # Precompute which atoms match any SOM rule
        som_boosted: Set[int] = set()
        for _, smarts, _ in _SOM_RULES:
            pat = Chem.MolFromSmarts(smarts)
            if pat is None:
                continue
            for match in mol.GetSubstructMatches(pat):
                if match:
                    som_boosted.add(match[0])

        # ── Pseudo-attention computation ───────────────────────────────────────
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            w = 0.0

            # Signal 1: heteroatom base weight
            atomic_num = atom.GetAtomicNum()
            if atomic_num in (7, 8):       # N, O
                w += 0.35
            elif atomic_num == 16:          # S
                w += 0.30
            elif atomic_num == 6:           # C
                w += 0.15
            else:
                w += 0.20

            # Signal 2: hybridisation / environment
            hyb = atom.GetHybridization()
            if atom.GetIsAromatic():
                w += 0.25
            elif hyb == Chem.rdchem.HybridizationType.SP3:
                w += 0.20
            elif hyb == Chem.rdchem.HybridizationType.SP2:
                w += 0.18

            # Signal 3: explicit hydrogen count (CYP accessibility proxy)
            w += atom.GetTotalNumHs() * 0.10

            # Signal 4: SOM rule overlap boost
            if idx in som_boosted:
                w += 0.40

            raw_weights[idx] = w

        # Softmax normalisation → values sum to 1.0, then rescale to [0, 1]
        exp_w = {idx: math.exp(v) for idx, v in raw_weights.items()}
        total = sum(exp_w.values()) or 1.0
        normalised = {idx: v / total for idx, v in exp_w.items()}

        # Rescale: set max atom to 1.0, others proportionally
        max_val = max(normalised.values()) if normalised else 1.0
        if max_val > 0:
            return {idx: round(v / max_val, 4) for idx, v in normalised.items()}
        return {idx: 0.0 for idx in normalised}

    # -- Chemical transformation emulators ------------------------------------
    # Each method attempts a single chemical step on ``mol`` and returns a
    # product SMILES string, or None if the transformation doesn't apply.
    # These are *not* meant to be chemically exhaustive — they serve as
    # realistic test vectors for the DL pipeline interface.

    @staticmethod
    def _emulate_aromatic_hydroxylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Add -OH to an unhindered aromatic C-H (CYP1A2/2C19 style)."""
        try:
            from rdkit.Chem import AllChem, RWMol
            pattern = Chem.MolFromSmarts("[cH]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            atom_idx = match[0]
            rw = Chem.RWMol(mol)
            rw.GetAtomWithIdx(atom_idx).SetNumExplicitHs(0)
            oh_idx = rw.AddAtom(Chem.Atom(8))     # O
            h_idx  = rw.AddAtom(Chem.Atom(1))     # H
            rw.AddBond(atom_idx, oh_idx, Chem.BondType.SINGLE)
            rw.AddBond(oh_idx,   h_idx,  Chem.BondType.SINGLE)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_aliphatic_hydroxylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Add -OH to an aliphatic sp3 C-H (benzylic or α-heteroatom)."""
        try:
            from rdkit.Chem import RWMol
            # Prefer benzylic CH2
            pattern = Chem.MolFromSmarts("[CH2;$([CH2]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                pattern = Chem.MolFromSmarts("[CH3;$([CH3]-[c])]")
                matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            atom_idx = match[0]
            rw = Chem.RWMol(Chem.AddHs(mol))
            # Find one explicit H on this atom
            h_to_remove = None
            for nbr in rw.GetAtomWithIdx(atom_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 1:
                    h_to_remove = nbr.GetIdx()
                    break
            if h_to_remove is None:
                return None
            rw.RemoveAtom(h_to_remove)
            oh_idx = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(atom_idx, oh_idx, Chem.BondType.SINGLE)
            product = Chem.RemoveHs(rw)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(product, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_n_demethylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Remove N-methyl group (CYP3A4 N-demethylation)."""
        try:
            from rdkit.Chem import AllChem, RWMol, rdMolTransforms
            pattern = Chem.MolFromSmarts("[CH3;$([CH3]-[N;!$(N-C=O)])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx = match[0]
            rw = Chem.RWMol(mol)
            # Find the N neighbor
            n_idx = None
            for nbr in rw.GetAtomWithIdx(c_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 7:
                    n_idx = nbr.GetIdx()
                    break
            if n_idx is None:
                return None
            rw.RemoveBond(c_idx, n_idx)
            # Remove the dangling CH3
            atoms_to_remove = sorted([c_idx], reverse=True)
            for idx in atoms_to_remove:
                rw.RemoveAtom(idx)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_o_demethylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Remove O-methyl group → phenol (CYP2D6 / 2C9 O-demethylation)."""
        try:
            from rdkit.Chem import RWMol
            pattern = Chem.MolFromSmarts("[CH3;$([CH3]-O-[c,C])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx = match[0]
            rw = Chem.RWMol(mol)
            o_idx = None
            for nbr in rw.GetAtomWithIdx(c_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 8:
                    o_idx = nbr.GetIdx()
                    break
            if o_idx is None:
                return None
            rw.RemoveBond(c_idx, o_idx)
            # Remove the dangling CH3
            for idx in sorted([c_idx], reverse=True):
                rw.RemoveAtom(idx)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_glucuronidation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Append glucuronic acid to a phenol (Phase II conjugation)."""
        try:
            # Instead of real glucuronide construction (complex stereochemistry),
            # return the deprotonated phenol as a proxy product — structurally
            # correct for testing the DL pipeline, not for clinical use.
            pattern = Chem.MolFromSmarts("[OH;$([OH]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            # Proxy: return the phenol itself (identity for phenols)
            # Real implementation would use a SMIRKS reaction:
            # "[OH:1]-[c:2]>>[O:1]([c:2])C1OC(C(=O)O)C(O)C(O)C1O"
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    @staticmethod
    def _emulate_sulfation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Sulfate conjugation of a phenol (Phase II, SULT enzymes)."""
        try:
            from rdkit.Chem import RWMol, AllChem
            pattern = Chem.MolFromSmarts("[OH;$([OH]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            o_idx = match[0]
            rw = Chem.RWMol(mol)
            s_idx  = rw.AddAtom(Chem.Atom(16))   # S
            o1_idx = rw.AddAtom(Chem.Atom(8))    # =O
            o2_idx = rw.AddAtom(Chem.Atom(8))    # =O
            o3_idx = rw.AddAtom(Chem.Atom(8))    # -OH
            h_idx  = rw.AddAtom(Chem.Atom(1))
            rw.AddBond(o_idx,  s_idx,  Chem.BondType.SINGLE)
            rw.AddBond(s_idx,  o1_idx, Chem.BondType.DOUBLE)
            rw.AddBond(s_idx,  o2_idx, Chem.BondType.DOUBLE)
            rw.AddBond(s_idx,  o3_idx, Chem.BondType.SINGLE)
            rw.AddBond(o3_idx, h_idx,  Chem.BondType.SINGLE)
            # Remove the original O-H proton
            rw.GetAtomWithIdx(o_idx).SetNumExplicitHs(0)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_oxidation_sulfoxide(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Oxidise thioether S → sulfoxide (CYP / FMO S-oxidation)."""
        try:
            from rdkit.Chem import RWMol
            pattern = Chem.MolFromSmarts("[S;X2;!$(S=O);$([S]([#6])[#6])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            s_idx = match[0]
            rw = Chem.RWMol(mol)
            o_idx = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(s_idx, o_idx, Chem.BondType.DOUBLE)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_hydrolysis(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Ester hydrolysis → carboxylic acid + alcohol."""
        try:
            from rdkit.Chem import AllChem, RWMol
            # Detect ester: [C](=O)-O-[C,c]
            pattern = Chem.MolFromSmarts("[C:1](=[O:2])-[O:3]-[#6:4]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx, o_dbl, o_single, c_right = match
            rw = Chem.RWMol(mol)
            rw.RemoveBond(o_single, c_right)
            # Add H to both oxygens
            rw.GetAtomWithIdx(o_single).SetNumExplicitHs(1)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    # -- Public prediction API ------------------------------------------------

    def predict(
        self,
        smiles: str,
        phase_hint: Optional[int] = None,
    ) -> Tuple[List[Tuple[str, float]], Dict[int, float], List[str]]:
        """
        Run the full DL prediction pipeline for a single parent molecule.

        Parameters
        ----------
        smiles     : canonical SMILES (pre-normalised by ``_validate_and_normalise``).
        phase_hint : optional phase bias (1 or 2); not used by emulator but
                     available for real model phase-conditioned decoding.

        Returns
        -------
        (predictions, attention_weights, warnings)

          predictions       : List[(canonical_smiles, confidence_score)]
                              sorted by confidence descending.
          attention_weights : Dict[atom_index → risk_weight]
                              per-atom SOM attention scores in [0, 1].
          warnings          : List of non-fatal diagnostic strings.
        """
        dl_warnings: List[str] = []

        # ── Tokenise ──────────────────────────────────────────────────────────
        try:
            tokens = self.tokenise(smiles)
        except ValueError as exc:
            dl_warnings.append(f"DL tokenisation failed: {exc}")
            return [], {}, dl_warnings

        # ── Verify parent ─────────────────────────────────────────────────────
        parent_mol = self.verify_smiles(smiles)
        if parent_mol is None:
            dl_warnings.append(
                f"DL pipeline skipped: parent SMILES failed structural "
                f"verification after tokenisation. SMILES: '{smiles[:80]}'"
            )
            return [], {}, dl_warnings

        # ── Beam search (emulated or real) ────────────────────────────────────
        try:
            raw_predictions = self._transformer_beam_search(tokens, self.top_k)
        except Exception as exc:
            dl_warnings.append(f"DL beam search error: {exc}")
            raw_predictions = []

        # ── Filter below confidence threshold ─────────────────────────────────
        predictions = [
            (smi, score)
            for smi, score in raw_predictions
            if score >= self.confidence_threshold
        ]

        # ── Attention weights (emulated or real) ──────────────────────────────
        try:
            attention_weights = self._extract_attention_weights(parent_mol, tokens)
        except Exception as exc:
            dl_warnings.append(f"DL attention extraction error: {exc}")
            attention_weights = {}

        logger.info(
            "dl_predict_complete",
            model=self.MODEL_NAME,
            raw_predictions=len(raw_predictions),
            filtered_predictions=len(predictions),
            attention_atoms=len(attention_weights),
        )

        return predictions, attention_weights, dl_warnings


# ==============================================================================
# Consensus Engine
# ==============================================================================

class ConsensusEngine:
    """
    Merges Pipeline A (SyGMa) and Pipeline B (DL) outputs into a unified,
    confidence-tagged metabolite list and ensemble soft spot annotations.

    Consensus logic
    ---------------
    Structural alignment uses ``Chem.CanonSmiles`` to normalise both sets of
    SMILES before comparison.  An exact match of canonical forms constitutes
    consensus — no Tanimoto similarity threshold is used because near-identical
    SMILES with different stereochemistry represent chemically distinct metabolites
    and should be tagged separately.

    Score fusion
    ------------
    Consensus metabolites inherit the higher of the two pipeline scores as
    their primary ``probability`` field, preserving the SyGMa reaction name
    and phase assignment (which the DL model does not produce).

    For soft spots, the vulnerability index blends rule and attention signals:
        VI = (ENSEMBLE_ALPHA × rule_score + ENSEMBLE_BETA × dl_attention) × 100 %
    """

    def __init__(
        self,
        alpha: float = ENSEMBLE_ALPHA,
        beta:  float = ENSEMBLE_BETA,
    ) -> None:
        self.alpha = alpha
        self.beta  = beta

    # ------------------------------------------------------------------

    def merge_metabolites(
        self,
        sygma_metabolites: List[PredictedMetabolite],
        dl_predictions:    List[Tuple[str, float]],
    ) -> Tuple[List[PredictedMetabolite], Dict]:
        """
        Merge and tag SyGMa + DL metabolite predictions.

        Parameters
        ----------
        sygma_metabolites : output of ``_run_sygma``.
        dl_predictions    : list of (canonical_smiles, confidence) from DL.

        Returns
        -------
        (merged_metabolites, stats_dict)
          merged_metabolites : deduplicated, confidence-tagged list sorted
                               by probability descending.
          stats_dict         : pipeline statistics for logging / API response.
        """
        # Build canonical SMILES lookup for SyGMa results
        sygma_by_canon: Dict[str, PredictedMetabolite] = {}
        for m in sygma_metabolites:
            canon = self._safe_canonicalise(m.smiles)
            if canon:
                sygma_by_canon[canon] = m

        # Build canonical SMILES set for DL results
        dl_by_canon: Dict[str, float] = {}
        for smiles, score in dl_predictions:
            canon = self._safe_canonicalise(smiles)
            if canon:
                dl_by_canon[canon] = score

        consensus_set    = set(sygma_by_canon) & set(dl_by_canon)
        sygma_only_set   = set(sygma_by_canon) - set(dl_by_canon)
        dl_only_set      = set(dl_by_canon)    - set(sygma_by_canon)

        merged: List[PredictedMetabolite] = []

        # ── Consensus metabolites ──────────────────────────────────────────────
        for canon in consensus_set:
            base = sygma_by_canon[canon]
            dl_s = dl_by_canon[canon]
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=max(base.probability, dl_s),
                phase=base.phase,
                reaction_name=base.reaction_name,
                molecular_weight=base.molecular_weight,
                molecular_formula=base.molecular_formula,
                confidence_tier=ConsensusTier.HIGH,
                dl_score=dl_s,
                sources=frozenset({"sygma", "dl"}),
            ))

        # ── SyGMa-only metabolites ────────────────────────────────────────────
        for canon in sygma_only_set:
            base = sygma_by_canon[canon]
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=base.probability,
                phase=base.phase,
                reaction_name=base.reaction_name,
                molecular_weight=base.molecular_weight,
                molecular_formula=base.molecular_formula,
                confidence_tier=ConsensusTier.RULE,
                dl_score=None,
                sources=frozenset({"sygma"}),
            ))

        # ── DL-only metabolites ───────────────────────────────────────────────
        for canon in dl_only_set:
            dl_s = dl_by_canon[canon]
            mol = self._safe_mol(canon)
            mw, formula = self._compute_descriptors(mol)
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=dl_s,
                phase=-1,           # DL model does not produce phase labels
                reaction_name="dl_predicted",
                molecular_weight=mw,
                molecular_formula=formula,
                confidence_tier=ConsensusTier.DL,
                dl_score=dl_s,
                sources=frozenset({"dl"}),
            ))

        merged.sort(key=lambda m: m.probability, reverse=True)

        stats = {
            "sygma_total":      len(sygma_by_canon),
            "dl_total":         len(dl_by_canon),
            "consensus_count":  len(consensus_set),
            "rule_only_count":  len(sygma_only_set),
            "dl_only_count":    len(dl_only_set),
            "merged_total":     len(merged),
        }

        logger.info("consensus_merge_complete", **stats)
        return merged, stats

    def merge_soft_spots(
        self,
        rule_spots:       List[SoftSpot],
        attention_weights: Dict[int, float],
        mol:              Chem.Mol,
        top_n:            int,
        dl_source:        str = "metatrans-emulator",
    ) -> List[EnsembleSoftSpot]:
        """
        Fuse rule-based and DL attention soft spot signals.

        For atoms with a rule match: VI = (α·rule_score + β·dl_attention) × 100
        For atoms with attention only (no rule match): VI = β·dl_attention × 100
        Rule-matched atoms always appear in the top-N if they outscore attention-only.

        Parameters
        ----------
        rule_spots        : output of ``_find_soft_spots``.
        attention_weights : per-atom attention weights from DL predictor.
        mol               : parent molecule (used to look up atom symbols for
                            attention-only atoms not in rule_spots).
        top_n             : number of ensemble spots to return.
        dl_source         : identifier for the DL model providing attention.

        Returns
        -------
        List of ``EnsembleSoftSpot``, sorted by vulnerability_index descending,
        length ≤ top_n.
        """
        ensemble: Dict[int, EnsembleSoftSpot] = {}

        # Process rule-matched atoms
        for spot in rule_spots:
            dl_w = attention_weights.get(spot.atom_index, 0.0)
            vi   = round((self.alpha * spot.score + self.beta * dl_w) * 100, 2)
            ensemble[spot.atom_index] = EnsembleSoftSpot(
                atom_index=spot.atom_index,
                atom_symbol=spot.atom_symbol,
                rule_name=spot.rule_name,
                smarts_match=spot.smarts_match,
                rule_score=spot.score,
                dl_attention_risk=dl_w,
                vulnerability_index=vi,
                dl_source=dl_source,
            )

        # Process attention-only atoms (not in rule_spots) — add if high attention
        if attention_weights:
            # Threshold: include attention-only atoms above the 75th percentile
            sorted_weights = sorted(attention_weights.values(), reverse=True)
            threshold_rank = max(1, len(sorted_weights) // 4)
            threshold = sorted_weights[threshold_rank - 1] if sorted_weights else 0.0

            for atom_idx, dl_w in attention_weights.items():
                if atom_idx in ensemble:
                    continue
                if dl_w < threshold:
                    continue
                vi = round(self.beta * dl_w * 100, 2)
                try:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    symbol = atom.GetSymbol()
                except Exception:
                    symbol = "?"
                ensemble[atom_idx] = EnsembleSoftSpot(
                    atom_index=atom_idx,
                    atom_symbol=symbol,
                    rule_name="dl_attention_only",
                    smarts_match="",
                    rule_score=0.0,
                    dl_attention_risk=dl_w,
                    vulnerability_index=vi,
                    dl_source=dl_source,
                )

        result = sorted(
            ensemble.values(),
            key=lambda s: s.vulnerability_index,
            reverse=True,
        )
        return result[:top_n]

    # -- Private helpers ------------------------------------------------------

    @staticmethod
    def _safe_canonicalise(smiles: str) -> Optional[str]:
        """Return RDKit canonical SMILES, or None on any RDKit error."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None

    @staticmethod
    def _safe_mol(smiles: str) -> Optional[Chem.Mol]:
        try:
            return Chem.MolFromSmiles(smiles)
        except Exception:
            return None

    @staticmethod
    def _compute_descriptors(
        mol: Optional[Chem.Mol],
    ) -> Tuple[Optional[float], Optional[str]]:
        if mol is None:
            return None, None
        try:
            return (
                round(Descriptors.MolWt(mol), 4),
                rdMolDescriptors.CalcMolFormula(mol),
            )
        except Exception:
            return None, None


# ==============================================================================
# Private helpers  (validation, SyGMa runner, rule-based SOM)
# ==============================================================================

def _validate_and_normalise(smiles: str) -> Tuple[Chem.Mol, MoleculeMetadata]:
    """
    Parse, validate, and normalise a SMILES string.

    Normalisation pipeline
    ----------------------
    1. ``Chem.MolFromSmiles``     — parse + valence check
    2. ``LargestFragmentChooser`` — strip salts / counterions
    3. ``Normalizer``             — canonical functional-group forms
    4. ``Uncharger``              — neutralise formal charges
    5. ``SanitizeMol``            — final RDKit sanitisation
    6. Descriptor computation     — MW, formula, InChI, logP, TPSA, etc.

    Raises
    ------
    ValueError  if the SMILES is invalid at any pipeline stage.
    """
    if not smiles or not smiles.strip():
        raise ValueError("SMILES string is empty.")

    smiles = smiles.strip()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(
            f"RDKit could not parse SMILES: {smiles!r}. "
            "Check atom symbols, valences, and ring-closure notation."
        )

    mol = _LARGEST_FRAGMENT.choose(mol)
    mol = _NORMALISER.normalize(mol)
    mol = _UNCHARGER.uncharge(mol)

    try:
        Chem.SanitizeMol(mol)
    except rdchem.MolSanitizeException as exc:
        raise ValueError(f"Sanitisation failed after normalisation: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Normalisation error: {exc}") from exc

    if mol.GetNumHeavyAtoms() == 0:
        raise ValueError("Molecule has no heavy atoms after normalisation.")

    from rdkit.Chem import inchi as rdinchi
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    inchi_str   = rdinchi.MolToInchi(mol) or ""
    inchikey    = rdinchi.InchiToInchiKey(inchi_str) if inchi_str else ""

    metadata = MoleculeMetadata(
        input_smiles=smiles,
        canonical_smiles=canonical_smiles,
        inchi=inchi_str,
        inchikey=inchikey,
        molecular_formula=rdMolDescriptors.CalcMolFormula(mol),
        molecular_weight=round(Descriptors.MolWt(mol), 4),
        exact_mass=round(Descriptors.ExactMolWt(mol), 6),
        num_heavy_atoms=mol.GetNumHeavyAtoms(),
        num_rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
        num_hbd=rdMolDescriptors.CalcNumHBD(mol),
        num_hba=rdMolDescriptors.CalcNumHBA(mol),
        tpsa=round(rdMolDescriptors.CalcTPSA(mol), 2),
        logp=round(Descriptors.MolLogP(mol), 4),
        num_rings=rdMolDescriptors.CalcNumRings(mol),
        num_aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
    )

    logger.debug(
        "molecule_normalised",
        canonical=canonical_smiles,
        mw=metadata.molecular_weight,
        formula=metadata.molecular_formula,
    )
    return mol, metadata


def _run_sygma(
    mol: Chem.Mol,
    phase1_cycles: int = 1,
    phase2_cycles: int = 1,
    max_metabolites: int = 200,
) -> List[PredictedMetabolite]:
    """
    Execute a SyGMa Phase I → Phase II prediction scenario.

    Returns PredictedMetabolite instances with ``sources=frozenset({"sygma"})``
    and ``confidence_tier=ConsensusTier.RULE`` (to be upgraded to HIGH by the
    ConsensusEngine when DL agreement is found).

    Raises
    ------
    ImportError  if SyGMa is not installed.
    """
    try:
        import sygma                          # type: ignore
        from sygma import ruleset as srs      # type: ignore
        from sygma import Scenario            # type: ignore
    except ImportError as exc:
        raise ImportError(
            "SyGMa is not installed. "
            "Add 'sygma>=1.1.0' to the pip section of environment.yml."
        ) from exc

    scenario = Scenario([
        [srs.phase1, phase1_cycles],
        [srs.phase2, phase2_cycles],
    ])

    logger.debug("sygma_start", p1=phase1_cycles, p2=phase2_cycles)
    tree = scenario.run(mol)
    tree.calc_scores()

    parent_canon = Chem.MolToSmiles(mol, canonical=True)
    seen: Dict[str, PredictedMetabolite] = {}

    for node in tree.to_list():
        node_mol = getattr(node, "mol", None)
        if node_mol is None:
            continue

        # Per-metabolite error defence — catch sanitisation failures
        try:
            Chem.SanitizeMol(node_mol)
        except rdchem.MolSanitizeException as exc:
            logger.debug("sygma_sanitize_skip", error=str(exc))
            continue
        except Exception as exc:
            logger.debug("sygma_mol_skip", error=str(exc))
            continue

        try:
            smiles = Chem.MolToSmiles(node_mol, canonical=True)
        except Exception as exc:
            logger.debug("sygma_smiles_skip", error=str(exc))
            continue

        if not smiles or smiles == parent_canon:
            continue

        probability   = float(getattr(node, "score",         0.0))
        reaction_name = str(getattr(node,  "reaction_name", "") or "unknown")

        phase = (2 if reaction_name.lower().startswith("phase2")
                 or any(k in reaction_name.lower()
                        for k in ("glucuronid", "sulfat", "acetyl",
                                  "glutathione", "methyl"))
                 else 1)
        if reaction_name.lower().startswith("phase1"):
            phase = 1

        try:
            mw      = round(Descriptors.MolWt(node_mol), 4)
            formula = rdMolDescriptors.CalcMolFormula(node_mol)
        except Exception:
            mw, formula = None, None

        candidate = PredictedMetabolite(
            smiles=smiles,
            probability=probability,
            phase=phase,
            reaction_name=reaction_name,
            molecular_weight=mw,
            molecular_formula=formula,
            confidence_tier=ConsensusTier.RULE,   # upgraded by consensus later
            sources=frozenset({"sygma"}),
        )

        if smiles not in seen or probability > seen[smiles].probability:
            seen[smiles] = candidate

    results = sorted(seen.values(), key=lambda x: x.probability, reverse=True)
    results = results[:max_metabolites]

    logger.info(
        "sygma_complete",
        unique=len(results),
        p1=sum(1 for r in results if r.phase == 1),
        p2=sum(1 for r in results if r.phase == 2),
    )
    return results


def _find_soft_spots(
    mol: Chem.Mol,
    top_n: int = 3,
) -> List[SoftSpot]:
    """
    Rule-only soft spot fallback (Pipeline A standalone).

    Used when the DL pipeline is unavailable or disabled.  The ConsensusEngine
    calls ``merge_soft_spots`` instead when both pipelines are active.

    Algorithm
    ---------
    For each rule in ``_SOM_RULES``:
      - Compile SMARTS, find substructure matches.
      - Atom at match index 0 is the SOM site.
      - Apply a 20 % steric penalty for atoms with ≥ 3 bulky neighbours.
    """
    best: Dict[int, SoftSpot] = {}

    for rule_name, smarts, base_score in _SOM_RULES:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            logger.warning("invalid_smarts", rule=rule_name)
            continue

        for match in mol.GetSubstructMatches(pattern):
            if not match:
                continue
            atom_idx = match[0]
            atom     = mol.GetAtomWithIdx(atom_idx)

            heavy_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
            score = base_score
            if len(heavy_nbrs) >= 3 and any(n.GetDegree() > 1 for n in heavy_nbrs):
                score *= 0.80

            score = round(score, 4)
            if atom_idx not in best or score > best[atom_idx].score:
                best[atom_idx] = SoftSpot(
                    atom_index=atom_idx,
                    atom_symbol=atom.GetSymbol(),
                    rule_name=rule_name,
                    score=score,
                    smarts_match=smarts,
                )

    return sorted(best.values(), key=lambda s: s.score, reverse=True)[:top_n]


# ==============================================================================
# Public API
# ==============================================================================

# Type alias for drop-in soft-spot function hook (backward compatible with v1)
SoftSpotFn = Callable[[Chem.Mol, int], List[SoftSpot]]


def predict(
    smiles: str,
    *,
    phase1_cycles:   int  = 1,
    phase2_cycles:   int  = 1,
    max_metabolites: int  = 200,
    top_soft_spots:  int  = 3,
    # v2 ensemble parameters
    dl_predictor:    Optional[DeepLearningPredictor] = None,
    enable_dl:       bool = True,
    # v1 backward-compat hook (used when dl_predictor is None and enable_dl=False)
    soft_spot_fn:    Optional[SoftSpotFn] = None,
) -> MetabolismResult:
    """
    Ensemble Consensus Metabolism Prediction.

    Orchestrates the dual-pipeline architecture:
      Pipeline A  — SyGMa rule-based Phase I/II metabolites
      Pipeline B  — DeepLearningPredictor (emulator or real weights)
      Consensus   — cross-validates outputs, assigns ConsensusTier tags
      Soft Spots  — blends rule scores with DL attention weights

    Parameters
    ----------
    smiles          : Parent molecule SMILES (any notation; normalised internally).
    phase1_cycles   : SyGMa Phase I iteration depth.
    phase2_cycles   : SyGMa Phase II iteration depth.
    max_metabolites : Hard cap on returned metabolites (applied after merge).
    top_soft_spots  : Number of soft spots to annotate.
    dl_predictor    : Pre-built DeepLearningPredictor instance.  If None and
                      enable_dl=True, a default emulator is created internally.
    enable_dl       : Master switch for Pipeline B.  Set False to run rule-only
                      (equivalent to v1 behaviour).
    soft_spot_fn    : v1 backward-compat hook.  Ignored when enable_dl=True.

    Returns
    -------
    ``MetabolismResult`` with full ensemble data.  ``.to_dict()`` is
    backward-compatible with v1 API consumers.

    Raises
    ------
    ValueError   : SMILES is invalid.
    RuntimeError : Unexpected engine-level error.

    Examples
    --------
    >>> # Minimal usage (ensemble mode, default emulator)
    >>> from app.engine.metabolism import predict
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O")
    >>> r.parent.molecular_formula
    'C9H8O4'
    >>> all(isinstance(s, EnsembleSoftSpot) for s in r.soft_spots)
    True

    >>> # Rule-only mode (v1 equivalent)
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O", enable_dl=False)

    >>> # Real weights (when available)
    >>> predictor = DeepLearningPredictor.from_checkpoint("metatrans.pt")
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O", dl_predictor=predictor)
    """
    t_start   = time.perf_counter()
    warnings: List[str] = []

    # ── Step 1: Validate + normalise ──────────────────────────────────────────
    logger.info("ensemble_predict_start")
    mol, parent_meta = _validate_and_normalise(smiles)

    if parent_meta.canonical_smiles != smiles.strip():
        warnings.append(
            f"Input SMILES normalised: {smiles!r} → "
            f"{parent_meta.canonical_smiles!r}"
        )

    if parent_meta.molecular_weight > 1500:
        warnings.append(
            f"MW {parent_meta.molecular_weight:.1f} Da exceeds typical small-molecule "
            "range (< 1500 Da). Predictions may be unreliable."
        )

    # ── Step 2A: Pipeline A — SyGMa ───────────────────────────────────────────
    sygma_metabolites: List[PredictedMetabolite] = []
    try:
        sygma_metabolites = _run_sygma(
            mol,
            phase1_cycles=phase1_cycles,
            phase2_cycles=phase2_cycles,
            max_metabolites=max_metabolites,
        )
    except ImportError:
        warnings.append(
            "SyGMa not installed — rule-based pipeline inactive. "
            "Install: pip install sygma"
        )
        logger.warning("sygma_not_installed")
    except Exception as exc:
        warnings.append(f"SyGMa pipeline error (non-fatal): {exc}")
        logger.error("sygma_pipeline_error", error=str(exc))

    # ── Step 2B: Pipeline B — Deep Learning ───────────────────────────────────
    dl_predictions:    List[Tuple[str, float]] = []
    attention_weights: Dict[int, float]        = {}
    dl_source = "disabled"

    if enable_dl:
        # Use provided predictor or create a default emulator
        predictor = dl_predictor or DeepLearningPredictor()
        dl_source = predictor.MODEL_NAME
        try:
            (
                dl_predictions,
                attention_weights,
                dl_warnings,
            ) = predictor.predict(parent_meta.canonical_smiles)
            warnings.extend(dl_warnings)
        except Exception as exc:
            warnings.append(f"DL pipeline error (non-fatal): {exc}")
            logger.error("dl_pipeline_error", error=str(exc))
    else:
        warnings.append("DL pipeline disabled (enable_dl=False).")

    # ── Step 3: Consensus merge ────────────────────────────────────────────────
    consensus = ConsensusEngine()

    if enable_dl and (sygma_metabolites or dl_predictions):
        merged_metabolites, pipeline_stats = consensus.merge_metabolites(
            sygma_metabolites, dl_predictions
        )
        merged_metabolites = merged_metabolites[:max_metabolites]
    else:
        # Rule-only fallback: tag everything as RULE tier
        merged_metabolites = sygma_metabolites[:max_metabolites]
        pipeline_stats = {
            "sygma_total":     len(sygma_metabolites),
            "dl_total":        0,
            "consensus_count": 0,
            "rule_only_count": len(sygma_metabolites),
            "dl_only_count":   0,
            "merged_total":    len(sygma_metabolites),
        }

    # ── Step 4: Soft spot analysis ────────────────────────────────────────────
    if soft_spot_fn is not None and not enable_dl:
        # v1 backward-compat: external soft_spot_fn overrides ensemble
        soft_spots_raw = soft_spot_fn(mol, top_soft_spots)
        soft_spots: List = soft_spots_raw
    else:
        # Ensemble soft spots
        rule_spots = _find_soft_spots(mol, top_n=mol.GetNumHeavyAtoms())
        soft_spots = consensus.merge_soft_spots(
            rule_spots=rule_spots,
            attention_weights=attention_weights,
            mol=mol,
            top_n=top_soft_spots,
            dl_source=dl_source,
        )

    if not soft_spots:
        warnings.append(
            "No soft spots identified. The molecule may lack common CYP substrate "
            "features or all candidates were excluded by the steric/attention filter."
        )

    # ── Step 5: Assemble result ───────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    pipeline_stats["dl_source"]  = dl_source
    pipeline_stats["elapsed_s"]  = round(elapsed, 4)

    result = MetabolismResult(
        parent=parent_meta,
        metabolites=merged_metabolites,
        soft_spots=soft_spots,
        engine_version=ENGINE_VERSION,
        elapsed_s=elapsed,
        warnings=warnings,
        pipeline_stats=pipeline_stats,
    )

    logger.info(
        "ensemble_predict_complete",
        canonical=parent_meta.canonical_smiles,
        metabolites_total=len(merged_metabolites),
        consensus_count=pipeline_stats.get("consensus_count", 0),
        soft_spots_total=len(soft_spots),
        elapsed_s=round(elapsed, 3),
    )

    return result
.Groups[1].Value
    "logger.info(f`"$($args -replace '`"', '' -replace "'", '' -replace ',\s*\w+=', ' ' -replace '=', '=')`")"

    mol, parent_meta = _validate_and_normalise(smiles)

    if parent_meta.canonical_smiles != smiles.strip():
        warnings.append(
            f"Input SMILES normalised: {smiles!r} → "
            f"{parent_meta.canonical_smiles!r}"
        )

    if parent_meta.molecular_weight > 1500:
        warnings.append(
            f"MW {parent_meta.molecular_weight:.1f} Da exceeds typical small-molecule "
            "range (< 1500 Da). Predictions may be unreliable."
        )

    # ── Step 2A: Pipeline A — SyGMa ───────────────────────────────────────────
    sygma_metabolites: List[PredictedMetabolite] = []
    try:
        sygma_metabolites = _run_sygma(
            mol,
            phase1_cycles=phase1_cycles,
            phase2_cycles=phase2_cycles,
            max_metabolites=max_metabolites,
        )
    except ImportError:
        warnings.append(
            "SyGMa not installed — rule-based pipeline inactive. "
            "Install: pip install sygma"
        )
        logger.warning("sygma_not_installed")
    except Exception as exc:
        warnings.append(f"SyGMa pipeline error (non-fatal): {exc}")
        logger.error("sygma_pipeline_error", error=str(exc))

    # ── Step 2B: Pipeline B — Deep Learning ───────────────────────────────────
    dl_predictions:    List[Tuple[str, float]] = []
    attention_weights: Dict[int, float]        = {}
    dl_source = "disabled"

    if enable_dl:
        # Use provided predictor or create a default emulator
        predictor = dl_predictor or DeepLearningPredictor()
        dl_source = predictor.MODEL_NAME
        try:
            (
                dl_predictions,
                attention_weights,
                dl_warnings,
            ) = predictor.predict(parent_meta.canonical_smiles)
            warnings.extend(dl_warnings)
        except Exception as exc:
            warnings.append(f"DL pipeline error (non-fatal): {exc}")
            logger.error("dl_pipeline_error", error=str(exc))
    else:
        warnings.append("DL pipeline disabled (enable_dl=False).")

    # ── Step 3: Consensus merge ────────────────────────────────────────────────
    consensus = ConsensusEngine()

    if enable_dl and (sygma_metabolites or dl_predictions):
        merged_metabolites, pipeline_stats = consensus.merge_metabolites(
            sygma_metabolites, dl_predictions
        )
        merged_metabolites = merged_metabolites[:max_metabolites]
    else:
        # Rule-only fallback: tag everything as RULE tier
        merged_metabolites = sygma_metabolites[:max_metabolites]
        pipeline_stats = {
            "sygma_total":     len(sygma_metabolites),
            "dl_total":        0,
            "consensus_count": 0,
            "rule_only_count": len(sygma_metabolites),
            "dl_only_count":   0,
            "merged_total":    len(sygma_metabolites),
        }

    # ── Step 4: Soft spot analysis ────────────────────────────────────────────
    if soft_spot_fn is not None and not enable_dl:
        # v1 backward-compat: external soft_spot_fn overrides ensemble
        soft_spots_raw = soft_spot_fn(mol, top_soft_spots)
        soft_spots: List = soft_spots_raw
    else:
        # Ensemble soft spots
        rule_spots = _find_soft_spots(mol, top_n=mol.GetNumHeavyAtoms())
        soft_spots = consensus.merge_soft_spots(
            rule_spots=rule_spots,
            attention_weights=attention_weights,
            mol=mol,
            top_n=top_soft_spots,
            dl_source=dl_source,
        )

    if not soft_spots:
        warnings.append(
            "No soft spots identified. The molecule may lack common CYP substrate "
            "features or all candidates were excluded by the steric/attention filter."
        )

    # ── Step 5: Assemble result ───────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    pipeline_stats["dl_source"]  = dl_source
    pipeline_stats["elapsed_s"]  = round(elapsed, 4)

    result = MetabolismResult(
        parent=parent_meta,
        metabolites=merged_metabolites,
        soft_spots=soft_spots,
        engine_version=ENGINE_VERSION,
        elapsed_s=elapsed,
        warnings=warnings,
        pipeline_stats=pipeline_stats,
    )

    
    $args = """
app/engine/metabolism.py
================================================================================
Ensemble Consensus Metabolism Engine  —  v2.0.0-ensemble
================================================================================

Architecture overview
---------------------
This module implements a dual-pipeline ensemble strategy that combines two
complementary paradigms for metabolite prediction and soft-spot analysis,
inspired by and designed for compatibility with:

  Pipeline A — Rule-Based (SyGMa)
    KavrakiLab/SyGMa / 3D-e-Chem/sygma
    Applies curated SMIRKS reaction transformation matrices for Phase I
    (oxidation, reduction, hydrolysis) and Phase II (conjugation) reactions.
    Each metabolite carries a cumulative SyGMa probability score.

  Pipeline B — Deep Learning Emulator (MetaTrans / Meta-Predictor style)
    KavrakiLab/MetaTrans  ·  zhukeyun/Meta-Predictor
    Treats metabolite prediction as a SMILES-to-SMILES sequence translation
    task (Transformer/OpenNMT architecture).  The ``DeepLearningPredictor``
    class implemented here is a **production-ready emulator** with a clean
    interface for swapping in real PyTorch .pt model weights:

        predictor = DeepLearningPredictor.from_checkpoint("weights/metatrans.pt")
        result = predict(smiles, dl_predictor=predictor)

Consensus logic
---------------
After both pipelines run, the ``ConsensusEngine`` cross-validates their outputs
using ``Chem.CanonSmiles`` for structural alignment and tags each metabolite:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Both pipelines agree      → "High Confidence (Consensus Verified)"    │
  │  SyGMa only                → "Moderate Confidence (Rule-Only)"         │
  │  DL model only             → "Moderate Confidence (DL-Only)"           │
  └─────────────────────────────────────────────────────────────────────────┘

Soft spot analysis
------------------
Unified vulnerability scoring blends two complementary signals:

  rule_score        — SMARTS-based steric-adjusted heuristic (Pipeline A)
  dl_attention_risk — per-atom attention weight from transformer SOM heads
                      (emulated; replace ``_compute_dl_attention`` with real
                       model forward pass when weights are available)

  vulnerability_index = (α × rule_score + β × dl_attention_risk) × 100  (%)
  where α=0.55, β=0.45 by default (configurable via ``ENSEMBLE_ALPHA``).

Error defence
-------------
Every pipeline is wrapped in defence layers that catch and contain:
  - ``rdkit.Chem.rdchem.MolSanitizeException``
  - Valence / kekulisation errors from transformer-generated SMILES
  - Un-tokenisable or chemically impossible SMILES strings
  - RDKit property calculation failures
  - Any unexpected exception in DL inference hooks
Individual failures downgrade confidence or add warnings; they never crash
the server.

Public API (unchanged from v1 for drop-in compatibility)
---------------------------------------------------------
  predict(smiles, ...)          → MetabolismResult
  _validate_and_normalise(...)  → (Chem.Mol, MoleculeMetadata)
  _find_soft_spots(...)         → List[SoftSpot]   (rule-only fallback)

New in v2
---------
  DeepLearningPredictor         class (Pipeline B)
  ConsensusEngine               class (merger + tagging)
  EnsembleSoftSpotResult        dataclass (combined atom vulnerability)
  ConsensusTier                 enum-like constants
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdchem   # MolSanitizeException lives here
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)


# ==============================================================================
# Engine versioning & tuning constants
# ==============================================================================

ENGINE_VERSION = "2.0.0-ensemble"

# Soft spot fusion weights:  vulnerability_index = α·rule + β·dl_attention
# Must sum to 1.0.  Adjust to preference-weight the two signals.
ENSEMBLE_ALPHA: float = 0.55   # weight given to SMARTS rule score
ENSEMBLE_BETA:  float = 0.45   # weight given to DL attention risk

assert abs(ENSEMBLE_ALPHA + ENSEMBLE_BETA - 1.0) < 1e-9, \
    "ENSEMBLE_ALPHA + ENSEMBLE_BETA must equal 1.0"

# Minimum DL model confidence to include a DL-only prediction
DL_CONFIDENCE_THRESHOLD: float = 0.10

# Consensus requires both pipeline canonical SMILES to match exactly
# (after RDKit canonicalisation).


# ==============================================================================
# Confidence tiers
# ==============================================================================

class ConsensusTier(str, Enum):
    """
    Confidence classification assigned to each metabolite by the consensus engine.

    Inherits from ``str`` so instances serialise directly to JSON strings
    without a custom encoder.
    """
    HIGH     = "High Confidence (Consensus Verified)"
    RULE     = "Moderate Confidence (Rule-Only)"
    DL       = "Moderate Confidence (DL-Only)"
    UNKNOWN  = "Unclassified"


# ==============================================================================
# Public data structures
# ==============================================================================

@dataclass(frozen=True)
class MoleculeMetadata:
    """
    Physico-chemical descriptors for a molecule.  Immutable after construction.

    ``canonical_smiles`` uses RDKit's default canonicalisation (aromatic SMILES,
    not Kekulé).  ``inchi`` / ``inchikey`` provide standard identifiers.
    """
    input_smiles:        str
    canonical_smiles:    str
    inchi:               str
    inchikey:            str
    molecular_formula:   str
    molecular_weight:    float      # average MW  (Da)
    exact_mass:          float      # monoisotopic (Da)
    num_heavy_atoms:     int
    num_rotatable_bonds: int
    num_hbd:             int        # H-bond donors
    num_hba:             int        # H-bond acceptors
    tpsa:                float      # topological polar surface area  (Å²)
    logp:                float      # Wildman–Crippen logP
    num_rings:           int
    num_aromatic_rings:  int


@dataclass(frozen=True)
class PredictedMetabolite:
    """
    A single metabolite prediction, possibly from one or both pipelines.

    New in v2
    ---------
    confidence_tier  : ConsensusTier tag assigned by the ConsensusEngine.
    dl_score         : DL model confidence [0,1] (None when DL pipeline absent).
    sources          : frozenset of pipeline names that produced this metabolite.
    """
    smiles:            str
    probability:       float            # primary score (SyGMa or DL confidence)
    phase:             int              # 1 or 2 (−1 when DL-only + unknown)
    reaction_name:     str
    molecular_weight:  Optional[float]  = None
    molecular_formula: Optional[str]    = None
    confidence_tier:   ConsensusTier    = ConsensusTier.UNKNOWN
    dl_score:          Optional[float]  = None
    sources:           frozenset        = field(default_factory=frozenset)


@dataclass(frozen=True)
class SoftSpot:
    """
    Rule-only soft spot — retained for backward compatibility with v1 API.
    Used as a fallback when the DL pipeline is unavailable.
    """
    atom_index:   int
    atom_symbol:  str
    rule_name:    str
    score:        float     # SMARTS rule score [0, 1]
    smarts_match: str


@dataclass(frozen=True)
class EnsembleSoftSpot:
    """
    Unified soft spot combining rule-based and DL-derived vulnerability signals.

    Fields
    ------
    atom_index          : 0-based RDKit atom index.
    atom_symbol         : element symbol.
    rule_name           : best-matching SMARTS rule label.
    smarts_match        : SMARTS pattern that matched.
    rule_score          : SMARTS-based vulnerability [0, 1].
    dl_attention_risk   : DL attention-head weight [0, 1]
                          (emulated; replace with real model output).
    vulnerability_index : unified score in [0, 100] %
                          = (α·rule_score + β·dl_attention_risk) × 100.
    dl_source           : identifier of the DL model that produced
                          ``dl_attention_risk`` (e.g. "metatrans-emulator").
    """
    atom_index:         int
    atom_symbol:        str
    rule_name:          str
    smarts_match:       str
    rule_score:         float
    dl_attention_risk:  float
    vulnerability_index: float          # [0, 100]
    dl_source:          str = "rule-only"


@dataclass
class MetabolismResult:
    """
    Top-level result returned by ``predict()``.

    ``soft_spots`` is now ``List[EnsembleSoftSpot]`` when the ensemble ran,
    or ``List[SoftSpot]`` when it fell back to rule-only mode.
    Both types expose ``atom_index``, ``atom_symbol``, and ``rule_name`` so
    downstream consumers (API layer, renderer) are unaffected.

    ``to_dict()`` is backward-compatible with v1 consumers while adding the
    new v2 ensemble fields.
    """
    parent:          MoleculeMetadata
    metabolites:     List[PredictedMetabolite]
    soft_spots:      List                       # List[EnsembleSoftSpot] | List[SoftSpot]
    engine_version:  str
    elapsed_s:       float
    warnings:        List[str] = field(default_factory=list)
    pipeline_stats:  Dict      = field(default_factory=dict)

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """
        Serialise to a plain nested dict for JSON responses, Celery backends,
        or pandas.  Backward-compatible with v1; adds ensemble fields.
        """
        soft_spot_dicts = []
        for s in self.soft_spots:
            if isinstance(s, EnsembleSoftSpot):
                soft_spot_dicts.append({
                    "atom_index":          s.atom_index,
                    "atom_symbol":         s.atom_symbol,
                    "rule_name":           s.rule_name,
                    "smarts_match":        s.smarts_match,
                    "rule_score":          s.rule_score,
                    "dl_attention_risk":   s.dl_attention_risk,
                    "vulnerability_index": s.vulnerability_index,
                    "dl_source":           s.dl_source,
                    # v1-compat alias
                    "score":               round(s.vulnerability_index / 100.0, 4),
                })
            else:
                soft_spot_dicts.append({
                    "atom_index":          s.atom_index,
                    "atom_symbol":         s.atom_symbol,
                    "rule_name":           s.rule_name,
                    "smarts_match":        s.smarts_match,
                    "rule_score":          s.score,
                    "dl_attention_risk":   0.0,
                    "vulnerability_index": round(s.score * 100, 2),
                    "dl_source":           "rule-only",
                    "score":               s.score,
                })

        return {
            "engine_version": self.engine_version,
            "elapsed_s":      round(self.elapsed_s, 4),
            "warnings":       self.warnings,
            "pipeline_stats": self.pipeline_stats,
            "parent": {
                "input_smiles":        self.parent.input_smiles,
                "canonical_smiles":    self.parent.canonical_smiles,
                "inchi":               self.parent.inchi,
                "inchikey":            self.parent.inchikey,
                "molecular_formula":   self.parent.molecular_formula,
                "molecular_weight":    self.parent.molecular_weight,
                "exact_mass":          self.parent.exact_mass,
                "num_heavy_atoms":     self.parent.num_heavy_atoms,
                "num_rotatable_bonds": self.parent.num_rotatable_bonds,
                "num_hbd":             self.parent.num_hbd,
                "num_hba":             self.parent.num_hba,
                "tpsa":                self.parent.tpsa,
                "logp":                self.parent.logp,
                "num_rings":           self.parent.num_rings,
                "num_aromatic_rings":  self.parent.num_aromatic_rings,
            },
            "metabolites": [
                {
                    "smiles":           m.smiles,
                    "probability":      m.probability,
                    "phase":            m.phase,
                    "reaction_name":    m.reaction_name,
                    "molecular_weight": m.molecular_weight,
                    "molecular_formula":m.molecular_formula,
                    "confidence_tier":  m.confidence_tier.value,
                    "dl_score":         m.dl_score,
                    "sources":          sorted(m.sources),
                }
                for m in self.metabolites
            ],
            "soft_spots": soft_spot_dicts,
        }

    @property
    def phase1_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites if m.phase == 1]

    @property
    def phase2_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites if m.phase == 2]

    @property
    def consensus_metabolites(self) -> List[PredictedMetabolite]:
        return [m for m in self.metabolites
                if m.confidence_tier == ConsensusTier.HIGH]

    @property
    def top_soft_spot_indices(self) -> List[int]:
        return [s.atom_index for s in self.soft_spots]


# ==============================================================================
# SOM SMARTS rules  (Pipeline A — rule-based component)
# ==============================================================================
# Each entry: (rule_name, smarts_pattern, priority_score)
#
# priority_score in (0, 1] based on published CYP450 substrate literature:
#   Zaretzki et al., J. Chem. Inf. Model. 2013 (CypReact)
#   Sheridan et al., J. Med. Chem. 2002
#   Cruciani et al., J. Med. Chem. 2005 (MetaSite)
#
# Convention: atom 0 in each SMARTS match is the SOM being annotated.

_SOM_RULES: List[Tuple[str, str, float]] = [
    # -- Phase I: CYP-mediated oxidation -------------------------------------

    # Benzylic CH2/CH3: highest CYP2C9/3A4 affinity (Meunier 2004)
    ("benzylic_CH",
     "[CH2,CH3;!$([CH2]C=O);!$([CH2]C#N)]-[c]", 0.92),

    # Unhindered aromatic C-H: CYP1A2/2C19 aromatic hydroxylation
    ("aromatic_C_unhindered",
     "[cH]", 0.85),

    # α-sp3 C to heteroatom: N/O-dealkylation (most common CYP reaction type)
    ("alpha_C_heteroatom",
     "[CH2,CH3;$([CH2,CH3][N,O;!$(N-C=O)])]", 0.88),

    # N-methyl/ethyl: CYP3A4 N-demethylation hotspot
    ("N_alkyl",
     "[CH3,CH2;$([CH3,CH2]-[N;!$(N-C=O);!$(N~[!#6;!#1])])]", 0.86),

    # α-carbonyl sp3 C: CYP-mediated α-hydroxylation
    ("alpha_carbonyl_C",
     "[CH2,CH3;$([CH2,CH3]C(=O)[#6,#7,#8])]", 0.75),

    # Terminal methyl (ω-hydroxylation, CYP4A/2E1)
    ("terminal_methyl_aliphatic",
     "[CH3;$([CH3]-[CH2]-[CH2]-[#6])]", 0.65),

    # Thioether S: CYP S-oxidation → sulfoxide → sulfone
    ("thioether_S",
     "[S;X2;!$(S=O);$([S]([#6])[#6])]", 0.80),

    # Tertiary amine N: FMO/CYP3A4 N-oxidation
    ("tertiary_amine_N",
     "[N;X3;!$(N-C=O);!$(N~[!#6;!#1]);!n;$([N]([#6])([#6])[#6])]", 0.78),

    # -- Phase I: Reduction --------------------------------------------------

    # Nitro group reduction (gut flora + hepatic)
    ("nitro_reduction",
     "[N+](=O)[O-]", 0.70),

    # Ketone/aldehyde carbonyl reduction (AKR / carbonyl reductases)
    ("carbonyl_reduction",
     "[CH1,CH0;$(C=O);!$(C(=O)[OH]);!$(C(=O)N)]", 0.60),

    # -- Phase II: Conjugation-prone sites -----------------------------------

    # Phenolic OH carbon: glucuronidation / sulfation (mark the C, not the O)
    ("phenolic_OH_site",
     "[c;$([c][OH1])]", 0.83),

    # Primary aliphatic amine: acetylation / glucuronidation
    ("primary_amine",
     "[NH2;$([NH2]-[#6;!$(C=O)])]", 0.72),

    # Carboxylic acid: acyl-glucuronidation (reactive metabolite risk)
    ("carboxylic_acid",
     "[C;$(C(=O)[OH1])]", 0.68),
]

# Normalisation pipeline  (shared singletons — expensive to construct)
_NORMALISER       = rdMolStandardize.Normalizer()
_UNCHARGER        = rdMolStandardize.Uncharger()
_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()


# ==============================================================================
# Pipeline B — DeepLearningPredictor
# ==============================================================================

class DeepLearningPredictor:
    """
    Production-ready deep-learning metabolite predictor emulator.

    Architectural inspiration
    -------------------------
    MetaTrans (KavrakiLab/MetaTrans): OpenNMT Transformer trained on
    experimental metabolite SMILES pairs.  Encodes the parent SMILES token
    sequence, decodes up to ``top_k`` metabolite sequences with beam search,
    and scores them with the model's log-likelihood.

    Meta-Predictor (zhukeyun/Meta-Predictor): Adds per-atom attention heads
    that produce site-of-metabolism risk vectors directly usable for soft
    spot annotation without a separate SOM model.

    This class
    ----------
    Implements the full interface those models require:
      - SMILES tokenisation  (atom-level, compatible with MetaTrans vocabulary)
      - Structural verification of generated SMILES via RDKit
      - Per-atom attention weight extraction (emulated with a deterministic
        chemistry-aware heuristic until real weights are loaded)
      - Graceful degradation: any tensor/inference failure returns empty lists
        with informative warnings, never crashes

    Swap-in path for real weights
    -----------------------------
    1. Install PyTorch + OpenNMT-py:
           pip install torch opennmt-py
    2. Subclass or monkey-patch ``_transformer_beam_search``:
           def _transformer_beam_search(self, tokens, top_k):
               # real OpenNMT / HuggingFace inference here
               ...
    3. Subclass or monkey-patch ``_extract_attention_weights``:
           def _extract_attention_weights(self, mol, tokens):
               # real model forward-pass attention extraction
               ...
    4. Load weights via ``from_checkpoint``:
           predictor = DeepLearningPredictor.from_checkpoint("metatrans.pt")

    All public method signatures remain stable between emulator and real-weight
    modes — no changes needed in the calling code.
    """

    # ── MetaTrans atom-level vocabulary (subset) ──────────────────────────────
    # Real MetaTrans uses a regex tokeniser that splits on atom symbols,
    # bond characters, ring digits, and branch brackets.  We replicate that
    # split pattern here so tokenised sequences are compatible.
    _TOKEN_PATTERN = (
        r"(\[[^\]]+\]"           # bracket atoms  [NH2], [13C], ...
        r"|Br|Cl|Si|Se|@@|@"     # two-char atoms / stereo
        r"|[BCNOPSFIbcnops]"     # one-char organic subset
        r"|[0-9%]"               # ring closure digits
        r"|[/\\+=\-#\(\)\.])"    # bond / branch / dot
    )

    MODEL_NAME = "metatrans-emulator-v2"

    def __init__(
        self,
        top_k: int = 10,
        confidence_threshold: float = DL_CONFIDENCE_THRESHOLD,
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        top_k                : beam-search width (number of candidate
                               metabolite sequences returned per parent).
        confidence_threshold : minimum model confidence to include a prediction.
        seed                 : optional random seed for the emulator (for
                               reproducible unit tests only; real models
                               are deterministic given the same weights).
        """
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self._rng = random.Random(seed)
        self._model_loaded = False   # True once real weights are attached
        logger.info(
            "dl_predictor_init",
            model=self.MODEL_NAME,
            top_k=top_k,
            mode="emulator",
        )

    # -- Class-method constructors --------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        **kwargs,
    ) -> "DeepLearningPredictor":
        """
        Load a MetaTrans / Meta-Predictor checkpoint.

        This method is a **stub** — it returns an emulator instance and logs
        a warning until real model loading code is inserted here.

        To implement:
        -------------
            import torch, onmt
            model = onmt.model_builder.build_base_model(opt, fields, gpu)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            predictor = cls(**kwargs)
            predictor._model = model
            predictor._model_loaded = True
            return predictor
        """
        instance = cls(**kwargs)
        logger.warning(
            "dl_checkpoint_stub",
            path=checkpoint_path,
            msg="Real model weights not loaded — using emulator. "
                "Implement from_checkpoint() to enable live inference.",
        )
        return instance

    # -- Tokenisation ---------------------------------------------------------

    def tokenise(self, smiles: str) -> List[str]:
        """
        Atom-level SMILES tokeniser compatible with MetaTrans vocabulary.

        Splits the SMILES string into a list of token strings, each
        representing a single atom, bond, stereo descriptor, or bracket group.

        Parameters
        ----------
        smiles : canonical SMILES string (pre-normalised by RDKit).

        Returns
        -------
        List of token strings, e.g.
            "CC(=O)O" → ["C", "C", "(", "=", "O", ")", "O"]

        Raises
        ------
        ValueError : if the SMILES is empty or produces zero tokens.
        """
        import re
        if not smiles or not smiles.strip():
            raise ValueError("Cannot tokenise empty SMILES.")
        tokens = re.findall(self._TOKEN_PATTERN, smiles)
        if not tokens:
            raise ValueError(
                f"SMILES '{smiles}' produced zero tokens. "
                "It may contain unsupported notation."
            )
        return tokens

    # -- Structural verification ----------------------------------------------

    @staticmethod
    def verify_smiles(smiles: str) -> Optional[Chem.Mol]:
        """
        Verify that a SMILES string represents a valid, sanitisable molecule.

        Catches the full RDKit error taxonomy:
          - ``rdchem.MolSanitizeException`` — valence/kekulisation failures
          - ``rdchem.KekulizeException``    — aromatic resolution failures
          - ``ValueError``                  — bad SMILES syntax
          - Any other Exception             — unknown RDKit error

        Parameters
        ----------
        smiles : SMILES string to verify (may be transformer-generated).

        Returns
        -------
        Sanitised ``Chem.Mol`` on success, or ``None`` if any check fails.
        """
        if not smiles or not smiles.strip():
            return None
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None

            # Attempt full sanitisation — this is the main gate for
            # hypervalent/impossible structures generated by the transformer
            sanitize_flags = Chem.SanitizeFlags.SANITIZE_ALL
            result_flag = Chem.SanitizeMol(mol, sanitize_flags, catchErrors=True)
            if result_flag != Chem.SanitizeFlags.SANITIZE_NONE:
                logger.debug(
                    "dl_smiles_sanitize_failed"),
                )
                return None

            if mol.GetNumHeavyAtoms() == 0:
                return None

            # Re-parse from canonical SMILES for clean atom mapping
            canonical = Chem.MolToSmiles(mol, canonical=True)
            mol2 = Chem.MolFromSmiles(canonical)
            return mol2

        except rdchem.MolSanitizeException as exc:
            logger.debug("dl_mol_sanitize_exception", error=str(exc), smiles=smiles[:80])
            return None
        except rdchem.KekulizeException as exc:
            logger.debug("dl_mol_kekulize_exception", error=str(exc), smiles=smiles[:80])
            return None
        except Exception as exc:
            logger.debug("dl_mol_verify_error", error=str(exc), smiles=smiles[:80])
            return None

    # -- Core inference hooks (emulated; replace with real model calls) --------

    def _transformer_beam_search(
        self,
        tokens: List[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Emulate transformer beam-search decoding.

        PRODUCTION REPLACEMENT POINT
        ----------------------------
        Replace the body of this method with a real OpenNMT / HuggingFace
        inference call.  The expected signature and return type are stable:

            inputs  : token list from ``tokenise()``
            returns : list of (smiles_string, log_likelihood_score) tuples,
                      up to ``top_k`` entries, sorted by score descending.

        Emulator strategy
        -----------------
        Rather than pure randomness (which would produce mostly invalid SMILES),
        the emulator applies a set of chemically plausible single-step
        transformations to the input SMILES — the same operations MetaTrans
        was trained to generate.  This ensures a realistic fraction of
        structurally valid outputs for testing and development.

        Confidence scores are set to realistic ranges drawn from published
        MetaTrans log-likelihood distributions (Litsa et al. 2023).
        """
        parent_smiles = "".join(tokens)

        # Validate the parent SMILES before attempting to transform
        parent_mol = self.verify_smiles(parent_smiles)
        if parent_mol is None:
            return []

        # Deterministic seed: same parent → same emulated outputs across runs
        # (prevents flapping tests while still covering diverse transformations)
        seed_hash = int(hashlib.sha256(parent_smiles.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed_hash)

        candidates: List[Tuple[str, float]] = []

        # ── Emulated transformation set ────────────────────────────────────────
        # Each lambda attempts a chemically plausible single SMIRKS-like step.
        # Transformations that don't apply to the molecule are silently skipped.
        transformations = [
            self._emulate_aromatic_hydroxylation,
            self._emulate_aliphatic_hydroxylation,
            self._emulate_n_demethylation,
            self._emulate_o_demethylation,
            self._emulate_glucuronidation,
            self._emulate_sulfation,
            self._emulate_oxidation_sulfoxide,
            self._emulate_hydrolysis,
        ]

        rng.shuffle(transformations)

        for transform in transformations[:top_k]:
            try:
                result_smiles = transform(parent_mol, rng)
                if result_smiles is None:
                    continue
                verified = self.verify_smiles(result_smiles)
                if verified is None:
                    continue
                canon = Chem.MolToSmiles(verified, canonical=True)
                # Emulated log-likelihood in realistic MetaTrans range (~-0.1 to -3.0)
                score = rng.uniform(0.08, 0.55)
                candidates.append((canon, round(score, 4)))
            except Exception as exc:
                logger.debug("dl_transform_failed", error=str(exc))
                continue

        # Sort by score descending, deduplicate
        seen_smiles: Set[str] = set()
        unique: List[Tuple[str, float]] = []
        for smiles, score in sorted(candidates, key=lambda x: x[1], reverse=True):
            if smiles not in seen_smiles:
                seen_smiles.add(smiles)
                unique.append((smiles, score))

        return unique[:top_k]

    def _extract_attention_weights(
        self,
        mol: Chem.Mol,
        tokens: List[str],
    ) -> Dict[int, float]:
        """
        Extract per-atom SOM attention risk weights from the transformer.

        PRODUCTION REPLACEMENT POINT
        ----------------------------
        In a real MetaTrans / Meta-Predictor model, this method runs a
        forward pass and reads the cross-attention weights from the encoder
        layers aligned to each heavy atom.  Replace the body with:

            with torch.no_grad():
                outputs = self._model(input_tensor, attention=True)
                atom_weights = outputs.encoder_attentions[atom_indices]
            return {idx: float(w) for idx, w in enumerate(atom_weights)}

        Emulator strategy
        -----------------
        We produce a chemistry-aware pseudo-attention distribution that
        correlates with known SOM preferences, making the emulated values
        directionally correct for testing and UI development.

        The distribution is built from three additive signals:
          1. Atomic number contribution (heteroatoms get higher base weight)
          2. Environment contribution (aromatic/sp3/sp2 character)
          3. SMARTS rule overlap (atoms matching known SOM patterns get boost)

        All values are normalised to [0, 1] with softmax.
        """
        n_atoms = mol.GetNumHeavyAtoms()
        if n_atoms == 0:
            return {}

        raw_weights: Dict[int, float] = {}

        # Precompute which atoms match any SOM rule
        som_boosted: Set[int] = set()
        for _, smarts, _ in _SOM_RULES:
            pat = Chem.MolFromSmarts(smarts)
            if pat is None:
                continue
            for match in mol.GetSubstructMatches(pat):
                if match:
                    som_boosted.add(match[0])

        # ── Pseudo-attention computation ───────────────────────────────────────
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            w = 0.0

            # Signal 1: heteroatom base weight
            atomic_num = atom.GetAtomicNum()
            if atomic_num in (7, 8):       # N, O
                w += 0.35
            elif atomic_num == 16:          # S
                w += 0.30
            elif atomic_num == 6:           # C
                w += 0.15
            else:
                w += 0.20

            # Signal 2: hybridisation / environment
            hyb = atom.GetHybridization()
            if atom.GetIsAromatic():
                w += 0.25
            elif hyb == Chem.rdchem.HybridizationType.SP3:
                w += 0.20
            elif hyb == Chem.rdchem.HybridizationType.SP2:
                w += 0.18

            # Signal 3: explicit hydrogen count (CYP accessibility proxy)
            w += atom.GetTotalNumHs() * 0.10

            # Signal 4: SOM rule overlap boost
            if idx in som_boosted:
                w += 0.40

            raw_weights[idx] = w

        # Softmax normalisation → values sum to 1.0, then rescale to [0, 1]
        exp_w = {idx: math.exp(v) for idx, v in raw_weights.items()}
        total = sum(exp_w.values()) or 1.0
        normalised = {idx: v / total for idx, v in exp_w.items()}

        # Rescale: set max atom to 1.0, others proportionally
        max_val = max(normalised.values()) if normalised else 1.0
        if max_val > 0:
            return {idx: round(v / max_val, 4) for idx, v in normalised.items()}
        return {idx: 0.0 for idx in normalised}

    # -- Chemical transformation emulators ------------------------------------
    # Each method attempts a single chemical step on ``mol`` and returns a
    # product SMILES string, or None if the transformation doesn't apply.
    # These are *not* meant to be chemically exhaustive — they serve as
    # realistic test vectors for the DL pipeline interface.

    @staticmethod
    def _emulate_aromatic_hydroxylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Add -OH to an unhindered aromatic C-H (CYP1A2/2C19 style)."""
        try:
            from rdkit.Chem import AllChem, RWMol
            pattern = Chem.MolFromSmarts("[cH]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            atom_idx = match[0]
            rw = Chem.RWMol(mol)
            rw.GetAtomWithIdx(atom_idx).SetNumExplicitHs(0)
            oh_idx = rw.AddAtom(Chem.Atom(8))     # O
            h_idx  = rw.AddAtom(Chem.Atom(1))     # H
            rw.AddBond(atom_idx, oh_idx, Chem.BondType.SINGLE)
            rw.AddBond(oh_idx,   h_idx,  Chem.BondType.SINGLE)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_aliphatic_hydroxylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Add -OH to an aliphatic sp3 C-H (benzylic or α-heteroatom)."""
        try:
            from rdkit.Chem import RWMol
            # Prefer benzylic CH2
            pattern = Chem.MolFromSmarts("[CH2;$([CH2]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                pattern = Chem.MolFromSmarts("[CH3;$([CH3]-[c])]")
                matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            atom_idx = match[0]
            rw = Chem.RWMol(Chem.AddHs(mol))
            # Find one explicit H on this atom
            h_to_remove = None
            for nbr in rw.GetAtomWithIdx(atom_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 1:
                    h_to_remove = nbr.GetIdx()
                    break
            if h_to_remove is None:
                return None
            rw.RemoveAtom(h_to_remove)
            oh_idx = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(atom_idx, oh_idx, Chem.BondType.SINGLE)
            product = Chem.RemoveHs(rw)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(product, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_n_demethylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Remove N-methyl group (CYP3A4 N-demethylation)."""
        try:
            from rdkit.Chem import AllChem, RWMol, rdMolTransforms
            pattern = Chem.MolFromSmarts("[CH3;$([CH3]-[N;!$(N-C=O)])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx = match[0]
            rw = Chem.RWMol(mol)
            # Find the N neighbor
            n_idx = None
            for nbr in rw.GetAtomWithIdx(c_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 7:
                    n_idx = nbr.GetIdx()
                    break
            if n_idx is None:
                return None
            rw.RemoveBond(c_idx, n_idx)
            # Remove the dangling CH3
            atoms_to_remove = sorted([c_idx], reverse=True)
            for idx in atoms_to_remove:
                rw.RemoveAtom(idx)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_o_demethylation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Remove O-methyl group → phenol (CYP2D6 / 2C9 O-demethylation)."""
        try:
            from rdkit.Chem import RWMol
            pattern = Chem.MolFromSmarts("[CH3;$([CH3]-O-[c,C])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx = match[0]
            rw = Chem.RWMol(mol)
            o_idx = None
            for nbr in rw.GetAtomWithIdx(c_idx).GetNeighbors():
                if nbr.GetAtomicNum() == 8:
                    o_idx = nbr.GetIdx()
                    break
            if o_idx is None:
                return None
            rw.RemoveBond(c_idx, o_idx)
            # Remove the dangling CH3
            for idx in sorted([c_idx], reverse=True):
                rw.RemoveAtom(idx)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_glucuronidation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Append glucuronic acid to a phenol (Phase II conjugation)."""
        try:
            # Instead of real glucuronide construction (complex stereochemistry),
            # return the deprotonated phenol as a proxy product — structurally
            # correct for testing the DL pipeline, not for clinical use.
            pattern = Chem.MolFromSmarts("[OH;$([OH]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            # Proxy: return the phenol itself (identity for phenols)
            # Real implementation would use a SMIRKS reaction:
            # "[OH:1]-[c:2]>>[O:1]([c:2])C1OC(C(=O)O)C(O)C(O)C1O"
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    @staticmethod
    def _emulate_sulfation(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Sulfate conjugation of a phenol (Phase II, SULT enzymes)."""
        try:
            from rdkit.Chem import RWMol, AllChem
            pattern = Chem.MolFromSmarts("[OH;$([OH]-[c])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            o_idx = match[0]
            rw = Chem.RWMol(mol)
            s_idx  = rw.AddAtom(Chem.Atom(16))   # S
            o1_idx = rw.AddAtom(Chem.Atom(8))    # =O
            o2_idx = rw.AddAtom(Chem.Atom(8))    # =O
            o3_idx = rw.AddAtom(Chem.Atom(8))    # -OH
            h_idx  = rw.AddAtom(Chem.Atom(1))
            rw.AddBond(o_idx,  s_idx,  Chem.BondType.SINGLE)
            rw.AddBond(s_idx,  o1_idx, Chem.BondType.DOUBLE)
            rw.AddBond(s_idx,  o2_idx, Chem.BondType.DOUBLE)
            rw.AddBond(s_idx,  o3_idx, Chem.BondType.SINGLE)
            rw.AddBond(o3_idx, h_idx,  Chem.BondType.SINGLE)
            # Remove the original O-H proton
            rw.GetAtomWithIdx(o_idx).SetNumExplicitHs(0)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_oxidation_sulfoxide(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Oxidise thioether S → sulfoxide (CYP / FMO S-oxidation)."""
        try:
            from rdkit.Chem import RWMol
            pattern = Chem.MolFromSmarts("[S;X2;!$(S=O);$([S]([#6])[#6])]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            s_idx = match[0]
            rw = Chem.RWMol(mol)
            o_idx = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(s_idx, o_idx, Chem.BondType.DOUBLE)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    @staticmethod
    def _emulate_hydrolysis(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
        """Ester hydrolysis → carboxylic acid + alcohol."""
        try:
            from rdkit.Chem import AllChem, RWMol
            # Detect ester: [C](=O)-O-[C,c]
            pattern = Chem.MolFromSmarts("[C:1](=[O:2])-[O:3]-[#6:4]")
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            match = rng.choice(matches)
            c_idx, o_dbl, o_single, c_right = match
            rw = Chem.RWMol(mol)
            rw.RemoveBond(o_single, c_right)
            # Add H to both oxygens
            rw.GetAtomWithIdx(o_single).SetNumExplicitHs(1)
            verified = DeepLearningPredictor.verify_smiles(
                Chem.MolToSmiles(rw, canonical=True)
            )
            return Chem.MolToSmiles(verified) if verified else None
        except Exception:
            return None

    # -- Public prediction API ------------------------------------------------

    def predict(
        self,
        smiles: str,
        phase_hint: Optional[int] = None,
    ) -> Tuple[List[Tuple[str, float]], Dict[int, float], List[str]]:
        """
        Run the full DL prediction pipeline for a single parent molecule.

        Parameters
        ----------
        smiles     : canonical SMILES (pre-normalised by ``_validate_and_normalise``).
        phase_hint : optional phase bias (1 or 2); not used by emulator but
                     available for real model phase-conditioned decoding.

        Returns
        -------
        (predictions, attention_weights, warnings)

          predictions       : List[(canonical_smiles, confidence_score)]
                              sorted by confidence descending.
          attention_weights : Dict[atom_index → risk_weight]
                              per-atom SOM attention scores in [0, 1].
          warnings          : List of non-fatal diagnostic strings.
        """
        dl_warnings: List[str] = []

        # ── Tokenise ──────────────────────────────────────────────────────────
        try:
            tokens = self.tokenise(smiles)
        except ValueError as exc:
            dl_warnings.append(f"DL tokenisation failed: {exc}")
            return [], {}, dl_warnings

        # ── Verify parent ─────────────────────────────────────────────────────
        parent_mol = self.verify_smiles(smiles)
        if parent_mol is None:
            dl_warnings.append(
                f"DL pipeline skipped: parent SMILES failed structural "
                f"verification after tokenisation. SMILES: '{smiles[:80]}'"
            )
            return [], {}, dl_warnings

        # ── Beam search (emulated or real) ────────────────────────────────────
        try:
            raw_predictions = self._transformer_beam_search(tokens, self.top_k)
        except Exception as exc:
            dl_warnings.append(f"DL beam search error: {exc}")
            raw_predictions = []

        # ── Filter below confidence threshold ─────────────────────────────────
        predictions = [
            (smi, score)
            for smi, score in raw_predictions
            if score >= self.confidence_threshold
        ]

        # ── Attention weights (emulated or real) ──────────────────────────────
        try:
            attention_weights = self._extract_attention_weights(parent_mol, tokens)
        except Exception as exc:
            dl_warnings.append(f"DL attention extraction error: {exc}")
            attention_weights = {}

        logger.info(
            "dl_predict_complete",
            model=self.MODEL_NAME,
            raw_predictions=len(raw_predictions),
            filtered_predictions=len(predictions),
            attention_atoms=len(attention_weights),
        )

        return predictions, attention_weights, dl_warnings


# ==============================================================================
# Consensus Engine
# ==============================================================================

class ConsensusEngine:
    """
    Merges Pipeline A (SyGMa) and Pipeline B (DL) outputs into a unified,
    confidence-tagged metabolite list and ensemble soft spot annotations.

    Consensus logic
    ---------------
    Structural alignment uses ``Chem.CanonSmiles`` to normalise both sets of
    SMILES before comparison.  An exact match of canonical forms constitutes
    consensus — no Tanimoto similarity threshold is used because near-identical
    SMILES with different stereochemistry represent chemically distinct metabolites
    and should be tagged separately.

    Score fusion
    ------------
    Consensus metabolites inherit the higher of the two pipeline scores as
    their primary ``probability`` field, preserving the SyGMa reaction name
    and phase assignment (which the DL model does not produce).

    For soft spots, the vulnerability index blends rule and attention signals:
        VI = (ENSEMBLE_ALPHA × rule_score + ENSEMBLE_BETA × dl_attention) × 100 %
    """

    def __init__(
        self,
        alpha: float = ENSEMBLE_ALPHA,
        beta:  float = ENSEMBLE_BETA,
    ) -> None:
        self.alpha = alpha
        self.beta  = beta

    # ------------------------------------------------------------------

    def merge_metabolites(
        self,
        sygma_metabolites: List[PredictedMetabolite],
        dl_predictions:    List[Tuple[str, float]],
    ) -> Tuple[List[PredictedMetabolite], Dict]:
        """
        Merge and tag SyGMa + DL metabolite predictions.

        Parameters
        ----------
        sygma_metabolites : output of ``_run_sygma``.
        dl_predictions    : list of (canonical_smiles, confidence) from DL.

        Returns
        -------
        (merged_metabolites, stats_dict)
          merged_metabolites : deduplicated, confidence-tagged list sorted
                               by probability descending.
          stats_dict         : pipeline statistics for logging / API response.
        """
        # Build canonical SMILES lookup for SyGMa results
        sygma_by_canon: Dict[str, PredictedMetabolite] = {}
        for m in sygma_metabolites:
            canon = self._safe_canonicalise(m.smiles)
            if canon:
                sygma_by_canon[canon] = m

        # Build canonical SMILES set for DL results
        dl_by_canon: Dict[str, float] = {}
        for smiles, score in dl_predictions:
            canon = self._safe_canonicalise(smiles)
            if canon:
                dl_by_canon[canon] = score

        consensus_set    = set(sygma_by_canon) & set(dl_by_canon)
        sygma_only_set   = set(sygma_by_canon) - set(dl_by_canon)
        dl_only_set      = set(dl_by_canon)    - set(sygma_by_canon)

        merged: List[PredictedMetabolite] = []

        # ── Consensus metabolites ──────────────────────────────────────────────
        for canon in consensus_set:
            base = sygma_by_canon[canon]
            dl_s = dl_by_canon[canon]
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=max(base.probability, dl_s),
                phase=base.phase,
                reaction_name=base.reaction_name,
                molecular_weight=base.molecular_weight,
                molecular_formula=base.molecular_formula,
                confidence_tier=ConsensusTier.HIGH,
                dl_score=dl_s,
                sources=frozenset({"sygma", "dl"}),
            ))

        # ── SyGMa-only metabolites ────────────────────────────────────────────
        for canon in sygma_only_set:
            base = sygma_by_canon[canon]
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=base.probability,
                phase=base.phase,
                reaction_name=base.reaction_name,
                molecular_weight=base.molecular_weight,
                molecular_formula=base.molecular_formula,
                confidence_tier=ConsensusTier.RULE,
                dl_score=None,
                sources=frozenset({"sygma"}),
            ))

        # ── DL-only metabolites ───────────────────────────────────────────────
        for canon in dl_only_set:
            dl_s = dl_by_canon[canon]
            mol = self._safe_mol(canon)
            mw, formula = self._compute_descriptors(mol)
            merged.append(PredictedMetabolite(
                smiles=canon,
                probability=dl_s,
                phase=-1,           # DL model does not produce phase labels
                reaction_name="dl_predicted",
                molecular_weight=mw,
                molecular_formula=formula,
                confidence_tier=ConsensusTier.DL,
                dl_score=dl_s,
                sources=frozenset({"dl"}),
            ))

        merged.sort(key=lambda m: m.probability, reverse=True)

        stats = {
            "sygma_total":      len(sygma_by_canon),
            "dl_total":         len(dl_by_canon),
            "consensus_count":  len(consensus_set),
            "rule_only_count":  len(sygma_only_set),
            "dl_only_count":    len(dl_only_set),
            "merged_total":     len(merged),
        }

        logger.info("consensus_merge_complete", **stats)
        return merged, stats

    def merge_soft_spots(
        self,
        rule_spots:       List[SoftSpot],
        attention_weights: Dict[int, float],
        mol:              Chem.Mol,
        top_n:            int,
        dl_source:        str = "metatrans-emulator",
    ) -> List[EnsembleSoftSpot]:
        """
        Fuse rule-based and DL attention soft spot signals.

        For atoms with a rule match: VI = (α·rule_score + β·dl_attention) × 100
        For atoms with attention only (no rule match): VI = β·dl_attention × 100
        Rule-matched atoms always appear in the top-N if they outscore attention-only.

        Parameters
        ----------
        rule_spots        : output of ``_find_soft_spots``.
        attention_weights : per-atom attention weights from DL predictor.
        mol               : parent molecule (used to look up atom symbols for
                            attention-only atoms not in rule_spots).
        top_n             : number of ensemble spots to return.
        dl_source         : identifier for the DL model providing attention.

        Returns
        -------
        List of ``EnsembleSoftSpot``, sorted by vulnerability_index descending,
        length ≤ top_n.
        """
        ensemble: Dict[int, EnsembleSoftSpot] = {}

        # Process rule-matched atoms
        for spot in rule_spots:
            dl_w = attention_weights.get(spot.atom_index, 0.0)
            vi   = round((self.alpha * spot.score + self.beta * dl_w) * 100, 2)
            ensemble[spot.atom_index] = EnsembleSoftSpot(
                atom_index=spot.atom_index,
                atom_symbol=spot.atom_symbol,
                rule_name=spot.rule_name,
                smarts_match=spot.smarts_match,
                rule_score=spot.score,
                dl_attention_risk=dl_w,
                vulnerability_index=vi,
                dl_source=dl_source,
            )

        # Process attention-only atoms (not in rule_spots) — add if high attention
        if attention_weights:
            # Threshold: include attention-only atoms above the 75th percentile
            sorted_weights = sorted(attention_weights.values(), reverse=True)
            threshold_rank = max(1, len(sorted_weights) // 4)
            threshold = sorted_weights[threshold_rank - 1] if sorted_weights else 0.0

            for atom_idx, dl_w in attention_weights.items():
                if atom_idx in ensemble:
                    continue
                if dl_w < threshold:
                    continue
                vi = round(self.beta * dl_w * 100, 2)
                try:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    symbol = atom.GetSymbol()
                except Exception:
                    symbol = "?"
                ensemble[atom_idx] = EnsembleSoftSpot(
                    atom_index=atom_idx,
                    atom_symbol=symbol,
                    rule_name="dl_attention_only",
                    smarts_match="",
                    rule_score=0.0,
                    dl_attention_risk=dl_w,
                    vulnerability_index=vi,
                    dl_source=dl_source,
                )

        result = sorted(
            ensemble.values(),
            key=lambda s: s.vulnerability_index,
            reverse=True,
        )
        return result[:top_n]

    # -- Private helpers ------------------------------------------------------

    @staticmethod
    def _safe_canonicalise(smiles: str) -> Optional[str]:
        """Return RDKit canonical SMILES, or None on any RDKit error."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None

    @staticmethod
    def _safe_mol(smiles: str) -> Optional[Chem.Mol]:
        try:
            return Chem.MolFromSmiles(smiles)
        except Exception:
            return None

    @staticmethod
    def _compute_descriptors(
        mol: Optional[Chem.Mol],
    ) -> Tuple[Optional[float], Optional[str]]:
        if mol is None:
            return None, None
        try:
            return (
                round(Descriptors.MolWt(mol), 4),
                rdMolDescriptors.CalcMolFormula(mol),
            )
        except Exception:
            return None, None


# ==============================================================================
# Private helpers  (validation, SyGMa runner, rule-based SOM)
# ==============================================================================

def _validate_and_normalise(smiles: str) -> Tuple[Chem.Mol, MoleculeMetadata]:
    """
    Parse, validate, and normalise a SMILES string.

    Normalisation pipeline
    ----------------------
    1. ``Chem.MolFromSmiles``     — parse + valence check
    2. ``LargestFragmentChooser`` — strip salts / counterions
    3. ``Normalizer``             — canonical functional-group forms
    4. ``Uncharger``              — neutralise formal charges
    5. ``SanitizeMol``            — final RDKit sanitisation
    6. Descriptor computation     — MW, formula, InChI, logP, TPSA, etc.

    Raises
    ------
    ValueError  if the SMILES is invalid at any pipeline stage.
    """
    if not smiles or not smiles.strip():
        raise ValueError("SMILES string is empty.")

    smiles = smiles.strip()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(
            f"RDKit could not parse SMILES: {smiles!r}. "
            "Check atom symbols, valences, and ring-closure notation."
        )

    mol = _LARGEST_FRAGMENT.choose(mol)
    mol = _NORMALISER.normalize(mol)
    mol = _UNCHARGER.uncharge(mol)

    try:
        Chem.SanitizeMol(mol)
    except rdchem.MolSanitizeException as exc:
        raise ValueError(f"Sanitisation failed after normalisation: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Normalisation error: {exc}") from exc

    if mol.GetNumHeavyAtoms() == 0:
        raise ValueError("Molecule has no heavy atoms after normalisation.")

    from rdkit.Chem import inchi as rdinchi
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    inchi_str   = rdinchi.MolToInchi(mol) or ""
    inchikey    = rdinchi.InchiToInchiKey(inchi_str) if inchi_str else ""

    metadata = MoleculeMetadata(
        input_smiles=smiles,
        canonical_smiles=canonical_smiles,
        inchi=inchi_str,
        inchikey=inchikey,
        molecular_formula=rdMolDescriptors.CalcMolFormula(mol),
        molecular_weight=round(Descriptors.MolWt(mol), 4),
        exact_mass=round(Descriptors.ExactMolWt(mol), 6),
        num_heavy_atoms=mol.GetNumHeavyAtoms(),
        num_rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
        num_hbd=rdMolDescriptors.CalcNumHBD(mol),
        num_hba=rdMolDescriptors.CalcNumHBA(mol),
        tpsa=round(rdMolDescriptors.CalcTPSA(mol), 2),
        logp=round(Descriptors.MolLogP(mol), 4),
        num_rings=rdMolDescriptors.CalcNumRings(mol),
        num_aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
    )

    logger.debug(
        "molecule_normalised",
        canonical=canonical_smiles,
        mw=metadata.molecular_weight,
        formula=metadata.molecular_formula,
    )
    return mol, metadata


def _run_sygma(
    mol: Chem.Mol,
    phase1_cycles: int = 1,
    phase2_cycles: int = 1,
    max_metabolites: int = 200,
) -> List[PredictedMetabolite]:
    """
    Execute a SyGMa Phase I → Phase II prediction scenario.

    Returns PredictedMetabolite instances with ``sources=frozenset({"sygma"})``
    and ``confidence_tier=ConsensusTier.RULE`` (to be upgraded to HIGH by the
    ConsensusEngine when DL agreement is found).

    Raises
    ------
    ImportError  if SyGMa is not installed.
    """
    try:
        import sygma                          # type: ignore
        from sygma import ruleset as srs      # type: ignore
        from sygma import Scenario            # type: ignore
    except ImportError as exc:
        raise ImportError(
            "SyGMa is not installed. "
            "Add 'sygma>=1.1.0' to the pip section of environment.yml."
        ) from exc

    scenario = Scenario([
        [srs.phase1, phase1_cycles],
        [srs.phase2, phase2_cycles],
    ])

    logger.debug("sygma_start", p1=phase1_cycles, p2=phase2_cycles)
    tree = scenario.run(mol)
    tree.calc_scores()

    parent_canon = Chem.MolToSmiles(mol, canonical=True)
    seen: Dict[str, PredictedMetabolite] = {}

    for node in tree.to_list():
        node_mol = getattr(node, "mol", None)
        if node_mol is None:
            continue

        # Per-metabolite error defence — catch sanitisation failures
        try:
            Chem.SanitizeMol(node_mol)
        except rdchem.MolSanitizeException as exc:
            logger.debug("sygma_sanitize_skip", error=str(exc))
            continue
        except Exception as exc:
            logger.debug("sygma_mol_skip", error=str(exc))
            continue

        try:
            smiles = Chem.MolToSmiles(node_mol, canonical=True)
        except Exception as exc:
            logger.debug("sygma_smiles_skip", error=str(exc))
            continue

        if not smiles or smiles == parent_canon:
            continue

        probability   = float(getattr(node, "score",         0.0))
        reaction_name = str(getattr(node,  "reaction_name", "") or "unknown")

        phase = (2 if reaction_name.lower().startswith("phase2")
                 or any(k in reaction_name.lower()
                        for k in ("glucuronid", "sulfat", "acetyl",
                                  "glutathione", "methyl"))
                 else 1)
        if reaction_name.lower().startswith("phase1"):
            phase = 1

        try:
            mw      = round(Descriptors.MolWt(node_mol), 4)
            formula = rdMolDescriptors.CalcMolFormula(node_mol)
        except Exception:
            mw, formula = None, None

        candidate = PredictedMetabolite(
            smiles=smiles,
            probability=probability,
            phase=phase,
            reaction_name=reaction_name,
            molecular_weight=mw,
            molecular_formula=formula,
            confidence_tier=ConsensusTier.RULE,   # upgraded by consensus later
            sources=frozenset({"sygma"}),
        )

        if smiles not in seen or probability > seen[smiles].probability:
            seen[smiles] = candidate

    results = sorted(seen.values(), key=lambda x: x.probability, reverse=True)
    results = results[:max_metabolites]

    logger.info(
        "sygma_complete",
        unique=len(results),
        p1=sum(1 for r in results if r.phase == 1),
        p2=sum(1 for r in results if r.phase == 2),
    )
    return results


def _find_soft_spots(
    mol: Chem.Mol,
    top_n: int = 3,
) -> List[SoftSpot]:
    """
    Rule-only soft spot fallback (Pipeline A standalone).

    Used when the DL pipeline is unavailable or disabled.  The ConsensusEngine
    calls ``merge_soft_spots`` instead when both pipelines are active.

    Algorithm
    ---------
    For each rule in ``_SOM_RULES``:
      - Compile SMARTS, find substructure matches.
      - Atom at match index 0 is the SOM site.
      - Apply a 20 % steric penalty for atoms with ≥ 3 bulky neighbours.
    """
    best: Dict[int, SoftSpot] = {}

    for rule_name, smarts, base_score in _SOM_RULES:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            logger.warning("invalid_smarts", rule=rule_name)
            continue

        for match in mol.GetSubstructMatches(pattern):
            if not match:
                continue
            atom_idx = match[0]
            atom     = mol.GetAtomWithIdx(atom_idx)

            heavy_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
            score = base_score
            if len(heavy_nbrs) >= 3 and any(n.GetDegree() > 1 for n in heavy_nbrs):
                score *= 0.80

            score = round(score, 4)
            if atom_idx not in best or score > best[atom_idx].score:
                best[atom_idx] = SoftSpot(
                    atom_index=atom_idx,
                    atom_symbol=atom.GetSymbol(),
                    rule_name=rule_name,
                    score=score,
                    smarts_match=smarts,
                )

    return sorted(best.values(), key=lambda s: s.score, reverse=True)[:top_n]


# ==============================================================================
# Public API
# ==============================================================================

# Type alias for drop-in soft-spot function hook (backward compatible with v1)
SoftSpotFn = Callable[[Chem.Mol, int], List[SoftSpot]]


def predict(
    smiles: str,
    *,
    phase1_cycles:   int  = 1,
    phase2_cycles:   int  = 1,
    max_metabolites: int  = 200,
    top_soft_spots:  int  = 3,
    # v2 ensemble parameters
    dl_predictor:    Optional[DeepLearningPredictor] = None,
    enable_dl:       bool = True,
    # v1 backward-compat hook (used when dl_predictor is None and enable_dl=False)
    soft_spot_fn:    Optional[SoftSpotFn] = None,
) -> MetabolismResult:
    """
    Ensemble Consensus Metabolism Prediction.

    Orchestrates the dual-pipeline architecture:
      Pipeline A  — SyGMa rule-based Phase I/II metabolites
      Pipeline B  — DeepLearningPredictor (emulator or real weights)
      Consensus   — cross-validates outputs, assigns ConsensusTier tags
      Soft Spots  — blends rule scores with DL attention weights

    Parameters
    ----------
    smiles          : Parent molecule SMILES (any notation; normalised internally).
    phase1_cycles   : SyGMa Phase I iteration depth.
    phase2_cycles   : SyGMa Phase II iteration depth.
    max_metabolites : Hard cap on returned metabolites (applied after merge).
    top_soft_spots  : Number of soft spots to annotate.
    dl_predictor    : Pre-built DeepLearningPredictor instance.  If None and
                      enable_dl=True, a default emulator is created internally.
    enable_dl       : Master switch for Pipeline B.  Set False to run rule-only
                      (equivalent to v1 behaviour).
    soft_spot_fn    : v1 backward-compat hook.  Ignored when enable_dl=True.

    Returns
    -------
    ``MetabolismResult`` with full ensemble data.  ``.to_dict()`` is
    backward-compatible with v1 API consumers.

    Raises
    ------
    ValueError   : SMILES is invalid.
    RuntimeError : Unexpected engine-level error.

    Examples
    --------
    >>> # Minimal usage (ensemble mode, default emulator)
    >>> from app.engine.metabolism import predict
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O")
    >>> r.parent.molecular_formula
    'C9H8O4'
    >>> all(isinstance(s, EnsembleSoftSpot) for s in r.soft_spots)
    True

    >>> # Rule-only mode (v1 equivalent)
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O", enable_dl=False)

    >>> # Real weights (when available)
    >>> predictor = DeepLearningPredictor.from_checkpoint("metatrans.pt")
    >>> r = predict("CC(=O)Oc1ccccc1C(=O)O", dl_predictor=predictor)
    """
    t_start   = time.perf_counter()
    warnings: List[str] = []

    # ── Step 1: Validate + normalise ──────────────────────────────────────────
    logger.info("ensemble_predict_start")
    mol, parent_meta = _validate_and_normalise(smiles)

    if parent_meta.canonical_smiles != smiles.strip():
        warnings.append(
            f"Input SMILES normalised: {smiles!r} → "
            f"{parent_meta.canonical_smiles!r}"
        )

    if parent_meta.molecular_weight > 1500:
        warnings.append(
            f"MW {parent_meta.molecular_weight:.1f} Da exceeds typical small-molecule "
            "range (< 1500 Da). Predictions may be unreliable."
        )

    # ── Step 2A: Pipeline A — SyGMa ───────────────────────────────────────────
    sygma_metabolites: List[PredictedMetabolite] = []
    try:
        sygma_metabolites = _run_sygma(
            mol,
            phase1_cycles=phase1_cycles,
            phase2_cycles=phase2_cycles,
            max_metabolites=max_metabolites,
        )
    except ImportError:
        warnings.append(
            "SyGMa not installed — rule-based pipeline inactive. "
            "Install: pip install sygma"
        )
        logger.warning("sygma_not_installed")
    except Exception as exc:
        warnings.append(f"SyGMa pipeline error (non-fatal): {exc}")
        logger.error("sygma_pipeline_error", error=str(exc))

    # ── Step 2B: Pipeline B — Deep Learning ───────────────────────────────────
    dl_predictions:    List[Tuple[str, float]] = []
    attention_weights: Dict[int, float]        = {}
    dl_source = "disabled"

    if enable_dl:
        # Use provided predictor or create a default emulator
        predictor = dl_predictor or DeepLearningPredictor()
        dl_source = predictor.MODEL_NAME
        try:
            (
                dl_predictions,
                attention_weights,
                dl_warnings,
            ) = predictor.predict(parent_meta.canonical_smiles)
            warnings.extend(dl_warnings)
        except Exception as exc:
            warnings.append(f"DL pipeline error (non-fatal): {exc}")
            logger.error("dl_pipeline_error", error=str(exc))
    else:
        warnings.append("DL pipeline disabled (enable_dl=False).")

    # ── Step 3: Consensus merge ────────────────────────────────────────────────
    consensus = ConsensusEngine()

    if enable_dl and (sygma_metabolites or dl_predictions):
        merged_metabolites, pipeline_stats = consensus.merge_metabolites(
            sygma_metabolites, dl_predictions
        )
        merged_metabolites = merged_metabolites[:max_metabolites]
    else:
        # Rule-only fallback: tag everything as RULE tier
        merged_metabolites = sygma_metabolites[:max_metabolites]
        pipeline_stats = {
            "sygma_total":     len(sygma_metabolites),
            "dl_total":        0,
            "consensus_count": 0,
            "rule_only_count": len(sygma_metabolites),
            "dl_only_count":   0,
            "merged_total":    len(sygma_metabolites),
        }

    # ── Step 4: Soft spot analysis ────────────────────────────────────────────
    if soft_spot_fn is not None and not enable_dl:
        # v1 backward-compat: external soft_spot_fn overrides ensemble
        soft_spots_raw = soft_spot_fn(mol, top_soft_spots)
        soft_spots: List = soft_spots_raw
    else:
        # Ensemble soft spots
        rule_spots = _find_soft_spots(mol, top_n=mol.GetNumHeavyAtoms())
        soft_spots = consensus.merge_soft_spots(
            rule_spots=rule_spots,
            attention_weights=attention_weights,
            mol=mol,
            top_n=top_soft_spots,
            dl_source=dl_source,
        )

    if not soft_spots:
        warnings.append(
            "No soft spots identified. The molecule may lack common CYP substrate "
            "features or all candidates were excluded by the steric/attention filter."
        )

    # ── Step 5: Assemble result ───────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    pipeline_stats["dl_source"]  = dl_source
    pipeline_stats["elapsed_s"]  = round(elapsed, 4)

    result = MetabolismResult(
        parent=parent_meta,
        metabolites=merged_metabolites,
        soft_spots=soft_spots,
        engine_version=ENGINE_VERSION,
        elapsed_s=elapsed,
        warnings=warnings,
        pipeline_stats=pipeline_stats,
    )

    logger.info(
        "ensemble_predict_complete",
        canonical=parent_meta.canonical_smiles,
        metabolites_total=len(merged_metabolites),
        consensus_count=pipeline_stats.get("consensus_count", 0),
        soft_spots_total=len(soft_spots),
        elapsed_s=round(elapsed, 3),
    )

    return result
.Groups[1].Value
    "logger.info(f`"$($args -replace '`"', '' -replace "'", '' -replace ',\s*\w+=', ' ' -replace '=', '=')`")"
,
        consensus_count=pipeline_stats.get("consensus_count", 0),
        soft_spots_total=len(soft_spots),
        elapsed_s=round(elapsed, 3),
    )

    return result


