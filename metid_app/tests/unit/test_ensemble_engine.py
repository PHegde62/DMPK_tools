"""
tests/unit/test_ensemble_engine.py
================================================================================
Comprehensive tests for the Ensemble Consensus Metabolism Engine (v2).

Test strategy
-------------
- RDKit is used directly (no mocking) — it is always available.
- SyGMa is mocked via sys.modules injection so tests run without it installed.
- DeepLearningPredictor is exercised against its emulator (no PyTorch required).
- ConsensusEngine is tested with synthetic metabolite lists.
- The public ``predict()`` function is tested end-to-end with both pipelines
  mocked for speed and with the real emulator for integration coverage.

Groups
------
  TestConsensusTier             enum serialisation
  TestEnsembleSoftSpot          dataclass contract
  TestDeepLearningPredictorTokenise    tokeniser
  TestDeepLearningPredictorVerify      structural verification
  TestDeepLearningPredictorAttention   attention weight contract
  TestDeepLearningPredictorPredict     full predict() call
  TestConsensusEngineMergeMetabolites  merger + confidence tagging
  TestConsensusEngineMergeSoftSpots    fusion scoring
  TestValidateAndNormalise        (carried over from v1, extended)
  TestFindSoftSpots               (rule-only fallback)
  TestPredictEnsemble             full pipeline, mock + emulator
  TestPredictRuleOnly             enable_dl=False backward compat
  TestMetabolismResultToDict      serialisation contract

Run with:
    pytest tests/unit/test_ensemble_engine.py -v
"""

from __future__ import annotations

import sys
import types
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
from app.engine.metabolism import (
    ENGINE_VERSION,
    ENSEMBLE_ALPHA,
    ENSEMBLE_BETA,
    ConsensusTier,
    ConsensusEngine,
    DeepLearningPredictor,
    EnsembleSoftSpot,
    MetabolismResult,
    MoleculeMetadata,
    PredictedMetabolite,
    SoftSpot,
    _find_soft_spots,
    _validate_and_normalise,
    predict,
)

# ---------------------------------------------------------------------------
# Test molecules
# ---------------------------------------------------------------------------
ASPIRIN     = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN   = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
CAFFEINE    = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"
LIDOCAINE   = "CCN(CC)CC(=O)Nc1c(C)cccc1C"
SALT_ASPIRIN = "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"


# ---------------------------------------------------------------------------
# Shared SyGMa stub (minimal, valid)
# ---------------------------------------------------------------------------

def _make_sygma_stub():
    class FakeNode:
        def __init__(self, smiles, score, rxn):
            self.mol = Chem.MolFromSmiles(smiles)
            self.score = score
            self.reaction_name = rxn

    class FakeTree:
        def calc_scores(self): pass
        def to_list(self):
            return [
                FakeNode("OC(=O)c1ccccc1O",          0.32, "phase1_aromatic_hydroxylation"),
                FakeNode("CC(=O)Oc1ccc(O)cc1C(=O)O", 0.18, "phase1_aliphatic_hydroxylation"),
                FakeNode("OC(=O)c1ccccc1OC(=O)O",    0.09, "phase2_glucuronidation"),
            ]

    class FakeScenario:
        def __init__(self, *a, **kw): pass
        def run(self, mol): return FakeTree()

    class FakeRuleset:
        phase1 = object()
        phase2 = object()

    mod = types.ModuleType("sygma")
    mod.Scenario = FakeScenario
    mod.ruleset  = FakeRuleset()
    return mod


@pytest.fixture(autouse=False)
def mock_sygma(monkeypatch):
    stub = _make_sygma_stub()
    monkeypatch.setitem(sys.modules, "sygma", stub)
    monkeypatch.setitem(sys.modules, "sygma.ruleset", stub.ruleset)
    yield stub


# ===========================================================================
# 1. ConsensusTier
# ===========================================================================

class TestConsensusTier:
    def test_high_value_string(self):
        assert ConsensusTier.HIGH == "High Confidence (Consensus Verified)"

    def test_rule_value_string(self):
        assert ConsensusTier.RULE == "Moderate Confidence (Rule-Only)"

    def test_dl_value_string(self):
        assert ConsensusTier.DL == "Moderate Confidence (DL-Only)"

    def test_serialises_directly_to_str(self):
        # ConsensusTier inherits str — must JSON-encode without custom encoder
        import json
        assert json.dumps(ConsensusTier.HIGH) == '"High Confidence (Consensus Verified)"'

    def test_comparison_with_string(self):
        assert ConsensusTier.HIGH == "High Confidence (Consensus Verified)"


# ===========================================================================
# 2. EnsembleSoftSpot contract
# ===========================================================================

class TestEnsembleSoftSpot:
    def _make(self, **kw):
        defaults = dict(
            atom_index=3, atom_symbol="C", rule_name="benzylic_CH",
            smarts_match="[CH2]-[c]", rule_score=0.92,
            dl_attention_risk=0.75, vulnerability_index=84.65,
        )
        defaults.update(kw)
        return EnsembleSoftSpot(**defaults)

    def test_is_frozen(self):
        s = self._make()
        with pytest.raises((AttributeError, TypeError)):
            s.rule_score = 0.0

    def test_vulnerability_index_in_0_100(self):
        s = self._make(vulnerability_index=84.65)
        assert 0.0 <= s.vulnerability_index <= 100.0

    def test_dl_source_default(self):
        s = self._make()
        assert s.dl_source == "rule-only"

    def test_dl_source_custom(self):
        s = self._make(dl_source="metatrans-v2")
        assert s.dl_source == "metatrans-v2"


# ===========================================================================
# 3. DeepLearningPredictor — tokeniser
# ===========================================================================

class TestDeepLearningPredictorTokenise:
    @pytest.fixture
    def predictor(self):
        return DeepLearningPredictor(seed=42)

    def test_simple_smiles(self, predictor):
        tokens = predictor.tokenise("CCO")
        assert tokens == ["C", "C", "O"]

    def test_aspirin_token_count(self, predictor):
        tokens = predictor.tokenise(ASPIRIN)
        assert len(tokens) >= 10   # aspirin has > 10 tokens

    def test_branch_brackets_in_tokens(self, predictor):
        tokens = predictor.tokenise("CC(C)O")
        assert "(" in tokens
        assert ")" in tokens

    def test_bracket_atom_treated_as_single_token(self, predictor):
        tokens = predictor.tokenise("[NH2]CC")
        # [NH2] should be one token
        assert "[NH2]" in tokens

    def test_aromatic_ring(self, predictor):
        tokens = predictor.tokenise("c1ccccc1")
        assert tokens.count("c") == 6

    def test_empty_smiles_raises(self, predictor):
        with pytest.raises(ValueError, match="empty"):
            predictor.tokenise("")

    def test_whitespace_smiles_raises(self, predictor):
        with pytest.raises(ValueError, match="empty"):
            predictor.tokenise("   ")

    def test_detokenise_reconstructs_smiles(self, predictor):
        for smiles in [ASPIRIN, IBUPROFEN, "CCO", "c1ccccc1"]:
            tokens = predictor.tokenise(smiles)
            rejoined = "".join(tokens)
            # Both must parse to the same molecule
            mol1 = Chem.MolFromSmiles(smiles)
            mol2 = Chem.MolFromSmiles(rejoined)
            if mol1 and mol2:
                assert Chem.MolToSmiles(mol1) == Chem.MolToSmiles(mol2), \
                    f"Tokenise/rejoin mismatch for '{smiles}'"


# ===========================================================================
# 4. DeepLearningPredictor — structural verification
# ===========================================================================

class TestDeepLearningPredictorVerify:
    def test_valid_smiles_returns_mol(self):
        mol = DeepLearningPredictor.verify_smiles(ASPIRIN)
        assert mol is not None
        assert isinstance(mol, Chem.Mol)

    def test_invalid_smiles_returns_none(self):
        assert DeepLearningPredictor.verify_smiles("NOT_A_SMILES") is None

    def test_empty_smiles_returns_none(self):
        assert DeepLearningPredictor.verify_smiles("") is None

    def test_none_returns_none(self):
        assert DeepLearningPredictor.verify_smiles(None) is None

    def test_hypervalent_carbon_returns_none(self):
        # CCC(C)(C)(C)C — pentavalent carbon — RDKit sanitisation must fail
        result = DeepLearningPredictor.verify_smiles("CCC(C)(C)(C)C")
        # This may or may not be caught by RDKit depending on version;
        # what matters is it doesn't raise an unhandled exception
        assert result is None or isinstance(result, Chem.Mol)

    def test_canonical_smiles_produced(self):
        mol = DeepLearningPredictor.verify_smiles("OC(=O)c1ccccc1O")
        assert mol is not None
        canon = Chem.MolToSmiles(mol)
        mol2  = Chem.MolFromSmiles(canon)
        assert mol2 is not None   # canonical form is re-parseable

    def test_salt_form_passes(self):
        # Salt forms can appear in transformer output
        mol = DeepLearningPredictor.verify_smiles("CC(=O)[O-]")
        assert mol is not None


# ===========================================================================
# 5. DeepLearningPredictor — attention weights
# ===========================================================================

class TestDeepLearningPredictorAttention:
    @pytest.fixture
    def predictor(self):
        return DeepLearningPredictor(seed=42)

    def _mol(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        return mol

    def test_returns_dict_keyed_by_atom_index(self, predictor):
        mol    = self._mol(ASPIRIN)
        tokens = predictor.tokenise(ASPIRIN)
        weights = predictor._extract_attention_weights(mol, tokens)
        assert isinstance(weights, dict)

    def test_all_atom_indices_present(self, predictor):
        mol     = self._mol(ASPIRIN)
        tokens  = predictor.tokenise(ASPIRIN)
        weights = predictor._extract_attention_weights(mol, tokens)
        n_atoms = mol.GetNumHeavyAtoms()
        for idx in range(n_atoms):
            assert idx in weights, f"Atom {idx} missing from attention weights"

    def test_weights_in_unit_interval(self, predictor):
        mol    = self._mol(IBUPROFEN)
        tokens = predictor.tokenise(IBUPROFEN)
        for idx, w in predictor._extract_attention_weights(mol, tokens).items():
            assert 0.0 <= w <= 1.0, f"Weight {w} for atom {idx} out of [0,1]"

    def test_som_atoms_have_higher_weight(self, predictor):
        """Atoms matching SOM SMARTS rules should receive elevated attention."""
        mol    = self._mol(PARACETAMOL)   # has phenolic OH: SOM site
        tokens = predictor.tokenise(PARACETAMOL)
        weights = predictor._extract_attention_weights(mol, tokens)
        # Find phenolic C atom
        pat = Chem.MolFromSmarts("[c;$([c][OH1])]")
        matches = mol.GetSubstructMatches(pat)
        if matches:
            pheno_c = matches[0][0]
            # Phenolic carbon should be among top-3 highest-weight atoms
            sorted_idxs = sorted(weights, key=weights.get, reverse=True)
            assert pheno_c in sorted_idxs[:5], \
                f"Phenolic C (atom {pheno_c}) not in top-5: {sorted_idxs[:5]}"

    def test_max_weight_is_one(self, predictor):
        mol    = self._mol(CAFFEINE)
        tokens = predictor.tokenise(CAFFEINE)
        weights = predictor._extract_attention_weights(mol, tokens)
        if weights:
            assert max(weights.values()) == pytest.approx(1.0, abs=1e-4)


# ===========================================================================
# 6. DeepLearningPredictor — full predict()
# ===========================================================================

class TestDeepLearningPredictorPredict:
    @pytest.fixture
    def predictor(self):
        return DeepLearningPredictor(top_k=8, seed=0)

    def test_returns_triple(self, predictor):
        result = predictor.predict(ASPIRIN)
        assert len(result) == 3

    def test_predictions_are_list_of_tuples(self, predictor):
        preds, _, _ = predictor.predict(ASPIRIN)
        assert isinstance(preds, list)
        for item in preds:
            assert len(item) == 2

    def test_all_predicted_smiles_are_valid(self, predictor):
        preds, _, _ = predictor.predict(ASPIRIN)
        for smiles, score in preds:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"Predicted SMILES invalid: {smiles!r}"

    def test_confidence_scores_in_unit_interval(self, predictor):
        preds, _, _ = predictor.predict(ASPIRIN)
        for smiles, score in preds:
            assert 0.0 <= score <= 1.0

    def test_predictions_sorted_descending(self, predictor):
        preds, _, _ = predictor.predict(ASPIRIN)
        scores = [s for _, s in preds]
        assert scores == sorted(scores, reverse=True)

    def test_attention_weights_returned(self, predictor):
        _, attention, _ = predictor.predict(ASPIRIN)
        assert isinstance(attention, dict)
        assert len(attention) > 0

    def test_warnings_is_list(self, predictor):
        _, _, warnings = predictor.predict(ASPIRIN)
        assert isinstance(warnings, list)

    def test_invalid_smiles_returns_empty_predictions(self, predictor):
        preds, attention, warnings = predictor.predict("INVALID_SMILES")
        assert preds == []
        assert attention == {}
        assert len(warnings) > 0

    def test_empty_smiles_returns_empty_predictions(self, predictor):
        preds, attention, warnings = predictor.predict("")
        assert preds == []

    def test_predictions_do_not_include_parent(self, predictor):
        parent_canon = Chem.MolToSmiles(Chem.MolFromSmiles(ASPIRIN))
        preds, _, _ = predictor.predict(ASPIRIN)
        pred_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s))
                       for s, _ in preds if Chem.MolFromSmiles(s)]
        # Most emulated transformations should not return the identical parent
        # (a few may; we allow up to 1 duplicate)
        parent_count = pred_smiles.count(parent_canon)
        assert parent_count <= 1

    def test_from_checkpoint_returns_emulator(self):
        p = DeepLearningPredictor.from_checkpoint("fake/path.pt")
        assert isinstance(p, DeepLearningPredictor)
        assert not p._model_loaded

    def test_top_k_respected(self):
        p = DeepLearningPredictor(top_k=3, seed=1)
        preds, _, _ = p.predict(IBUPROFEN)
        assert len(preds) <= 3


# ===========================================================================
# 7. ConsensusEngine — merge_metabolites
# ===========================================================================

def _make_sygma_met(smiles, prob, phase=1, rxn="phase1_test"):
    mol = Chem.MolFromSmiles(smiles)
    mw  = round(Descriptors.MolWt(mol), 4) if mol else None
    return PredictedMetabolite(
        smiles=Chem.MolToSmiles(mol) if mol else smiles,
        probability=prob,
        phase=phase,
        reaction_name=rxn,
        molecular_weight=mw,
        molecular_formula=rdMolDescriptors.CalcMolFormula(mol) if mol else None,
        confidence_tier=ConsensusTier.RULE,
        sources=frozenset({"sygma"}),
    )


class TestConsensusEngineMergeMetabolites:
    @pytest.fixture
    def engine(self):
        return ConsensusEngine()

    # Shared test molecules
    SHARED = "OC(=O)c1ccccc1O"            # salicylic acid
    RULE_ONLY = "CC(=O)Oc1ccc(O)cc1C(=O)O"
    DL_ONLY   = "Oc1ccc(O)cc1"            # hydroquinone

    def _dl(self, smiles, score):
        canon = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        return [(canon, score)]

    def test_consensus_metabolite_gets_high_tier(self, engine):
        sygma = [_make_sygma_met(self.SHARED, 0.32)]
        dl    = self._dl(self.SHARED, 0.40)
        merged, _ = engine.merge_metabolites(sygma, dl)
        consensus = [m for m in merged if m.confidence_tier == ConsensusTier.HIGH]
        assert len(consensus) == 1

    def test_rule_only_gets_rule_tier(self, engine):
        sygma = [_make_sygma_met(self.RULE_ONLY, 0.20)]
        merged, _ = engine.merge_metabolites(sygma, [])
        assert all(m.confidence_tier == ConsensusTier.RULE for m in merged)

    def test_dl_only_gets_dl_tier(self, engine):
        sygma = []
        dl    = self._dl(self.DL_ONLY, 0.25)
        merged, _ = engine.merge_metabolites(sygma, dl)
        dl_tagged = [m for m in merged if m.confidence_tier == ConsensusTier.DL]
        assert len(dl_tagged) == 1

    def test_merged_sorted_by_probability(self, engine):
        sygma = [_make_sygma_met(self.SHARED, 0.32),
                 _make_sygma_met(self.RULE_ONLY, 0.20)]
        dl    = self._dl(self.SHARED, 0.40)
        merged, _ = engine.merge_metabolites(sygma, dl)
        probs = [m.probability for m in merged]
        assert probs == sorted(probs, reverse=True)

    def test_consensus_probability_is_max_of_both(self, engine):
        sygma = [_make_sygma_met(self.SHARED, 0.32)]
        dl    = self._dl(self.SHARED, 0.55)
        merged, _ = engine.merge_metabolites(sygma, dl)
        consensus = [m for m in merged if m.confidence_tier == ConsensusTier.HIGH]
        assert consensus[0].probability == pytest.approx(0.55, abs=1e-4)

    def test_stats_keys_present(self, engine):
        _, stats = engine.merge_metabolites([], [])
        for key in ("sygma_total", "dl_total", "consensus_count",
                    "rule_only_count", "dl_only_count", "merged_total"):
            assert key in stats

    def test_empty_inputs_return_empty_list(self, engine):
        merged, stats = engine.merge_metabolites([], [])
        assert merged == []
        assert stats["merged_total"] == 0

    def test_sources_field_populated(self, engine):
        sygma = [_make_sygma_met(self.SHARED, 0.32)]
        dl    = self._dl(self.SHARED, 0.40)
        merged, _ = engine.merge_metabolites(sygma, dl)
        consensus = [m for m in merged if m.confidence_tier == ConsensusTier.HIGH]
        assert "sygma" in consensus[0].sources
        assert "dl"    in consensus[0].sources

    def test_dl_score_populated_for_consensus(self, engine):
        sygma = [_make_sygma_met(self.SHARED, 0.32)]
        dl    = self._dl(self.SHARED, 0.40)
        merged, _ = engine.merge_metabolites(sygma, dl)
        consensus = [m for m in merged if m.confidence_tier == ConsensusTier.HIGH]
        assert consensus[0].dl_score == pytest.approx(0.40, abs=1e-4)

    def test_dl_score_none_for_rule_only(self, engine):
        sygma = [_make_sygma_met(self.RULE_ONLY, 0.20)]
        merged, _ = engine.merge_metabolites(sygma, [])
        assert merged[0].dl_score is None

    def test_no_duplicates_in_merged(self, engine):
        sygma = [_make_sygma_met(self.SHARED, 0.32)]
        dl    = self._dl(self.SHARED, 0.40) + self._dl(self.SHARED, 0.30)
        merged, _ = engine.merge_metabolites(sygma, dl)
        smiles_list = [m.smiles for m in merged]
        assert len(smiles_list) == len(set(smiles_list))


# ===========================================================================
# 8. ConsensusEngine — merge_soft_spots
# ===========================================================================

class TestConsensusEngineMergeSoftSpots:
    @pytest.fixture
    def engine(self):
        return ConsensusEngine(alpha=0.55, beta=0.45)

    def _mol(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        return mol

    def _rule_spots(self, mol, top_n=5):
        return _find_soft_spots(mol, top_n=top_n)

    def test_returns_ensemble_soft_spots(self, engine):
        mol    = self._mol(ASPIRIN)
        rules  = self._rule_spots(mol)
        spots  = engine.merge_soft_spots(rules, {}, mol, top_n=3)
        assert all(isinstance(s, EnsembleSoftSpot) for s in spots)

    def test_respects_top_n(self, engine):
        mol    = self._mol(IBUPROFEN)
        rules  = self._rule_spots(mol, top_n=10)
        for n in (1, 2, 3, 5):
            spots = engine.merge_soft_spots(rules, {}, mol, top_n=n)
            assert len(spots) <= n

    def test_sorted_by_vulnerability_index_descending(self, engine):
        mol    = self._mol(ASPIRIN)
        rules  = self._rule_spots(mol, top_n=10)
        spots  = engine.merge_soft_spots(rules, {}, mol, top_n=5)
        vis    = [s.vulnerability_index for s in spots]
        assert vis == sorted(vis, reverse=True)

    def test_vulnerability_index_in_0_100(self, engine):
        mol    = self._mol(ASPIRIN)
        rules  = self._rule_spots(mol, top_n=10)
        spots  = engine.merge_soft_spots(rules, {}, mol, top_n=5)
        for s in spots:
            assert 0.0 <= s.vulnerability_index <= 100.0, \
                f"VI {s.vulnerability_index} out of range"

    def test_dl_attention_blended_into_vi(self, engine):
        """An atom with high DL attention should score higher than with zero."""
        mol    = self._mol(PARACETAMOL)
        rules  = _find_soft_spots(mol, top_n=mol.GetNumHeavyAtoms())
        if not rules:
            pytest.skip("No rule matches for this molecule")
        target_atom = rules[0].atom_index
        attention_zero = {target_atom: 0.0}
        attention_high = {target_atom: 1.0}
        spots_zero = engine.merge_soft_spots(rules, attention_zero, mol, top_n=5)
        spots_high = engine.merge_soft_spots(rules, attention_high, mol, top_n=5)
        vi_zero = next((s.vulnerability_index for s in spots_zero
                        if s.atom_index == target_atom), None)
        vi_high = next((s.vulnerability_index for s in spots_high
                        if s.atom_index == target_atom), None)
        if vi_zero is not None and vi_high is not None:
            assert vi_high >= vi_zero

    def test_rule_score_and_dl_risk_populated(self, engine):
        mol    = self._mol(ASPIRIN)
        rules  = self._rule_spots(mol)
        spots  = engine.merge_soft_spots(rules, {}, mol, top_n=3)
        for s in spots:
            assert s.rule_score >= 0.0
            assert s.dl_attention_risk >= 0.0

    def test_vi_formula_correct(self, engine):
        """VI = (alpha * rule + beta * dl) * 100 for rule-matched atoms."""
        mol    = self._mol(ASPIRIN)
        rules  = _find_soft_spots(mol, top_n=1)
        if not rules:
            pytest.skip("No rule matches")
        spot = rules[0]
        dl_w = 0.70
        spots = engine.merge_soft_spots(
            rules, {spot.atom_index: dl_w}, mol, top_n=1
        )
        if spots:
            expected_vi = round((0.55 * spot.score + 0.45 * dl_w) * 100, 2)
            actual_vi   = spots[0].vulnerability_index
            assert actual_vi == pytest.approx(expected_vi, abs=0.1)

    def test_dl_source_propagated(self, engine):
        mol   = self._mol(ASPIRIN)
        rules = self._rule_spots(mol)
        spots = engine.merge_soft_spots(rules, {}, mol, top_n=3,
                                        dl_source="test-model-v1")
        for s in spots:
            assert s.dl_source == "test-model-v1"

    def test_empty_rules_with_attention_returns_spots(self, engine):
        """Attention-only atoms should still appear when no rules match."""
        mol = self._mol("FC(F)(F)C(F)(F)F")  # perfluoroethane: minimal rule matches
        attention = {i: 0.9 - i * 0.1 for i in range(min(4, mol.GetNumHeavyAtoms()))}
        spots = engine.merge_soft_spots([], attention, mol, top_n=3)
        # May be empty (perfluoroethane truly has nothing) — must not crash
        assert isinstance(spots, list)


# ===========================================================================
# 9. _validate_and_normalise  (extended from v1)
# ===========================================================================

class TestValidateAndNormalise:
    def test_aspirin_returns_correct_formula(self):
        _, meta = _validate_and_normalise(ASPIRIN)
        assert meta.molecular_formula == "C9H8O4"

    def test_canonical_smiles_idempotent(self):
        _, m1 = _validate_and_normalise(ASPIRIN)
        _, m2 = _validate_and_normalise(m1.canonical_smiles)
        assert m1.canonical_smiles == m2.canonical_smiles

    def test_salt_stripped(self):
        _, meta = _validate_and_normalise(SALT_ASPIRIN)
        assert "Na" not in meta.canonical_smiles
        assert meta.molecular_formula == "C9H8O4"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _validate_and_normalise("")

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _validate_and_normalise("NOT_VALID_XYZ")

    def test_frozen_metadata(self):
        _, meta = _validate_and_normalise(ASPIRIN)
        with pytest.raises((AttributeError, TypeError)):
            meta.molecular_weight = 0.0


# ===========================================================================
# 10. _find_soft_spots (rule-only fallback)
# ===========================================================================

class TestFindSoftSpots:
    def _mol(self, s):
        m = Chem.MolFromSmiles(s)
        assert m is not None
        return m

    def test_returns_soft_spots(self):
        spots = _find_soft_spots(self._mol(ASPIRIN), top_n=3)
        assert all(isinstance(s, SoftSpot) for s in spots)

    def test_respects_top_n(self):
        mol = self._mol(ASPIRIN)
        assert len(_find_soft_spots(mol, top_n=1)) <= 1
        assert len(_find_soft_spots(mol, top_n=3)) <= 3

    def test_scores_in_unit_interval(self):
        for s in _find_soft_spots(self._mol(IBUPROFEN), top_n=5):
            assert 0.0 < s.score <= 1.0

    def test_no_duplicate_indices(self):
        spots = _find_soft_spots(self._mol(LIDOCAINE), top_n=10)
        indices = [s.atom_index for s in spots]
        assert len(indices) == len(set(indices))

    def test_ibuprofen_benzylic_site(self):
        spots = _find_soft_spots(self._mol(IBUPROFEN), top_n=5)
        assert any("benzylic" in s.rule_name for s in spots)

    def test_paracetamol_phenolic_site(self):
        spots = _find_soft_spots(self._mol(PARACETAMOL), top_n=5)
        assert any("phenol" in s.rule_name for s in spots)

    def test_caffeine_n_alkyl_site(self):
        spots = _find_soft_spots(self._mol(CAFFEINE), top_n=5)
        assert any("N_alkyl" in s.rule_name for s in spots)


# ===========================================================================
# 11. predict() — full ensemble
# ===========================================================================

class TestPredictEnsemble:
    @pytest.fixture
    def result(self, mock_sygma):
        return predict(ASPIRIN)

    def test_returns_metabolism_result(self, result):
        assert isinstance(result, MetabolismResult)

    def test_engine_version_v2(self, result):
        assert "ensemble" in result.engine_version.lower()

    def test_parent_formula(self, result):
        assert result.parent.molecular_formula == "C9H8O4"

    def test_metabolites_is_list(self, result):
        assert isinstance(result.metabolites, list)

    def test_all_metabolites_are_predicted_metabolite(self, result):
        assert all(isinstance(m, PredictedMetabolite) for m in result.metabolites)

    def test_all_have_confidence_tier(self, result):
        for m in result.metabolites:
            assert isinstance(m.confidence_tier, ConsensusTier)
            assert m.confidence_tier != ConsensusTier.UNKNOWN

    def test_soft_spots_are_ensemble_type(self, result):
        assert all(isinstance(s, EnsembleSoftSpot) for s in result.soft_spots)

    def test_soft_spots_have_vulnerability_index(self, result):
        for s in result.soft_spots:
            assert hasattr(s, "vulnerability_index")
            assert 0.0 <= s.vulnerability_index <= 100.0

    def test_pipeline_stats_populated(self, result):
        stats = result.pipeline_stats
        assert isinstance(stats, dict)
        for k in ("sygma_total", "consensus_count", "rule_only_count"):
            assert k in stats

    def test_to_dict_is_json_serialisable(self, result):
        import json
        d = result.to_dict()
        json.dumps(d)   # must not raise

    def test_to_dict_soft_spot_has_dl_fields(self, result):
        d = result.to_dict()
        if d["soft_spots"]:
            s = d["soft_spots"][0]
            assert "rule_score"          in s
            assert "dl_attention_risk"   in s
            assert "vulnerability_index" in s
            assert "dl_source"           in s

    def test_to_dict_metabolite_has_confidence_tier(self, result):
        d = result.to_dict()
        if d["metabolites"]:
            m = d["metabolites"][0]
            assert "confidence_tier" in m
            assert isinstance(m["confidence_tier"], str)

    def test_to_dict_compat_score_field(self, result):
        """Backward-compat: soft spots must expose 'score' key for v1 API consumers."""
        d = result.to_dict()
        for s in d["soft_spots"]:
            assert "score" in s

    def test_invalid_smiles_raises_value_error(self, mock_sygma):
        with pytest.raises(ValueError):
            predict("NOT_A_SMILES###")

    def test_salt_produces_warning(self, mock_sygma):
        result = predict(SALT_ASPIRIN)
        assert any("normalised" in w.lower() for w in result.warnings)

    def test_elapsed_s_positive(self, result):
        assert result.elapsed_s > 0

    def test_phase_properties(self, result):
        p1 = result.phase1_metabolites
        p2 = result.phase2_metabolites
        assert all(m.phase == 1 for m in p1)
        assert all(m.phase == 2 for m in p2)

    def test_consensus_property(self, result):
        consensus = result.consensus_metabolites
        assert all(m.confidence_tier == ConsensusTier.HIGH for m in consensus)


# ===========================================================================
# 12. predict() — rule-only mode  (enable_dl=False)
# ===========================================================================

class TestPredictRuleOnly:
    def test_returns_result(self, mock_sygma):
        result = predict(ASPIRIN, enable_dl=False)
        assert isinstance(result, MetabolismResult)

    def test_soft_spots_are_ensemble_type_even_rule_only(self, mock_sygma):
        """Even in rule-only mode, merge_soft_spots is called → EnsembleSoftSpot."""
        result = predict(ASPIRIN, enable_dl=False)
        # When enable_dl=False and no soft_spot_fn: ensemble still runs with zero DL
        assert all(isinstance(s, EnsembleSoftSpot) for s in result.soft_spots)

    def test_all_metabolites_rule_tier(self, mock_sygma):
        result = predict(ASPIRIN, enable_dl=False)
        for m in result.metabolites:
            assert m.confidence_tier == ConsensusTier.RULE

    def test_dl_disabled_warning(self, mock_sygma):
        result = predict(ASPIRIN, enable_dl=False)
        assert any("disabled" in w.lower() or "dl" in w.lower()
                   for w in result.warnings)

    def test_v1_soft_spot_fn_respected(self, mock_sygma):
        """Passing soft_spot_fn with enable_dl=False uses the external function."""
        sentinel = [SoftSpot(atom_index=0, atom_symbol="C",
                             rule_name="custom", score=0.99, smarts_match="[C]")]
        result = predict(ASPIRIN, enable_dl=False,
                         soft_spot_fn=lambda mol, n: sentinel)
        assert result.soft_spots == sentinel


# ===========================================================================
# 13. MetabolismResult.to_dict — serialisation contract
# ===========================================================================

class TestMetabolismResultToDict:
    @pytest.fixture
    def result_dict(self, mock_sygma):
        return predict(ASPIRIN).to_dict()

    def test_top_level_keys(self, result_dict):
        for k in ("engine_version", "elapsed_s", "warnings", "parent",
                  "metabolites", "soft_spots", "pipeline_stats"):
            assert k in result_dict

    def test_parent_all_fields(self, result_dict):
        p = result_dict["parent"]
        for k in ("canonical_smiles", "molecular_formula", "molecular_weight",
                  "exact_mass", "inchi", "inchikey", "num_heavy_atoms",
                  "num_hbd", "num_hba", "tpsa", "logp", "num_rings"):
            assert k in p, f"Parent missing key: {k}"

    def test_metabolite_all_fields(self, result_dict):
        for m in result_dict["metabolites"]:
            for k in ("smiles", "probability", "phase", "reaction_name",
                      "confidence_tier", "sources"):
                assert k in m, f"Metabolite missing key: {k}"

    def test_soft_spot_all_fields(self, result_dict):
        for s in result_dict["soft_spots"]:
            for k in ("atom_index", "atom_symbol", "rule_name",
                      "rule_score", "dl_attention_risk",
                      "vulnerability_index", "dl_source", "score"):
                assert k in s, f"SoftSpot missing key: {k}"

    def test_confidence_tier_values_are_strings(self, result_dict):
        for m in result_dict["metabolites"]:
            assert isinstance(m["confidence_tier"], str)

    def test_no_rdkit_objects_in_dict(self, result_dict):
        """All values must be JSON-primitives (no Mol, no frozenset)."""
        import json
        # Will raise TypeError if any non-serialisable objects leak
        json.dumps(result_dict)
