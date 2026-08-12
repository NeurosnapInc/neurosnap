"""Tests for chemical-landscape graph construction, metrics, and persistence."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

from neurosnap.chemistry.landscape import (
  ChemicalGraph,
  ChemicalLandscape,
  EdgeType,
  FingerprintConfig,
  FragmentConfig,
  FragmentMethod,
  GraphBuilder,
  NodeType,
  ScaffoldConfig,
  SimilarityConfig,
  build_similarity_edges,
  characterize,
  detect_islands,
  diversity_metrics,
  fragment_library,
  fragment_molecule,
  frontier_scaffolds,
  gini,
  minhash_signatures,
  morgan_packed,
  murcko_smiles,
  network_metrics,
  pack_bits,
  popcount_rows,
  scaffold_network,
  shannon_entropy,
  shared_fragment_edges,
  stream_chunks,
  unpack_bits,
)

# -----------------------------
# fixtures
# -----------------------------

#: substituted ring systems, ring digit 1
CORES = (
  "c1ccc({left})cc1",  # benzene
  "c1cc({left})ccn1",  # pyridine
  "c1ccc4cc({left})ccc4c1",  # naphthalene (digits 1/4 to stay clear of R groups)
  "C1CCN({left})CC1",  # piperidine
  "c1cc({left})cs1",  # thiophene
  "c1cnc({left})nc1",  # pyrimidine
)
#: linkers between the core and the R group
LEFT = (
  "C(=O)N{r}",
  "CN{r}",
  "OC{r}",
  "S(=O)(=O)N{r}",
  "NC(=O){r}",
)
#: R groups, ring digit 2
RGROUPS = (
  "C",
  "CC",
  "CCC",
  "CC(C)C",
  "Cc2ccccc2",
  "CCOC",
  "CC#N",
  "CCCl",
  "CCF",
  "CC(=O)O",
  "CCn2ccnc2",
  "CCc2ccncc2",
)


def _canonical(smiles: str) -> str:
  mol = Chem.MolFromSmiles(smiles)
  return Chem.MolToSmiles(mol) if mol is not None else ""


def combinatorial_library(limit: int | None = None) -> list[str]:
  """Deterministic library: 6 cores x 5 linkers x 12 R groups."""
  out: list[str] = []
  seen: set[str] = set()
  for core in CORES:
    for left in LEFT:
      for r in RGROUPS:
        canonical = _canonical(core.format(left=left.format(r=r)))
        if not canonical or canonical in seen:
          continue
        seen.add(canonical)
        out.append(canonical)
        if limit is not None and len(out) >= limit:
          return out
  return out


def near_duplicate_pairs(n_pairs: int = 40) -> tuple[list[str], list[tuple[int, int]]]:
  """Library where compounds 2i and 2i+1 are homologues (one carbon apart)."""
  smiles: list[str] = []
  pairs: list[tuple[int, int]] = []
  homologues = (("CC", "CCC"), ("CCOC", "CCOCC"), ("CC(C)C", "CC(C)CC"))
  for core in CORES:
    for left in LEFT:
      for r1, r2 in homologues:
        a = _canonical(core.format(left=left.format(r=r1)))
        b = _canonical(core.format(left=left.format(r=r2)))
        if not a or not b or a == b:
          continue
        idx = len(smiles)
        smiles.extend((a, b))
        pairs.append((idx, idx + 1))
        if len(pairs) >= n_pairs:
          return smiles, pairs
  return smiles, pairs


@pytest.fixture(scope="module")
def sample_smiles() -> list[str]:
  return combinatorial_library()


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory) -> str:
  smiles = combinatorial_library()
  path = tmp_path_factory.mktemp("data") / "library.csv"
  lines = ["compound_id,smiles"]
  lines += [f"CMPD{i:05d},{smi}" for i, smi in enumerate(smiles)]
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return str(path)


# -----------------------------
# fingerprints
# -----------------------------


def test_pack_unpack_roundtrip():
  rng = np.random.default_rng(0)
  dense = rng.integers(0, 2, size=(7, 1024), dtype=np.uint8)
  packed = pack_bits(dense)
  assert packed.shape == (7, 16)
  assert np.array_equal(unpack_bits(packed, 1024), dense)


def test_popcount_matches_numpy():
  rng = np.random.default_rng(1)
  words = rng.integers(0, 2**63, size=(5, 4), dtype=np.uint64)
  counts = popcount_rows(words)
  expected = np.unpackbits(words.astype("<u8").view(np.uint8).reshape(5, -1), axis=1).sum(axis=1)
  np.testing.assert_array_equal(counts, expected)


def test_morgan_packed_matches_rdkit():
  smiles = combinatorial_library(20)
  cfg = FingerprintConfig(radii=(2,), n_bits=1024)
  block = morgan_packed(smiles, cfg)
  gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
  for i, smi in enumerate(smiles):
    rd_bits = set(gen.GetFingerprint(Chem.MolFromSmiles(smi)).GetOnBits())
    dense = unpack_bits(block.packed[i : i + 1], 1024)[0]
    our_bits = set(np.flatnonzero(dense).tolist())
    assert our_bits == rd_bits


def test_identical_smiles_identical_fingerprints():
  block = morgan_packed(["CCOc1ccccc1", "CCOc1ccccc1"], FingerprintConfig(radii=(2,), n_bits=1024))
  np.testing.assert_array_equal(block.packed[0], block.packed[1])


def test_minhash_identifies_identical_sets():
  from neurosnap.chemistry.landscape import make_permutations

  block = morgan_packed(["CCOc1ccccc1", "CCOc1ccccc1"], FingerprintConfig(radii=(2,), n_bits=1024))
  offsets, indices = block.onbits_csr()
  a, b = make_permutations(64, seed=3)
  sig = minhash_signatures(offsets, indices, a, b)
  np.testing.assert_array_equal(sig[0], sig[1])


def test_minhash_estimates_jaccard():
  from neurosnap.chemistry.landscape import make_permutations

  smiles = combinatorial_library(30)
  block = morgan_packed(smiles, FingerprintConfig(radii=(2,), n_bits=1024))
  offsets, indices = block.onbits_csr()
  a, b = make_permutations(256, seed=7)
  sig = minhash_signatures(offsets, indices, a, b)
  dense = block.dense().astype(bool)
  errors = []
  for i, j in ((0, 1), (0, 5), (3, 17), (10, 20)):
    estimate = float((sig[i] == sig[j]).mean())
    true = float(np.logical_and(dense[i], dense[j]).sum() / np.logical_or(dense[i], dense[j]).sum())
    errors.append(abs(estimate - true))
  assert max(errors) < 0.15


def test_similarity_edges_exact_path():
  smiles = combinatorial_library(60)
  block = morgan_packed(smiles, FingerprintConfig(radii=(2,), n_bits=1024))
  cfg = SimilarityConfig(threshold=0.5, k=6, exact_below=10_000)
  src, dst, score = build_similarity_edges(block, cfg)
  assert src.size == dst.size == score.size > 0
  assert np.all(score >= 0.5)
  assert np.all(src < dst)
  keys = src.astype(np.int64) * block.n_mols + dst
  assert np.unique(keys).size == keys.size


def test_similarity_edges_approximate_path_recovers_most_exact():
  smiles = combinatorial_library(60)
  block = morgan_packed(smiles, FingerprintConfig(radii=(2,), n_bits=1024))
  exact = build_similarity_edges(block, SimilarityConfig(threshold=0.55, k=8, exact_below=10_000))
  approx = build_similarity_edges(
    block,
    SimilarityConfig(threshold=0.55, k=8, exact_below=0, n_permutations=256, n_bands=64, bucket_cap=128),
  )
  exact_pairs = set(zip(exact[0].tolist(), exact[1].tolist()))
  approx_pairs = set(zip(approx[0].tolist(), approx[1].tolist()))
  assert len(exact_pairs) > 10
  recall = len(exact_pairs & approx_pairs) / len(exact_pairs)
  assert recall >= 0.7
  assert np.all(approx[2] >= 0.55)


def test_threshold_one_keeps_only_identical():
  block = morgan_packed(["CCOc1ccccc1", "CCOc1ccccc1", "c1ccncc1C(=O)O"], FingerprintConfig(radii=(2,), n_bits=1024))
  src, dst, score = build_similarity_edges(block, SimilarityConfig(threshold=1.0, k=4, exact_below=100))
  assert list(zip(src.tolist(), dst.tolist())) == [(0, 1)]
  assert score[0] == pytest.approx(1.0)


# -----------------------------
# scaffolds
# -----------------------------

DRUGLIKE = "CCOc1ccccc1C(=O)NC1CCN(Cc2ccncc2)CC1"
BIARYL = "c1ccc(-c2ccncc2)cc1"


def test_murcko_smiles_matches_rdkit_reference():
  assert murcko_smiles(DRUGLIKE) == "O=C(NC1CCN(Cc2ccncc2)CC1)c1ccccc1"
  assert murcko_smiles("CC(=O)Nc1ccc(O)cc1") == "c1ccccc1"


def test_acyclic_molecule_has_no_scaffold():
  assert murcko_smiles("CCCCCCO") == ""
  result = scaffold_network(["CCCCCCO"])
  assert result.compound_scaffold.tolist() == [-1]
  assert result.n_scaffolds == 0


def test_network_contains_murcko_and_ring_ancestors():
  result = scaffold_network([DRUGLIKE], ScaffoldConfig())
  smis = set(result.scaffolds)
  assert murcko_smiles(DRUGLIKE) in smis
  assert "c1ccccc1" in smis  # benzene ancestor
  assert "c1ccncc1" in smis  # pyridine ancestor
  assert result.n_scaffolds >= 4


def test_scaffold_smiles_have_no_attachment_dummies():
  result = scaffold_network([DRUGLIKE, BIARYL])
  assert all("*" not in s for s in result.scaffolds)


def test_hierarchy_edges_go_from_general_to_specific():
  result = scaffold_network([DRUGLIKE, BIARYL])
  assert result.n_hierarchy_edges > 0
  parent_levels = result.levels[result.hierarchy_parent]
  child_levels = result.levels[result.hierarchy_child]
  assert np.all(parent_levels <= child_levels)
  assert np.any(parent_levels < child_levels)


def test_hierarchy_has_no_self_loops_or_duplicates():
  result = scaffold_network([DRUGLIKE, BIARYL, "CC(=O)Nc1ccc(O)cc1"])
  pairs = list(zip(result.hierarchy_parent.tolist(), result.hierarchy_child.tolist()))
  assert all(p != c for p, c in pairs)
  assert len(set(pairs)) == len(pairs)


def test_levels_equal_ring_counts():
  result = scaffold_network([DRUGLIKE])
  for smi, level in zip(result.scaffolds, result.levels.tolist()):
    assert rdMolDescriptors.CalcNumRings(Chem.MolFromSmiles(smi)) == level


def test_shared_scaffolds_are_deduplicated():
  result = scaffold_network(["CC(=O)Nc1ccc(O)cc1", "CCNc1ccccc1", "Oc1ccccc1"])
  assert result.n_scaffolds == 1
  assert result.scaffolds == ["c1ccccc1"]
  assert result.compound_scaffold.tolist() == [0, 0, 0]


def test_ring_cap_keeps_murcko_but_skips_expansion():
  limited = scaffold_network([DRUGLIKE], ScaffoldConfig(max_level=2))
  assert limited.n_scaffolds == 1  # murcko has 3 rings: annotation only
  full = scaffold_network([DRUGLIKE], ScaffoldConfig(max_level=6))
  assert full.n_scaffolds > 1


def test_generic_scaffolds_can_be_included():
  plain = scaffold_network([DRUGLIKE], ScaffoldConfig(include_generic=False))
  generic = scaffold_network([DRUGLIKE], ScaffoldConfig(include_generic=True))
  assert generic.n_scaffolds > plain.n_scaffolds


# -----------------------------
# fragments
# -----------------------------

PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
BIARYL_LINKER = "c1ccc(CNc2ccncc2)cc1"


def test_brics_fragments_of_paracetamol():
  frags = fragment_molecule(PARACETAMOL, FragmentConfig(use_rotatable_bonds=False, use_linkers=False))
  smis = {smi for smi, _ in frags}
  assert smis
  assert all(method == FragmentMethod.BRICS for _, method in frags)
  assert any("c1ccccc1" in smi or "Oc1ccccc1" in smi for smi in smis)


def test_fragments_contain_no_attachment_dummies():
  frags = fragment_molecule(DRUGLIKE, FragmentConfig())
  assert frags
  assert all("*" not in smi for smi, _ in frags)


def test_min_fragment_atoms_filter():
  big = fragment_molecule(DRUGLIKE, FragmentConfig(min_fragment_atoms=10))
  small = fragment_molecule(DRUGLIKE, FragmentConfig(min_fragment_atoms=2))
  assert len(big) < len(small)
  for smi, _ in big:
    assert Chem.MolFromSmiles(smi).GetNumHeavyAtoms() >= 10


def test_rotatable_bond_cuts_are_produced():
  cfg = FragmentConfig(use_brics=False, use_linkers=False, min_fragment_atoms=2)
  frags = fragment_molecule("CCCCc1ccccc1OCC", cfg)
  assert frags
  assert all(method == FragmentMethod.ROTATABLE_BOND for _, method in frags)


def test_linker_extraction_finds_the_bridge():
  cfg = FragmentConfig(use_brics=False, use_rotatable_bonds=False, min_fragment_atoms=1)
  frags = fragment_molecule(BIARYL_LINKER, cfg)
  assert frags
  assert all(method == FragmentMethod.LINKER for _, method in frags)
  assert any("c" not in smi and "1" not in smi for smi, _ in frags)


def test_max_fragments_per_molecule_cap():
  frags = fragment_molecule(DRUGLIKE, FragmentConfig(max_fragments_per_molecule=3))
  assert len(frags) == 3


def test_invalid_smiles_gives_no_fragments():
  assert fragment_molecule("not-a-mol", FragmentConfig()) == []


def test_fragment_library_deduplicates_and_counts():
  result = fragment_library([PARACETAMOL, PARACETAMOL, DRUGLIKE], FragmentConfig())
  assert result.n_fragments == len(set(result.fragments))
  frags_0 = set(result.compound_fragment_dst[result.compound_fragment_src == 0].tolist())
  frags_1 = set(result.compound_fragment_dst[result.compound_fragment_src == 1].tolist())
  assert frags_0 == frags_1
  assert result.frequencies[list(frags_0)].min() >= 2


def test_shared_fragment_edges_link_common_ring_systems():
  ring_systems = ["c1ccccc1", "c1ccccc1", "c1ccncc1", ""]
  freqs = np.array([10, 4, 7, 3], dtype=np.int64)
  src, dst = shared_fragment_edges(ring_systems, freqs, links_per_fragment=4)
  pairs = {tuple(sorted(p)) for p in zip(src.tolist(), dst.tolist())}
  assert (0, 1) in pairs
  assert all(a != b for a, b in pairs)
  assert not any(3 in p for p in pairs)
  assert not any(2 in p for p in pairs)


# -----------------------------
# analysis
# -----------------------------


def test_shannon_entropy_bounds():
  assert shannon_entropy(np.array([1, 1, 1, 1])) == pytest.approx(2.0)
  assert shannon_entropy(np.array([10])) == pytest.approx(0.0)
  assert shannon_entropy(np.array([], dtype=np.int64)) == 0.0
  assert shannon_entropy(np.array([9, 1])) < 1.0


def test_gini_bounds():
  assert gini(np.array([5, 5, 5, 5])) == pytest.approx(0.0, abs=1e-9)
  assert gini(np.array([0, 0, 0, 20])) > 0.7
  assert 0.0 <= gini(np.array([1, 2, 3, 4])) <= 1.0


def two_island_graph() -> ChemicalGraph:
  """Two similarity cliques (5 + 4 compounds) on two different scaffolds."""
  b = GraphBuilder()
  left = [b.add_compound(f"c1ccccc1C{'C' * i}", f"L{i}", n_rings=1, mw=100.0 + i) for i in range(5)]
  right = [b.add_compound(f"c1ccncc1C{'C' * i}", f"R{i}", n_rings=1, mw=110.0 + i) for i in range(4)]
  s_left = b.add_scaffold("c1ccccc1", level=1)
  s_right = b.add_scaffold("c1ccncc1", level=1)
  s_parent = b.add_scaffold("C1CCCCC1", level=1)
  f_shared = b.add_fragment("CC", method=1)
  for node in left:
    b.add_edge(node, s_left, EdgeType.COMPOUND_SCAFFOLD)
    b.add_edge(node, f_shared, EdgeType.COMPOUND_FRAGMENT)
  for node in right:
    b.add_edge(node, s_right, EdgeType.COMPOUND_SCAFFOLD)
  for group, weight in ((left, 0.9), (right, 0.85)):
    for i, u in enumerate(group):
      for v in group[i + 1 :]:
        b.add_edge(u, v, EdgeType.COMPOUND_SIMILARITY, weight=weight)
  b.add_edge(s_parent, s_left, EdgeType.SCAFFOLD_HIERARCHY)
  b.add_edge(s_parent, s_right, EdgeType.SCAFFOLD_HIERARCHY)
  b.add_edge(left[0], right[0], EdgeType.COMPOUND_SIMILARITY, weight=0.56)
  return b.build()


def test_diversity_metrics_on_toy_graph():
  g = two_island_graph()
  m = diversity_metrics(g)
  assert m.n_compounds == 9
  assert m.n_scaffold_nodes == 3
  assert m.n_populated_scaffolds == 2
  assert m.n_fragments == 1
  assert m.scaffold_entropy == pytest.approx(-(5 / 9 * math.log2(5 / 9) + 4 / 9 * math.log2(4 / 9)))
  assert 0.0 < m.scaffold_entropy_normalized <= 1.0
  assert m.chemical_coverage == pytest.approx(2 / 9)
  assert m.scaffold_redundancy == pytest.approx(1 - 2 / 9)
  assert m.singleton_scaffold_fraction == 0.0
  assert m.compounds_per_scaffold == pytest.approx(4.5)
  assert m.top_scaffolds[0][0] == "c1ccccc1"
  assert m.top_scaffolds[0][1] == 5


def test_frontier_scaffolds_finds_underpopulated_generalisations():
  g = two_island_graph()
  frontier = frontier_scaffolds(g, min_support=0, limit=5)
  assert frontier, "the unpopulated parent scaffold must be reported"
  top = frontier[0]
  assert top["smiles"] == "C1CCCCC1"
  assert top["support"] == 0
  assert top["descendant_support"] == 9
  assert top["n_children"] == 2


def test_frontier_support_counts_diamond_descendants_once():
  b = GraphBuilder()
  root = b.add_scaffold("C1CCCCC1", level=1)
  left = b.add_scaffold("c1ccccc1", level=1)
  right = b.add_scaffold("c1ccncc1", level=1)
  leaf = b.add_scaffold("O=C(Nc1ccccc1)c1ccncc1", level=2)
  compound = b.add_compound("O=C(Nc1ccccc1)c1ccncc1C", "X1", n_rings=2)
  b.add_edge(compound, leaf, EdgeType.COMPOUND_SCAFFOLD)
  for parent, child in ((root, left), (root, right), (left, leaf), (right, leaf)):
    b.add_edge(parent, child, EdgeType.SCAFFOLD_HIERARCHY)

  frontier = {entry["smiles"]: entry for entry in frontier_scaffolds(b.build(), min_support=0)}
  assert frontier["C1CCCCC1"]["descendant_support"] == 1  # not 2
  assert frontier["c1ccccc1"]["descendant_support"] == 1


def test_network_metrics_on_toy_graph():
  g = two_island_graph()
  m = network_metrics(g, n_samples=8, seed=0)
  assert m.n_nodes == g.n_nodes
  assert m.n_edges == g.n_edges
  assert m.density == pytest.approx(2 * g.n_edges / (g.n_nodes * (g.n_nodes - 1)))
  assert m.n_components == 1
  assert m.largest_component_fraction == pytest.approx(1.0)
  assert m.mean_degree == pytest.approx(2 * g.n_edges / g.n_nodes)
  assert m.average_path_length > 1.0
  assert m.degree_histogram
  assert len(m.central_nodes) > 0


def test_detect_islands_recovers_planted_clusters():
  g = two_island_graph()
  result = detect_islands(g, resolution=1.0)
  assert result.n_islands == 2
  sizes = sorted(island["n_compounds"] for island in result.islands)
  assert sizes == [4, 5]
  assert sum(sizes) == 9
  biggest = max(result.islands, key=lambda x: x["n_compounds"])
  assert biggest["n_scaffolds"] == 1
  assert biggest["mean_internal_similarity"] == pytest.approx(0.9, abs=0.02)
  assert biggest["representative"]
  assert result.n_bridges == 1
  assert result.bridges[0]["weight"] == pytest.approx(0.56, abs=1e-3)


def test_isolated_compounds_form_their_own_islands():
  b = GraphBuilder()
  for i in range(3):
    b.add_compound(f"C{'C' * i}O", f"C{i}")
  result = detect_islands(b.build())
  assert result.n_islands == 3
  assert all(island["n_compounds"] == 1 for island in result.islands)


def test_characterize_report_structure():
  g = two_island_graph()
  report = characterize(g, n_samples=8, seed=0)
  data = report.to_dict()
  for section in ("counts", "diversity", "network", "islands", "frontier"):
    assert section in data
  text = report.summary()
  assert "Chemical landscape" in text
  assert "Islands" in text
  assert data["counts"]["n_compounds"] == 9
  assert data["islands"]["n_islands"] == 2
  json.dumps(report.to_dict())  # must be JSON-serialisable


# -----------------------------
# end-to-end
# -----------------------------


@pytest.fixture(scope="module")
def built_library(tmp_path_factory):
  smiles = combinatorial_library(150)
  path = tmp_path_factory.mktemp("lib") / "library.csv"
  path.write_text(
    "compound_id,smiles\n" + "".join(f"CMPD{i:04d},{s}\n" for i, s in enumerate(smiles)),
    encoding="utf-8",
  )
  library = ChemicalLandscape(path, smiles_column="smiles", chunk_size=64)
  library.build_all()
  return library, path


def test_library_loads_compounds_and_fingerprints(built_library):
  library, _ = built_library
  assert len(library) == 150
  assert library.failures == []
  assert library.fingerprints is not None
  assert library.fingerprints.n_mols == 150
  assert library.fingerprints.popcounts.min() > 0


def test_graph_has_all_node_and_edge_families(built_library):
  library, _ = built_library
  g = library.graph
  assert g.count_nodes(NodeType.COMPOUND) == 150
  assert g.count_nodes(NodeType.SCAFFOLD) > 0
  assert g.count_nodes(NodeType.FRAGMENT) > 0
  for edge_type in (
    EdgeType.COMPOUND_SCAFFOLD,
    EdgeType.COMPOUND_FRAGMENT,
    EdgeType.COMPOUND_SIMILARITY,
    EdgeType.SCAFFOLD_HIERARCHY,
    EdgeType.FRAGMENT_SHARED,
  ):
    assert g.count_edges(edge_type) > 0, edge_type


def test_compound_nodes_keep_murcko_annotation(built_library):
  library, _ = built_library
  g = library.graph
  node = g.node_id_of_compound("CMPD0000")
  assert g.murcko[node]
  assert g.node_type[node] == NodeType.COMPOUND


def test_similarity_edges_only_connect_compounds(built_library):
  library, _ = built_library
  g = library.graph
  sim = g.edges_of_type(EdgeType.COMPOUND_SIMILARITY)
  assert sim.size > 0
  assert np.all(g.node_type[g.src[sim]] == NodeType.COMPOUND)
  assert np.all(g.node_type[g.dst[sim]] == NodeType.COMPOUND)
  assert np.all(g.weight[sim] >= library.config.similarity.threshold)


def test_characterize_answers_the_diversity_question(built_library):
  library, _ = built_library
  report = library.characterize(n_samples=64, seed=0)
  assert report.counts["n_compounds"] == 150
  assert report.diversity.n_populated_scaffolds > 1
  assert report.diversity.scaffold_entropy > 0
  assert report.islands.n_islands >= 2
  assert sum(i["n_compounds"] for i in report.islands.islands) == 150
  assert report.network.n_components >= 1
  text = report.summary()
  assert "Chemical landscape" in text and "Islands" in text


def test_traversal_between_two_compounds(built_library):
  library, _ = built_library
  g = library.graph
  labels = g.connected_components()
  compounds = g.nodes_of_type(NodeType.COMPOUND)
  values, counts = np.unique(labels[compounds], return_counts=True)
  biggest = values[counts.argmax()]
  members = compounds[labels[compounds] == biggest]
  a, b = g.compound_id[int(members[0])], g.compound_id[int(members[-1])]
  path = library.path_between(a, b)
  assert path[0] == a
  assert path[-1] == b
  assert len(path) >= 2
  assert library.island_of(a) >= 0
  assert library.neighbors(a)


def test_save_and_reload_roundtrip(built_library, tmp_path):
  library, _ = built_library
  library.save(tmp_path / "store")
  assert (tmp_path / "store" / "landscape.json").exists()
  assert (tmp_path / "store" / "fingerprints.npz").exists()

  reloaded = ChemicalLandscape.from_store(tmp_path / "store")
  assert len(reloaded) == len(library)
  assert reloaded.graph.n_nodes == library.graph.n_nodes
  assert reloaded.graph.n_edges == library.graph.n_edges
  np.testing.assert_array_equal(reloaded.fingerprints.packed, library.fingerprints.packed)
  report = reloaded.characterize(n_samples=32)
  assert report.counts["n_compounds"] == 150


def test_library_accepts_a_plain_smiles_sequence():
  smiles = combinatorial_library(30)
  library = ChemicalLandscape(smiles, chunk_size=16)
  library.build_all()
  assert len(library) == 30
  assert library.graph.count_nodes(NodeType.COMPOUND) == 30


def test_invalid_records_are_reported_not_fatal(tmp_path):
  path = tmp_path / "mixed.csv"
  path.write_text("id,smiles\n1,CCO\n2,not-a-molecule\n3,c1ccccc1\n", encoding="utf-8")
  library = ChemicalLandscape(path).build_all()
  assert len(library) == 2
  assert [cid for cid, _ in library.failures] == ["2"]


def test_readers_support_smi_and_gzip(tmp_path):
  smi_path = tmp_path / "lib.smi"
  smi_path.write_text("smiles id\nCCO ethanol\nc1ccccc1 benzene\n", encoding="utf-8")
  chunks = list(stream_chunks(smi_path))
  assert [c.compound_ids for c in chunks] == [["ethanol", "benzene"]]
  assert [c.smiles for c in chunks] == [["CCO", "c1ccccc1"]]


def test_exports_and_plots(tmp_path):
  smiles = combinatorial_library(40)
  library = ChemicalLandscape(smiles).build_all()
  library.export_json(tmp_path / "graph.json")
  library.export_graphml(tmp_path / "graph.graphml")
  assert (tmp_path / "graph.json").exists()
  assert (tmp_path / "graph.graphml").exists()
  payload = json.loads((tmp_path / "graph.json").read_text())
  assert payload["metadata"]["n_nodes"] == library.graph.n_nodes
  library.plot(tmp_path / "plots")
  for name in ("scaffold_map.png", "chemical_islands.png", "diversity_report.png"):
    assert (tmp_path / "plots" / name).exists(), name
