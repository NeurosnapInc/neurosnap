from __future__ import annotations

from typing import Iterator

from neurosnap.algos.wolfpsort import WoLFPSortPredictor, compute_features_dataframe, predict_localization


TEST_SEQUENCES = [
  (
    "RCC1_YEAST",
    (
      "MVKRTVATNGDASGAHRAKKMSKTHASHIINAQEDYKHMYLSVQPLDIFCWGTGSMCELG"
      "LGPLAKNKEVKRPRLNPFLPRDEAKIISFAVGGMHTLALDEESNVWSWGCNDVGALGRDT"
      "SNAKEQLKDMDADDSSDDEDGDLNELESTPAKIPRESFPPLAEGHKVVQLAATDNMSCAL"
      "FSNGEVYAWGTFRCNEGILGFYQDKIKIQKTPWKVPTFSKYNIVQLAPGKDHILFLDEEG"
      "MVFAWGNGQQNQLGRKVMERFRLKTLDPRPFGLRHVKYIASGENHCFALTKDNKLVSWGL"
      "NQFGQCGVSEDVEDGALVTKPKRLALPDNVVIRSIAAGEHHSLILSQDGDLYSCGRLDMF"
      "EVGIPKDNLPEYTYKDVHGKARAVPLPTKLNNVPKFKSVAAGSHHSVAVAQNGIAYSWGF"
      "GETYAVGLGPFEDDTEVPTRIKNTATQDHNIILVGCGGQFSVSGGVKLSDEDAEKRADEM"
      "DD"
    ),
  ),
  (
    "RCL1_YEAST",
    (
      "MSSSAPKYTTFQGSQNFRLRIVLATLSGKPIKIEKIRSGDLNPGLKDYEVSFLRLIESVT"
      "NGSVIEISYTGTTVIYRPGIIVGGASTHICPSSKPVGYFVEPMLYLAPFSKKKFSILFKG"
      "ITASHNDAGIEAIKWGLMPVMEKFGVRECALHTLKRGSPPLGGGEVHLVVDSLIAQPITM"
      "HEIDRPIISSITGVAYSTRVSPSLVNRMIDGAKKVLKNLQCEVNITADVWRGENSGKSPG"
      "WGITLVAQSKQKGWSYFAEDIGDAGSIPEELGEKVACQLLEEISKSAAVGRNQLPLAIVY"
      "MVIGKEDIGRLRINKEQIDERFIILLRDIKKIFNTEVFLKPVDEADNEDMIATIKGIGFT"
      "NTSKKIA"
    ),
  ),
  (
    "RT04_YEAST",
    (
      "MQRHVFARNFRRLSLLRNPSLTKRFQSSASGAANTPNNNDEVMLLQQKLLYDEIRSELKS"
      "LSQVPEDEILPELKKSLEQDKLSDKEQQLEAELSDFFRNYALLNKLFDSKTLDGQSSTTT"
      "AAATPTKPYPNLIPSANDKPYSSQELFLRQLNHSMRTAKLGATISKVYYPHKDIFYPPLP"
      "ENITVESLMSAGVHLGQSTSLWRSSTQSYIYGEYKGIHIIDLNQTLSYLKRAAKVVEGVS"
      "ESGGIILFLGTRQGQKRGLEEAAKKTHGYYVSTRWIPGTLTNSTEISGIWEKQEIDSNDN"
      "PTERALSPNETSKQVKPDLLVVLNPTENRNALLEAIKSRVPTIAIIDTDSEPSLVTYPIP"
      "GNDDSLRSVNFLLGVLARAGQRGLQNRLARNNEK"
    ),
  ),
  (
    "EF1A_ASHGO",
    (
      "MGKEKTHVNVVVIGHVDSGKSTTTGHLIYKCGGIDKRTIEKFEKEAAELGKGSFKYAWVL"
      "DKLKAERERGITIDIALWKFETPKYHVTVIDAPGHRDFIKNMITGTSQADCAILIIAGGV"
      "GEFEAGISKDGQTREHALLAYTLGVKQLIVAINKMDSVKWDESRYQEIVKETSNFIKKVG"
      "YNPKTVPFVPISGWNGDNMIEATTNAPWYKGWEKETKAGAVKGKTLLEAIDAIEPPVRPT"
      "DKALRLPLQDVYKIGGIGTVPVGRVETGVIKPGMVVTFAPSGVTTEVKSVEMHHEQLEEG"
      "VPGDNVGFNVKNVSVKEIRRGNVCGDSKNDPPKAAESFNATVIVLNHPGQISAGYSPVLD"
      "CHTAHIACKFDELLEKNDRRTGKKLEDSPKFLKAGDAAMVKFVPSKPMCVEAFTDYPPLG"
      "RFAVRDMRQTVAVGVIKSVVKSDKAGKVTKAAQKAGKK"
    ),
  ),
]

EXPECTED_FEATURE_ROWS = """
RCC1_YEAST 0.0788381742738589 0.0394190871369295 0.0497925311203319 0.0767634854771784 0.0186721991701245 0.033195020746888 0.0601659751037344 0.0912863070539419 0.0311203319502075 0.04149377593361 0.0829875518672199 0.0726141078838174 0.0228215767634855 0.0394190871369295 0.0477178423236514 0.0622406639004149 0.0435684647302905 0.016597510373444 0.0228215767634855 0.0684647302904564 482 0 3.60517647058823 5 0 0 3 0 1 0 -8.09 0 0 0 0 0 0 0 1 -3.3 -2.9 3 27 -3.04593201751494 0 0.0278 0 -4.4 0 0 0 0 0 0 0 0
RCL1_YEAST 0.0599455040871935 0.0435967302452316 0.0354223433242507 0.0408719346049046 0.0108991825613079 0.0245231607629428 0.0653950953678474 0.0899182561307902 0.0136239782016349 0.108991825613079 0.0790190735694823 0.0790190735694823 0.0217983651226158 0.0354223433242507 0.0490463215258856 0.0844686648501362 0.0517711171662125 0.0108991825613079 0.0245231607629428 0.0708446866485014 367 0 2.75623529411764 3 0 0 0 0 2 0 -7.89 0 0 0 0 0 0 0 1 13.8 4.19999999999999 2 30 -1.70986445765632 0 -0.129 0 -4.4 0 0 0 0 0 0 0 0
RT04_YEAST 0.0609137055837563 0.0583756345177665 0.0634517766497462 0.0456852791878173 0 0.050761421319797 0.0685279187817259 0.050761421319797 0.0152284263959391 0.0532994923857868 0.124365482233503 0.0634517766497462 0.0101522842639594 0.0253807106598985 0.0558375634517767 0.101522842639594 0.065989847715736 0.00761421319796954 0.0355329949238579 0.0431472081218274 394 0 3.07458823529411 7 0 0 0 0 1 0 -7.12 0 0 0 0 0 0 0 1 5.2 0.999999999999998 3 34 3.99438955381604 0 -0.4738 0 -4.4 1 2 0 0 0 0 0 0
EF1A_ASHGO 0.0807860262008734 0.0349344978165939 0.0349344978165939 0.0545851528384279 0.0131004366812227 0.0196506550218341 0.0676855895196507 0.0938864628820961 0.0240174672489083 0.0655021834061135 0.0502183406113537 0.111353711790393 0.0196506550218341 0.0327510917030568 0.0502183406113537 0.0436681222707424 0.0655021834061135 0.0131004366812227 0.0218340611353712 0.102620087336245 458 0 -0.639529411764711 1 1 0 0 0 2 0 -10.53 0 0 0 0 0 0 0 1 13.7 14.5 1 0 -7.67951670943444 0 -0.4738 0 -4.4 0 0 0 0 0 0 0 0
""".strip()

EXPECTED_REGRESSION_SCORES = {
  "fungi": {
    "RCC1_YEAST": {"nucl": 12.5, "mito_nucl": 11.5, "mito": 9.5, "cyto": 3.0},
    "RCL1_YEAST": {"mito": 15.5, "cyto_mito": 12.833, "cyto": 9.0, "cyto_nucl": 5.833},
    "RT04_YEAST": {"mito": 16.0, "nucl": 8.0, "cyto": 3.0},
    "EF1A_ASHGO": {"cyto": 27.0},
  },
  "animal": {
    "RCC1_YEAST": {"nucl": 30.5, "cyto_nucl": 17.5},
    "RCL1_YEAST": {"mito": 22.5, "cyto_mito": 15.0, "cyto": 6.5},
    "RT04_YEAST": {"cyto": 15.0, "mito": 15.0, "cyto_mito": 15.0},
    "EF1A_ASHGO": {"cyto": 27.0, "mito": 4.0},
  },
  "plant": {
    "RCC1_YEAST": {"chlo": 4.0, "nucl": 4.0, "mito": 4.0, "chlo_mito": 4.0},
    "RCL1_YEAST": {"chlo": 7.0, "cyto": 2.0, "mito": 2.0, "nucl": 1.0},
    "RT04_YEAST": {"mito": 7.0, "chlo": 3.0, "nucl": 3.0},
    "EF1A_ASHGO": {"cyto": 14.0},
  },
}


def _load_test_sequences() -> list[tuple[str, str]]:
  return list(TEST_SEQUENCES)


def _test_sequence_iterator() -> Iterator[tuple[str, str]]:
  return iter(_load_test_sequences())


def _parse_expected_features() -> dict[str, list[float]]:
  expected = {}
  for line in EXPECTED_FEATURE_ROWS.splitlines():
    parts = line.split()
    expected[parts[0]] = [float(value) for value in parts[1:]]
  return expected


def test_feature_dataframe_matches_reference_rows():
  predictor = WoLFPSortPredictor("fungi")
  df = predictor.compute_features_dataframe(_test_sequence_iterator())
  expected = _parse_expected_features()
  for _, row in df.iterrows():
    actual_values = [float(row[column]) for column in df.columns[1:]]
    expected_values = expected[row["id"]]
    assert len(actual_values) == len(expected_values)
    for actual, target in zip(actual_values, expected_values):
      assert abs(actual - target) < 1e-12


def test_prediction_scores_match_published_regression_examples():
  for organism, expected_rows in EXPECTED_REGRESSION_SCORES.items():
    predictor = WoLFPSortPredictor(organism)
    predictions = {row["id"]: row for row in predictor.predict(_test_sequence_iterator())}
    for seq_id, expected_scores in expected_rows.items():
      scores = predictions[seq_id]["scores_by_class"]
      for class_name, expected_score in expected_scores.items():
        assert round(scores[class_name], 3) == round(expected_score, 3)


def test_public_dataframe_api_returns_structured_scores():
  df = predict_localization(_test_sequence_iterator(), organism_type="fungi", as_dataframe=True)
  assert set(["id", "predicted_class", "predicted_label", "top_class", "top_label", "top_score", "scores_by_class", "scores_by_label"]).issubset(
    df.columns
  )
  features_df = compute_features_dataframe(_test_sequence_iterator())
  assert len(df) == len(features_df) == 4


def test_public_prediction_dict_output_is_minimal_by_default():
  rows = predict_localization(_test_sequence_iterator(), organism_type="fungi", as_dataframe=False)
  assert set(rows[0]) == {
    "id",
    "predicted_class",
    "predicted_label",
    "ranked_classes",
    "ranked_labels",
    "scores_by_class",
    "scores_by_label",
    "best_overall_k",
  }
  assert "description" not in rows[0]


def test_prediction_output_includes_human_readable_labels():
  row = predict_localization(_test_sequence_iterator(), organism_type="fungi", as_dataframe=False)[0]
  assert row["predicted_class"] == "nucl"
  assert row["predicted_label"] == "Nuclear"
  assert row["ranked_labels"][1][0] == "Mitochondrial / Nuclear"
  assert row["scores_by_label"]["Nuclear"] == row["scores_by_class"]["nucl"]
