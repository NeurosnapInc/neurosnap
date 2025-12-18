from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from neurosnap import conformers


@pytest.mark.slow
def test_generate_writes_clustered_conformers(tmp_path: Path):
  output_base = tmp_path / "aspirin_confs"

  df = conformers.generate(
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    output_name=str(output_base),
    num_confs=500,
    min_method=None,
  )

  # basic shape/fields
  assert isinstance(df, pd.DataFrame)
  assert not df.empty
  assert {"conformer_id", "cluster_id", "energy", "cluster_size"} <= set(df.columns)
  assert df["cluster_size"].gt(0).all()
  # energy minimization disabled → energy column should be empty/NaN
  assert df["energy"].isna().all()

  # outputs written per conformer
  assert output_base.exists() and output_base.is_dir()
  sdf_files = list(output_base.glob("conformer_*.sdf"))
  assert len(sdf_files) == len(df["conformer_id"])
  for cid in df["conformer_id"]:
    assert (output_base / f"conformer_{cid}.sdf").exists()


@pytest.mark.slow
@pytest.mark.parametrize(
  "smiles,min_method",
  [
    ("CCO", "UFF"),
    ("c1ccccc1O", "MMFF94"),
    ("N[N+](N)N", "MMFF94s"),
  ],
)
def test_generate_with_force_fields(tmp_path: Path, smiles: str, min_method: str):
  out_dir = tmp_path / f"{min_method.lower()}_confs"

  df = conformers.generate(
    smiles,
    output_name=str(out_dir),
    num_confs=25,
    min_method=min_method,
  )

  assert not df.empty
  assert {"conformer_id", "cluster_id", "energy", "cluster_size"} <= set(df.columns)
  assert df["cluster_size"].gt(0).all()
  # energy minimization on -> energies should be finite numbers
  assert df["energy"].notna().all()
  assert np.isfinite(df["energy"]).all()

  assert out_dir.exists() and out_dir.is_dir()
  sdf_files = list(out_dir.glob("conformer_*.sdf"))
  assert len(sdf_files) == len(df["conformer_id"])
  for cid in df["conformer_id"]:
    assert (out_dir / f"conformer_{cid}.sdf").exists()
