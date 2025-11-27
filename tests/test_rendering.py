import numpy as np
from PIL import Image

from neurosnap.protein import Protein
from neurosnap.rendering import animate_frames, render_pseudo3D, render_protein_pseudo3D


def test_render_pseudo3D_creates_image():
  t = np.linspace(0, 4 * np.pi, 80)
  coords = np.stack([np.cos(t) * 5, np.sin(t) * 5, np.linspace(-3, 3, len(t))], axis=1)
  img = render_pseudo3D([coords], image_size=(160, 120), padding=5, upsample=2)
  assert isinstance(img, Image.Image)
  assert img.size == (160, 120)
  arr = np.asarray(img)
  assert arr.shape == (120, 160, 4)
  assert arr[..., :3].max() > arr[..., :3].min()

  mask = (arr[..., :3] < 250).any(-1)
  ys, xs = np.nonzero(mask)
  assert ys.min() > 0 and xs.min() > 0
  assert ys.max() < arr.shape[0] - 1 and xs.max() < arr.shape[1] - 1
  bbox_center = np.array([xs.mean(), ys.mean()])
  image_center = np.array([arr.shape[1] / 2, arr.shape[0] / 2])
  assert np.linalg.norm(bbox_center - image_center) < min(arr.shape[:2]) * 0.2


def test_render_protein_pseudo3D_with_radii(tmp_path):
  prot = Protein("tests/files/1nkp_mycmax.pdb")
  img = render_protein_pseudo3D(
    prot,
    style="plddt",  # uses bfactor bins internally
    use_radii=True,
    image_size=(320, 240),
    padding=10,
    upsample=2,
  )
  assert isinstance(img, Image.Image)
  out = tmp_path / "prot.png"
  img.save(out)
  assert out.exists() and out.stat().st_size > 0


def test_animate_frames_writes_gif(tmp_path):
  frames = [
    np.zeros((20, 20, 3), dtype=np.uint8),
    np.ones((20, 20, 3), dtype=np.uint8) * 128,
    np.ones((20, 20, 3), dtype=np.uint8) * 255,
  ]
  output = tmp_path / "anim.gif"
  animate_frames(frames, output, title="t", subtitles=["a", "b", "c"], interval=50, repeat=False)
  assert output.exists()
  assert output.stat().st_size > 0
