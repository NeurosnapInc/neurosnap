"""
Provides functions and classes related to rendering, plotting, animating, and visualizing data.
"""

import colorsys
import io
import json
import os
import pathlib
import re
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import matplotlib
import matplotlib.animation as animation
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from Bio.PDB import PDBIO, MMCIFParser, NeighborSearch, PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Residue import Residue as ResidueType
from Bio.PDB.Superimposer import Superimposer
from matplotlib import collections as mcoll
from PIL import Image, ImageDraw
from rdkit import Chem
from requests_toolbelt.multipart.encoder import MultipartEncoder
from scipy.special import expit as sigmoid
from tqdm import tqdm

from neurosnap.log import logger


def draw_pseudo_3D(
  segments: Iterable[Union[np.ndarray, pd.DataFrame]],
  *,
  c: Optional[Iterable[np.ndarray]] = None,
  sizes: Optional[Iterable[np.ndarray]] = None,
  chainbreak: int = 5,
  cmap: str = "gist_rainbow",
  line_w: float = 2.0,
  cmin: Optional[float] = None,
  cmax: Optional[float] = None,
  shadow: float = 0.95,
  image_size: Tuple[int, int] = (800, 800),
  padding: int = 20,
  background_color: Tuple[int, int, int] = (255, 255, 255),
  upsample: int = 2,
) -> Image.Image:
  """Plot the famous Pseudo 3D projection of a protein using Pillow.

  Algorithm originally written By Dr. Sergey Ovchinnikov.
  Adapted from :func:`plot_pseudo_3D` to render into a Pillow Image instead of Matplotlib.

  Parameters:
    segments: Iterable of XYZ coordinates, where each element is a segment/molecule to draw separately
    c: Iterable of 1D arrays used to color the protein, aligned one-to-one with ``segments``; defaults to residue index
    sizes: Iterable of 1D arrays of radii/size values, aligned one-to-one with ``segments``; interpreted in the same units as coordinates
    chainbreak: Minimum distance in angstroms between chains / segments before being considered a chain break (int)
    cmap: Color map name or callable used for coloring the protein
    line_w: Line width (interpreted in data space; converted to pixels)
    cmin: Minimum value for coloring, automatically calculated if None
    cmax: Maximum value for coloring, automatically calculated if None
    shadow: Shadow intensity between 0 and 1 inclusive, lower numbers mean darker more intense shadows
    image_size: Final image size in pixels (width, height)
    padding: Padding in pixels around the drawing region
    background_color: Background RGB color
    upsample: Factor to draw at higher resolution and downsample for antialiasing

  Returns:
    Pillow Image containing the rendering

  """

  def rescale(a, amin=None, amax=None):
    a = np.copy(a)
    if amin is None:
      amin = a.min()
    if amax is None:
      amax = a.max()
    a[a < amin] = amin
    a[a > amax] = amax
    return (a - amin) / (amax - amin if amax != amin else 1)

  def make_colors(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    name = str(cmap).lower() if isinstance(cmap, str) else None
    if name == "gist_rainbow":
      hues = values * 0.75
      rgb = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues])
    elif name == "viridis":
      stops = (
        np.array(
          [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
          ],
          dtype=float,
        )
        / 255.0
      )
      pos = np.linspace(0, 1, len(stops))
      r = np.interp(values, pos, stops[:, 0])
      g = np.interp(values, pos, stops[:, 1])
      b = np.interp(values, pos, stops[:, 2])
      rgb = np.stack([r, g, b], axis=1)
    elif callable(cmap):
      out = np.asarray(cmap(values))
      if out.shape[-1] == 4:
        out = out[:, :3]
      rgb = np.clip(out, 0.0, 1.0)
    else:
      # fallback to a simple blue→red gradient
      rgb = np.stack([values, np.zeros_like(values), 1 - values], axis=1)
    alpha = np.ones((len(values), 1), dtype=float)
    return np.concatenate([rgb, alpha], axis=1)

  def _to_array(coords: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    arr = coords.to_numpy() if isinstance(coords, pd.DataFrame) else np.asarray(coords)
    if arr.ndim != 2 or arr.shape[1] < 3:
      raise ValueError("Each segment must be an array-like of shape (N, 3)")
    return arr[:, :3]

  if isinstance(segments, (np.ndarray, pd.DataFrame)):
    segment_list = [_to_array(segments)]
  else:
    segment_list = [_to_array(seg) for seg in segments]

  if len(segment_list) == 0:
    raise ValueError("No segments provided.")
  if any(len(seg) == 0 for seg in segment_list):
    raise ValueError("All provided segments must contain at least one coordinate.")

  all_xyz = np.concatenate(segment_list, 0)
  lengths = [len(seg) for seg in segment_list]

  c_segments: Optional[List[np.ndarray]] = None
  if c is not None:
    c_list = list(c)
    if len(c_list) != len(segment_list):
      raise ValueError("c must be an iterable aligning one-to-one with the provided segments.")
    c_segments = []
    for ci, seg_len in zip(c_list, lengths):
      ci_arr = np.asarray(ci)
      if ci_arr.ndim != 1 or len(ci_arr) != seg_len:
        raise ValueError("Each element of c must be 1D and match the length of its corresponding segment.")
      c_segments.append(ci_arr)

  size_segments: Optional[List[np.ndarray]] = None
  if sizes is not None:
    size_list = list(sizes)
    if len(size_list) != len(segment_list):
      raise ValueError("size must be an iterable aligning one-to-one with the provided segments.")
    size_segments = []
    for si, seg_len in zip(size_list, lengths):
      si_arr = np.asarray(si, dtype=float)
      if si_arr.ndim != 1 or len(si_arr) != seg_len:
        raise ValueError("Each element of size must be 1D and match the length of its corresponding segment.")
      size_segments.append(si_arr)

  # clip color values and produce warning if necessary
  if c_segments is not None and cmin is not None and cmax is not None:
    flat_c = np.concatenate(c_segments)
    if np.any(flat_c < cmin):
      logger.warning(f"The provided c colors array contains values that are less than cmin ({cmin}). Out of range values will be clipped into range.")
    if np.any(flat_c > cmax):
      logger.warning(
        f"The provided c colors array contains values that are greater than cmax ({cmax}). Out of range values will be clipped into range."
      )
    c_segments = [np.clip(ci, a_min=cmin, a_max=cmax) for ci in c_segments]

  # make segments and colors for each segment
  seg = []
  c_seg = []
  size_seg = []
  for idx, sub_xyz in enumerate(segment_list):
    seg.append(np.concatenate([sub_xyz[:, None], np.roll(sub_xyz, 1, 0)[:, None]], axis=1))
    if c_segments is not None:
      sub_c = c_segments[idx]
      c_seg.append((sub_c + np.roll(sub_c, 1, 0)) / 2)
    if size_segments is not None:
      sub_s = size_segments[idx]
      size_seg.append((sub_s + np.roll(sub_s, 1, 0)) / 2)

  seg = np.concatenate(seg, 0)
  if len(seg) == 0:
    raise ValueError("Provided segments produced no drawable segments.")
  c_seg = np.arange(len(seg))[::-1] if c_segments is None else np.concatenate(c_seg, 0)
  size_seg = None if size_segments is None else np.concatenate(size_seg, 0)

  # set colors
  c_seg = rescale(c_seg, cmin, cmax)
  colors = make_colors(c_seg)

  # remove segments that aren't connected
  seg_len = np.sqrt(np.square(seg[:, 0] - seg[:, 1]).sum(-1))
  if chainbreak is not None:
    idx = seg_len < chainbreak
    seg = seg[idx]
    seg_len = seg_len[idx]
    colors = colors[idx]
    if size_seg is not None:
      size_seg = size_seg[idx]

  if len(seg) == 0:
    raise ValueError("No drawable segments after applying chainbreak filtering.")

  seg_mid = seg.mean(1)
  seg_xy = seg[..., :2]
  seg_z = seg[..., 2].mean(-1)
  order = seg_z.argsort()

  # add shade/tint based on z-dimension
  z = rescale(seg_z)[:, None]

  # add shadow (make lines darker if they are behind other lines)
  seg_len_cutoff = (seg_len[:, None] + seg_len[None, :]) / 2
  seg_mid_z = seg_mid[:, 2]
  seg_mid_dist = np.sqrt(np.square(seg_mid[:, None] - seg_mid[None, :]).sum(-1))
  shadow_mask = sigmoid(seg_len_cutoff * 2.0 - seg_mid_dist) * (seg_mid_z[:, None] < seg_mid_z[None, :])
  np.fill_diagonal(shadow_mask, 0.0)
  shadow_mask = shadow ** shadow_mask.sum(-1, keepdims=True)

  seg_mid_xz = seg_mid[:, :2]
  seg_mid_xydist = np.sqrt(np.square(seg_mid_xz[:, None] - seg_mid_xz[None, :]).sum(-1))
  tint_mask = sigmoid(seg_len_cutoff / 2 - seg_mid_xydist) * (seg_mid_z[:, None] < seg_mid_z[None, :])
  np.fill_diagonal(tint_mask, 0.0)
  tint_mask = 1 - tint_mask.max(-1, keepdims=True)

  colors[:, :3] = colors[:, :3] + (1 - colors[:, :3]) * (0.50 * z + 0.50 * tint_mask) / 3
  colors[:, :3] = colors[:, :3] * (0.20 + 0.25 * z + 0.55 * shadow_mask)
  colors = np.clip(colors, 0.0, 1.0)

  upscale = max(1, int(upsample))
  target_size = (int(image_size[0] * upscale), int(image_size[1] * upscale))
  pad_px = int(padding * upscale)

  xy = all_xyz[:, :2]
  xy_min = xy.min(0)
  xy_max = xy.max(0)
  xy_range = xy_max - xy_min
  xy_range[xy_range == 0] = 1.0

  usable = np.array(target_size) - 2 * pad_px - 2
  usable[usable <= 0] = 1
  scale = np.min(usable / xy_range)
  if not np.isfinite(scale) or scale <= 0:
    scale = 1.0

  offset = pad_px + (usable - xy_range * scale) / 2
  seg_xy_px = (seg_xy - xy_min) * scale + offset
  seg_xy_px[..., 1] = target_size[1] - seg_xy_px[..., 1]

  linewidth_base_px = line_w * scale
  linewidth_base_px = np.clip(linewidth_base_px, 1, max(target_size) * 0.05)
  if size_seg is not None:
    linewidth_px = linewidth_base_px + size_seg * scale
    linewidth_px = np.clip(np.round(linewidth_px), 1, max(target_size) * 0.1)
  else:
    linewidth_px = np.array([np.clip(np.round(linewidth_base_px), 1, max(target_size) * 0.05)])

  img = Image.new("RGBA", target_size, (*background_color, 255))
  draw = ImageDraw.Draw(img, "RGBA")
  for idx in order:
    color = tuple((colors[idx, :3] * 255).astype(np.uint8)) + (255,)
    p1 = tuple(seg_xy_px[idx, 0])
    p2 = tuple(seg_xy_px[idx, 1])
    width_px = int(linewidth_px[idx if len(linewidth_px) > 1 else 0])
    draw.line([p1, p2], fill=color, width=width_px)
    # add small caps at segment ends to reduce visible gaps when downsampling
    r = max(1.0, width_px / 2.0)
    for px, py in (p1, p2):
      x0 = max(0.0, px - r)
      y0 = max(0.0, py - r)
      x1 = min(target_size[0] - 2, px + r)
      y1 = min(target_size[1] - 2, py + r)
      draw.ellipse((x0, y0, x1, y1), fill=color)

  if upscale > 1:
    img = img.resize(image_size, resample=Image.LANCZOS)

  # clear 1px border to background to avoid edge artifacts after resampling
  if image_size[0] > 2 and image_size[1] > 2:
    border_draw = ImageDraw.Draw(img, "RGBA")
    bg = (*background_color, 255)
    w, h = image_size
    border_draw.rectangle((0, 0, w - 1, 0), fill=bg)
    border_draw.rectangle((0, h - 1, w - 1, h - 1), fill=bg)
    border_draw.rectangle((0, 0, 0, h - 1), fill=bg)
    border_draw.rectangle((w - 1, 0, w - 1, h - 1), fill=bg)
  return img
