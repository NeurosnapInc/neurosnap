"""
Provides functions and classes related to rendering, plotting, animating, and visualizing data.
"""

import colorsys
import pathlib
import re
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.special import expit as sigmoid
from tqdm import tqdm

from neurosnap.log import logger
from neurosnap.protein import Protein


def render_pseudo3D(
  segments: Iterable[Union[np.ndarray, pd.DataFrame]],
  *,
  c: Optional[Iterable[np.ndarray]] = None,
  sizes: Optional[Iterable[np.ndarray]] = None,
  cmap: str = "gist_rainbow",
  cmin: Optional[float] = None,
  cmax: Optional[float] = None,
  image_size: Tuple[int, int] = (800, 800),
  padding: int = 20,
  line_w: float = 2.0,
  shadow: float = 0.95,
  background_color: Tuple[int, int, int] = (255, 255, 255),
  upsample: int = 2,
  chainbreak: int = 5,
) -> Image.Image:
  """Plot the famous Pseudo 3D projection of a protein using Pillow.

  Adapted from an algorithm originally written By Dr. Sergey Ovchinnikov.

  Parameters:
    segments: Iterable of XYZ coordinates, where each element is a segment/molecule to draw separately
    c: Iterable of 1D arrays used to color the protein, aligned one-to-one with ``segments``; defaults to residue index
    sizes: Iterable of 1D arrays of radii/size values, aligned one-to-one with ``segments``; interpreted in the same units as coordinates
    cmap: Color map name or callable used for coloring the protein
    cmin: Minimum value for coloring, automatically calculated if None
    cmax: Maximum value for coloring, automatically calculated if None
    image_size: Final image size in pixels (width, height)
    padding: Padding in pixels around the drawing region
    line_w: Line width (interpreted in data space; converted to pixels)
    shadow: Shadow intensity between 0 and 1 inclusive, lower numbers mean darker more intense shadows
    background_color: Background RGB color
    upsample: Factor to draw at higher resolution and downsample for antialiasing
    chainbreak: Minimum distance in angstroms between chains / segments before being considered a chain break (int)

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


def render_protein_pseudo3D(
  protein: Protein,
  *,
  style: str = "residue_id",
  use_radii: bool = False,
  image_size: Tuple[int, int] = (576, 432),
  padding: int = 20,
  shadow: float = 0.95,
  upsample: int = 2,
  chainbreak: int = 5,
) -> Image.Image:
  """Render a protein using the pseudo-3D Pillow renderer.

  Parameters:
    protein: Protein to render
    style: Coloring mode (residue_id, chain_id, b-factor, pLDDT, residue_type)
    use_radii: If True, apply van der Waals radii as per-atom sizes
    image_size: Output image size (width, height)
    padding: Padding in pixels around the drawing region
    upsample: Supersampling factor for antialiasing
    chainbreak: Distance threshold for breaking segments
    shadow: Shadow intensity between 0 and 1

  Returns:
    Pillow Image containing the rendering

  """
  df = protein.df
  coords = df[["x", "y", "z"]]
  chains = df["chain"].to_numpy()
  segments = [coords.to_numpy()[chains == chain_id] for chain_id in pd.unique(chains)]

  # build colors per style
  style = style.lower()
  color_segments: Optional[List[np.ndarray]] = None
  custom_cmap = None
  cmin = None
  cmax = None
  if style in {"residue_id", "chain_id", "b-factor", "plddt", "residue_type"}:
    color_segments = []
    if style == "chain_id":
      chain_map = {cid: idx for idx, cid in enumerate(pd.unique(chains))}
    elif style == "residue_type":
      res_codes = {res: idx for idx, res in enumerate(pd.unique(df["res_name"]))}
    elif style == "plddt":
      palette = [
        np.array([0xFF, 0x7D, 0x45, 0xFF], dtype=float) / 255.0,  # <=50
        np.array([0xFF, 0xDB, 0x13, 0xFF], dtype=float) / 255.0,  # <=70
        np.array([0x65, 0xCB, 0xF3, 0xFF], dtype=float) / 255.0,  # <=90
        np.array([0x00, 0x53, 0xD6, 0xFF], dtype=float) / 255.0,  # >90
      ]

      def cmap_plddt(vals: np.ndarray) -> np.ndarray:
        vals = np.asarray(vals)
        out = np.zeros((len(vals), 4))
        out[vals <= 0.33] = palette[0]
        out[(vals > 0.33) & (vals <= 0.5)] = palette[1]
        out[(vals > 0.5) & (vals <= 0.75)] = palette[2]
        out[vals > 0.75] = palette[3]
        return out

      custom_cmap = cmap_plddt
      cmin = 0.0
      cmax = 1.0
    for chain_id in pd.unique(chains):
      mask = chains == chain_id
      if style == "residue_id":
        color_segments.append(df.loc[mask, "res_id"].to_numpy(dtype=float))
      elif style == "chain_id":
        color_segments.append(np.full(mask.sum(), chain_map[chain_id], dtype=float))
      elif style == "b-factor":
        color_segments.append(df.loc[mask, "bfactor"].to_numpy(dtype=float))
      elif style == "plddt":
        bvals = df.loc[mask, "bfactor"].to_numpy(dtype=float)
        bins = np.zeros_like(bvals, dtype=float)
        bins[bvals <= 50.0] = 0.0
        bins[(bvals > 50.0) & (bvals <= 70.0)] = 0.33
        bins[(bvals > 70.0) & (bvals <= 90.0)] = 0.5
        bins[bvals > 90.0] = 1.0
        color_segments.append(bins)
      elif style == "residue_type":
        color_segments.append(df.loc[mask, "res_name"].map(res_codes).to_numpy(dtype=float))
  else:
    raise ValueError(f"Unsupported style '{style}'.")

  size_segments: Optional[List[np.ndarray]] = None
  if use_radii:
    vdw_radii = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52, "P": 1.8, "S": 1.8}
    atoms = df["atom_name"].astype(str).to_numpy()
    radii = []
    for name in atoms:
      stripped = name.strip()
      element = re.sub(r"[^A-Za-z]", "", stripped).upper()
      if len(element) >= 2 and element[:2] in vdw_radii:
        elem = element[:2]
      elif element:
        elem = element[0]
      else:
        elem = ""
      radii.append(vdw_radii.get(elem, 1.5))
    radii = np.asarray(radii, dtype=float)
    size_segments = [radii[chains == chain_id] for chain_id in pd.unique(chains)]

  return render_pseudo3D(
    segments,
    c=color_segments,
    sizes=size_segments,
    cmap=custom_cmap if custom_cmap is not None else "gist_rainbow",
    cmin=cmin,
    cmax=cmax,
    image_size=image_size,
    padding=padding,
    upsample=upsample,
    chainbreak=chainbreak,
    shadow=shadow,
  )


def animate_frames(
  frames: Iterable[Union[Image.Image, np.ndarray]],
  output_fpath: Union[str, pathlib.Path],
  *,
  title: str = "",
  subtitles: Optional[Iterable[str]] = None,
  interval: int = 200,
  repeat: bool = True,
):
  """Animate a sequence of frames using Pillow only and write to disk.

  Parameters:
    frames: Iterable of frames to animate (Pillow Images or arrays convertible to images)
    output_fpath: Path where the animation will be written; format inferred from extension (gif, webp, mp4)
    title: Title text to display above the animation; omit if empty
    subtitles: Iterable of subtitle strings, one per frame (must match length of frames)
    interval: Delay between frames in milliseconds
    repeat: Whether the animation repeats when the sequence of frames is completed (loop=0 if True else 1 for gif/webp; ignored for mp4)
  """
  frame_list = list(frames)
  if len(frame_list) == 0:
    raise ValueError("No frames provided to animate.")

  if subtitles is None:
    subtitle_list = [""] * len(frame_list)
  else:
    subtitle_list = list(subtitles)
    if len(subtitle_list) != len(frame_list):
      raise ValueError(f"subtitles length ({len(subtitle_list)}) must match number of frames ({len(frame_list)}).")

  def to_image(obj: Union[Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(obj, Image.Image):
      return obj.convert("RGBA")
    arr = np.asarray(obj)
    if arr.ndim < 2:
      raise ValueError("Frames must be images or 2D/3D arrays.")
    if arr.dtype != np.uint8:
      arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
      arr = np.stack([arr] * 3, axis=-1)
    mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    return Image.fromarray(arr, mode=mode).convert("RGBA")

  pil_frames = [to_image(fr) for fr in frame_list]
  output_path = pathlib.Path(output_fpath)
  ext = output_path.suffix.lower()
  if ext not in {".gif", ".webp", ".mp4"}:
    raise ValueError(f"Unsupported output format '{ext}'. Supported: .gif, .webp, .mp4")

  font = ImageFont.load_default()

  # Measure text heights to allocate a top padding region
  def text_size(text: str) -> Tuple[int, int]:
    if not text:
      return (0, 0)
    # use dummy draw to measure
    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])

  title_w, title_h = text_size(title)
  subtitle_heights = [text_size(sub)[1] for sub in subtitle_list]
  subtitle_h = max(subtitle_heights) if subtitle_heights else 0
  text_padding = 4 if title or subtitle_h else 0
  top_pad = title_h + subtitle_h + text_padding

  animated_frames: List[Image.Image] = []
  for idx, (img, sub) in enumerate(tqdm(zip(pil_frames, subtitle_list), total=len(pil_frames), desc="Animating frames")):
    if top_pad > 0:
      canvas = Image.new("RGBA", (img.width, img.height + top_pad), (255, 255, 255, 0))
      canvas.paste(img, (0, top_pad))
    else:
      canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    y = 2
    if title:
      tw, th = text_size(title)
      tx = (canvas.width - tw) / 2
      draw.text((tx, y), title, fill=(0, 0, 0, 255), font=font)
      y += th + 2
    if sub:
      sw, sh = text_size(sub)
      sx = (canvas.width - sw) / 2
      draw.text((sx, y), sub, fill=(0, 0, 0, 255), font=font)
    animated_frames.append(canvas)

  if ext in {".gif", ".webp"}:
    save_kwargs = {
      "save_all": True,
      "append_images": animated_frames[1:],
      "duration": interval,
      "loop": 0 if repeat else 1,
      "optimize": False,
    }
    animated_frames[0].save(output_path, **save_kwargs)
  elif ext == ".mp4":
    try:
      import imageio
    except ImportError as e:
      raise ImportError("imageio is required to write mp4 animations. Install via `pip install imageio imageio-ffmpeg`.") from e
    fps = max(1e-3, 1000.0 / float(interval))  # avoid zero, allow sub-1 fps
    writer = imageio.get_writer(output_path, fps=fps, macro_block_size=None)
    try:
      for frm in animated_frames:
        writer.append_data(np.asarray(frm.convert("RGB")))
    finally:
      writer.close()
