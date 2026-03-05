"""OpenCV cone-vision pipeline for MuSHR camera observations.

   RED cones with GREEN (left) and BLUE (right) stripe differentiation.

   Key design decisions
   --------------------
   * Connected components are run on RED pixels only.  Red cone bodies are
     separated by dark ground, so individual cones never merge into one blob.
     (Running CC on the union of all colours causes distant cones to merge
     into a single blob that is misclassified by whichever stripe colour has
     more pixels in the cluster — the main cause of flickering.)
   * After classifying each red blob by its nearest stripe colour, the output
     mask is built from red body pixels + the stripe pixels found inside the
     expansion zone.  A small morphological CLOSE then fills the gap between
     the red body and the stripe band so the full cone (body + stripe + tip)
     appears in the output.  The red tip above the stripe is a separate red
     blob in the CC pass; it is independently classified and merged via |=.
   * Detection runs at max(output_size, min_detect_size) so small / distant
     cones are not destroyed by downscaling before the HSV pipeline runs.
     Results are resized to the requested output resolution afterwards.
   * Green (H 35-85) and blue (H 95-145) HSV ranges have a deliberate gap
     at the cyan zone (H 85-95) to prevent ambiguous matches.
     Saturation/value floors are 50/50 (green) and 50/40 (blue) — permissive
     enough for dim distant-cone renders while still rejecting plain white/grey.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

try:
    import cv2
except Exception:  # pragma: no cover - runtime dependency in Isaac env
    cv2 = None


# ============================================================
# Configuration
# ============================================================

@dataclass(slots=True)
class ConeVisionCfg:
    """Configuration for colored cone segmentation and policy image generation."""

    # Policy (output) width.  Height is NOT configured here — ConeVisionProcessor
    # computes it automatically as:
    #   output_height = max(4, round(output_width × camera_height_width_ratio × (1 − crop_top − crop_bottom)))
    # This makes the mask shrink vertically proportional to the crop, reducing the
    # policy observation size rather than stretching a cropped view back to a fixed size.
    output_width: int = 256

    # Native camera height-to-width ratio (h/w).
    # 360/640 = 0.5625 for the Pi Camera Module 3 Wide (16:9).
    # The same value applies for any 16:9 resolution (e.g. 576/1024 in visualize).
    camera_height_width_ratio: float = 360.0 / 640.0

    # Detection always runs at max(output_size, min_detect_*) so small cones
    # are not destroyed by downscaling before the HSV pipeline runs.
    # 384×216 = 60% of the 640×360 training camera resolution after sky crop.
    # The 640×360 camera feeds this as a clean downscale (no upscaling artefacts).
    min_detect_width: int = 384
    min_detect_height: int = 216

    crop_top_fraction: float = 0.45    # fraction of frame rows to discard from top (sky)
    crop_bottom_fraction: float = 0.275  # fraction of frame rows to discard from bottom (near ground)
    near_field_fraction: float = 0.35  # bottom fraction of the CROPPED image treated as "near field"

    # MORPH_OPEN kernel applied to each colour mask individually to remove noise.
    # 2×2 (not 3×3) so that 2-pixel wide stripes on distant cones survive.
    # A 3×3 open would erase any cluster < 9 pixels — killing small far stripes.
    morph_open_size: int = 2

    # MORPH_CLOSE kernel applied to the per-cone pixel set (red body | nearby
    # stripe) to fill the small gap between the red body and the stripe band.
    morph_close_size: int = 5

    # Minimum stripe pixels required inside the expansion zone of a red blob
    # to classify it as a cone.  Lowered to 2 so that distant cones with only
    # 2 surviving stripe pixels are not silently dropped.
    stripe_min_pixels: int = 2

    # --- RED cone body (dual HSV range because hue wraps around 0/180) ---
    red1_hsv_min: tuple[int, int, int] = (0,   100,  60)
    red1_hsv_max: tuple[int, int, int] = (12,  255, 255)
    red2_hsv_min: tuple[int, int, int] = (165, 100,  60)
    red2_hsv_max: tuple[int, int, int] = (180, 255, 255)

    # --- GREEN stripe → LEFT cones ---
    # H 35-85: slightly wider than before to catch subtly yellow-green or
    #   cyan-shifted renders on distant cones.  Still well away from H 90 (cyan).
    # S/V floor 50: distant/small cones render dimmer; 80 was too strict and
    #   caused green to flicker far more than blue (blue had V=60 already).
    green_hsv_min: tuple[int, int, int] = (35,  50,  50)
    green_hsv_max: tuple[int, int, int] = (85, 255, 255)

    # --- BLUE stripe → RIGHT cones ---
    # H 95-145: slightly wider to catch dark or slightly cyan-shifted blue.
    # S/V floor 50/40: more permissive for distant small cones.
    blue_hsv_min: tuple[int, int, int] = (95,  50,  40)
    blue_hsv_max: tuple[int, int, int] = (145, 255, 255)


# ============================================================
# Processor
# ============================================================

class ConeVisionProcessor:
    """Processes RGB camera frames into LEFT/RIGHT cone masks.

    Output is a 2-channel observation [left_channel, right_channel], each of
    shape (output_height, output_width), flattened and concatenated.
    """

    def __init__(self, cfg: ConeVisionCfg):
        self.cfg = cfg

        # Compute output dimensions from crop fractions and camera aspect ratio.
        # output_height shrinks proportionally with the retained frame fraction so
        # the policy mask preserves the cropped aspect ratio instead of stretching.
        remaining = max(0.0, 1.0 - cfg.crop_top_fraction - cfg.crop_bottom_fraction)
        self.output_width  = cfg.output_width
        self.output_height = max(4, round(cfg.output_width * cfg.camera_height_width_ratio * remaining))

        if cv2 is None:
            raise ModuleNotFoundError(
                "OpenCV is required for cone-vision observations. "
                "Install `opencv-python` in the Isaac Lab Python environment."
            )

        self._red1_min  = np.array(self.cfg.red1_hsv_min,  dtype=np.uint8)
        self._red1_max  = np.array(self.cfg.red1_hsv_max,  dtype=np.uint8)
        self._red2_min  = np.array(self.cfg.red2_hsv_min,  dtype=np.uint8)
        self._red2_max  = np.array(self.cfg.red2_hsv_max,  dtype=np.uint8)
        self._green_min = np.array(self.cfg.green_hsv_min, dtype=np.uint8)
        self._green_max = np.array(self.cfg.green_hsv_max, dtype=np.uint8)
        self._blue_min  = np.array(self.cfg.blue_hsv_min,  dtype=np.uint8)
        self._blue_max  = np.array(self.cfg.blue_hsv_max,  dtype=np.uint8)

        # Noise removal: one kernel per colour mask.
        s = self.cfg.morph_open_size
        self._open_k = np.ones((s, s), np.uint8)

        # Stripe expansion: 11×11 = 5 px radius.  Large enough to reach the
        # stripe even for cones close to the camera (~5 px above body top at
        # 384×216 for a 2 m cone), small enough to never reach a neighbour
        # (cones are typically ≥ 20 px apart at detection resolution).
        self._stripe_expand_k = np.ones((11, 11), np.uint8)

        # Within-cone close: fills the narrow gap between red body and stripe.
        c = self.cfg.morph_close_size
        self._cone_close_k = np.ones((c, c), np.uint8)

    # --------------------------------------------------------

    @property
    def obs_size(self) -> int:
        """Flattened observation size: 2 channels × height × width."""
        return 2 * self.output_height * self.output_width

    # --------------------------------------------------------

    def process_batch(
        self,
        rgb_batch: torch.Tensor,
        debug_env_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, np.ndarray] | None]:
        """Convert a batch of RGB frames [N, H, W, C] into flattened LEFT/RIGHT masks."""
        if rgb_batch.ndim != 4:
            raise ValueError(
                f"Expected [N, H, W, C] RGB batch, got shape={tuple(rgb_batch.shape)}"
            )

        frames_np = rgb_batch.detach().cpu().numpy()
        batch     = frames_np.shape[0]

        policy_obs = np.zeros((batch, self.obs_size), dtype=np.float32)
        near_ratio = np.zeros(batch, dtype=np.float32)
        coverage   = np.zeros(batch, dtype=np.float32)
        debug_images: dict[str, np.ndarray] | None = None

        for env_id in range(batch):
            obs_flat, near, cov, overlay_bgr, mask_bgr = self._process_frame(
                frames_np[env_id]
            )
            policy_obs[env_id] = obs_flat
            near_ratio[env_id] = near
            coverage[env_id]   = cov

            if debug_env_id is not None and env_id == debug_env_id:
                debug_images = {"overlay_bgr": overlay_bgr, "mask_bgr": mask_bgr}

        return (
            torch.from_numpy(policy_obs),
            torch.from_numpy(near_ratio),
            torch.from_numpy(coverage),
            debug_images,
        )

    # --------------------------------------------------------

    def show_debug(
        self, debug_images: dict[str, np.ndarray], prefix: str = "MuSHR Camera"
    ) -> None:
        import os
        out_dir = "/tmp/mushr_debug"
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "rgb_mask.png"),  debug_images["overlay_bgr"])
        cv2.imwrite(os.path.join(out_dir, "cone_mask.png"), debug_images["mask_bgr"])
        if os.environ.get("DISPLAY"):
            try:
                cv2.imshow(f"{prefix} - RGB+Mask", debug_images["overlay_bgr"])
                cv2.imshow(f"{prefix} - ConeMask", debug_images["mask_bgr"])
                cv2.waitKey(1)
            except cv2.error:
                pass

    def close_debug(self, prefix: str = "MuSHR Camera") -> None:
        import os
        if os.environ.get("DISPLAY"):
            try:
                cv2.destroyWindow(f"{prefix} - RGB+Mask")
                cv2.destroyWindow(f"{prefix} - ConeMask")
            except cv2.error:
                pass

    # --------------------------------------------------------

    def _process_frame(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
        """Process a single RGB frame.

        Pipeline
        --------
        1.  Crop sky rows.
        2.  Resize to detection resolution.
        3.  Gentle 3×3 Gaussian blur.
        4.  HSV: build separate red, green, blue masks.
        5.  MORPH_OPEN each mask to remove tiny noise blobs.
        6.  Connected components on red_clean only.
            → Individual cone bodies stay separate (dark ground between them).
        7.  For each red blob:
              a. Dilate by 5 px to create an expansion zone.
              b. Count green / blue pixels inside the zone.
              c. If neither colour meets stripe_min_pixels → skip (noise).
              d. Build full-cone pixels = red blob | (zone ∩ stripe_all).
              e. MORPH_CLOSE to fill body-to-stripe gap.
              f. Add to left_det (green) or right_det (blue).
            The red tip above the stripe is its own separate red blob; it is
            classified by the same stripe below it and merged via |=.
        8.  Resize left/right masks to output resolution; re-binarize.
        9.  Compute near-field and full-frame coverage metrics.
        10. Build 2-channel float observation and BGR debug images.
        """
        rgb = self._as_uint8_rgb(frame)

        # 1. Crop sky (top) and near-ground clutter (bottom)
        h = rgb.shape[0]
        crop_start = int(self.cfg.crop_top_fraction * h)
        crop_end   = h - int(self.cfg.crop_bottom_fraction * h)
        cropped = rgb[crop_start:crop_end, :, :]

        # 2. Detection resolution
        det_w = max(self.output_width,  self.cfg.min_detect_width)
        det_h = max(self.output_height, self.cfg.min_detect_height)
        det_frame = cv2.resize(cropped, (det_w, det_h), interpolation=cv2.INTER_AREA)

        # 3. Gentle blur
        det_frame = cv2.GaussianBlur(det_frame, (3, 3), 0)
        hsv = cv2.cvtColor(det_frame, cv2.COLOR_RGB2HSV)

        # 4. Per-colour masks
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, self._red1_min, self._red1_max),
            cv2.inRange(hsv, self._red2_min, self._red2_max),
        )
        green_mask = cv2.inRange(hsv, self._green_min, self._green_max)
        blue_mask  = cv2.inRange(hsv, self._blue_min,  self._blue_max)

        # 5. Noise removal on each channel
        red_clean   = cv2.morphologyEx(red_mask,   cv2.MORPH_OPEN, self._open_k)
        green_clean = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, self._open_k)
        blue_clean  = cv2.morphologyEx(blue_mask,  cv2.MORPH_OPEN, self._open_k)

        # 6. Connected components on RED only
        num_labels, labels = cv2.connectedComponents(red_clean)

        left_det  = np.zeros((det_h, det_w), dtype=np.uint8)
        right_det = np.zeros((det_h, det_w), dtype=np.uint8)

        # 7. Classify and build full-cone pixels per red blob
        for label_id in range(1, num_labels):
            component = (labels == label_id).astype(np.uint8) * 255

            # --- Classification: small 5 px expansion, just enough to touch the stripe ---
            expanded = cv2.dilate(component, self._stripe_expand_k, iterations=1)
            n_green  = cv2.countNonZero(cv2.bitwise_and(expanded, green_clean))
            n_blue   = cv2.countNonZero(cv2.bitwise_and(expanded, blue_clean))

            if n_green < self.cfg.stripe_min_pixels and n_blue < self.cfg.stripe_min_pixels:
                continue  # no recognisable stripe — skip (noise or unlit far cone)

            # --- Full-cone pixel set: bounding-box region extended by the blob's own height ---
            #
            # The 5 px classification expansion only touches the very edge of the stripe,
            # leaving the stripe centre (and often the whole stripe) black.
            # Instead, extend the bounding box of the red blob by its own height in every
            # direction, then take (red | matching-stripe) pixels inside that box.
            #
            # Why this works for body vs tip blobs:
            #   Body (large, at bottom): extending upward by blob_h comfortably covers
            #     the stripe (≈33 % of blob_h) and the tip (≈33 % of blob_h) above it.
            #   Tip (small, at top): extending downward by blob_h covers the full stripe
            #     below it (stripe ≈ same height as the tip).
            # Both blobs independently grab the full stripe and merge via |=.
            bx, by, bw, bh = cv2.boundingRect(component)
            x_lo = max(0,     bx - 2)
            x_hi = min(det_w, bx + bw + 2)
            y_lo = max(0,     by - bh)        # extend upward   by one blob height
            y_hi = min(det_h, by + bh + bh)   # extend downward by one blob height

            region = np.zeros((det_h, det_w), dtype=np.uint8)
            region[y_lo:y_hi, x_lo:x_hi] = 255

            # Use the specific stripe colour (not stripe_all) so we never bleed
            # a wrong-colour neighbour stripe into this cone's output mask.
            stripe_color = green_clean if n_green >= n_blue else blue_clean
            cone_pixels  = cv2.bitwise_and(red_clean | stripe_color, region)
            cone_pixels  = cv2.morphologyEx(cone_pixels, cv2.MORPH_CLOSE, self._cone_close_k)

            if n_green >= n_blue:
                left_det  |= cone_pixels
            else:
                right_det |= cone_pixels

        # 8. Resize to output resolution and re-binarize
        out_w, out_h = self.output_width, self.output_height
        left_out  = cv2.resize(left_det,  (out_w, out_h), interpolation=cv2.INTER_AREA)
        right_out = cv2.resize(right_det, (out_w, out_h), interpolation=cv2.INTER_AREA)
        _, left_out  = cv2.threshold(left_out,  64, 255, cv2.THRESH_BINARY)
        _, right_out = cv2.threshold(right_out, 64, 255, cv2.THRESH_BINARY)

        # 9. Coverage metrics on the combined output mask
        combined   = cv2.bitwise_or(left_out, right_out)
        near_row   = int((1.0 - self.cfg.near_field_fraction) * out_h)
        near_ratio = float(combined[near_row:, :].mean() / 255.0)
        coverage   = float(combined.mean() / 255.0)

        # 10. Policy observation: [left_channel, right_channel] flattened
        obs_flat = np.stack([
            left_out.astype(np.float32)  / 255.0,
            right_out.astype(np.float32) / 255.0,
        ], axis=0).reshape(-1)

        # Debug visualisations at output resolution
        out_frame   = cv2.resize(det_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        out_bgr     = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
        color_layer = out_bgr.copy()
        color_layer[left_out  > 0] = (0,   255,   0)   # green  → left cones
        color_layer[right_out > 0] = (255,   0,   0)   # blue   → right cones
        overlay = cv2.addWeighted(out_bgr, 0.5, color_layer, 0.5, 0.0)
        cv2.line(overlay, (0, near_row), (out_w - 1, near_row), (0, 0, 255), 1)

        mask_bgr = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        mask_bgr[left_out  > 0] = (0,   255,   0)
        mask_bgr[right_out > 0] = (255,   0,   0)

        return obs_flat, near_ratio, coverage, overlay, mask_bgr

    # --------------------------------------------------------

    @staticmethod
    def _as_uint8_rgb(frame: np.ndarray) -> np.ndarray:
        if frame.shape[-1] > 3:
            frame = frame[..., :3]
        if frame.dtype == np.uint8:
            return frame
        frame   = frame.astype(np.float32)
        max_val = float(frame.max()) if frame.size else 0.0
        if max_val <= 1.0:
            frame = frame * 255.0
        return np.clip(frame, 0.0, 255.0).astype(np.uint8)
