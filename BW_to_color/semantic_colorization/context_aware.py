"""
Context-Aware Consistency Layer
================================

Post-processes the base Zhang colorizer's predicted a,b channels so that
color choices respect *relationships between regions* rather than treating
every pixel independently:

  • Shadow consistency
      A shadow falling across a surface should be a darker/desaturated
      version of THAT surface's color, not an independently guessed hue.
      We find dark sub-regions inside each semantic class's mask and pull
      their color toward the class's own median color, scaled down by how
      dark the shadow is.

  • Reflection consistency
      Reflective surfaces (water, glass, mirrors) should echo the color of
      whatever is directly above them, flipped and desaturated -- not an
      independently predicted color.

This module is intentionally model-free / classical-CV: it works directly
on the ab channels already produced by the base colorizer and the class
map already produced by the segmenter, so it adds contextual reasoning
without requiring a new heavyweight model.
"""

import cv2 as cv
import numpy as np


# -----------------------------------------------------------------
# Which ADE20K classes count as "reflective" (by substring match,
# since ADE20K label strings are things like "windowpane, window")
# -----------------------------------------------------------------

REFLECTIVE_KEYWORDS = [
    "water",
    "sea",
    "river",
    "lake",
    "pool",
    "windowpane",
    "window",
    "mirror",
    "glass",
]

MIN_REGION_PIXELS = 250


def _reflective_class_ids(id2label):
    """Return the set of ADE20K class ids that behave like reflective
    surfaces, based on keyword matching against their label text."""

    reflective_ids = set()

    for idx, name in id2label.items():
        lname = name.lower()
        if any(keyword in lname for keyword in REFLECTIVE_KEYWORDS):
            reflective_ids.add(int(idx))

    return reflective_ids


# -----------------------------------------------------------------
# Shadow consistency
# -----------------------------------------------------------------

def apply_shadow_consistency(
    ab,
    class_map,
    L,
    shadow_ratio=0.65,
    min_darkness_scale=0.35,
    min_region_pixels=MIN_REGION_PIXELS,
):
    """
    For each detected semantic region, find pixels that are notably
    darker than the rest of that region (i.e. likely in shadow) and
    re-color them as a darkened version of the region's own median
    color, instead of leaving them at whatever the base model guessed
    independently.

    Parameters
    ----------
    ab : (H, W, 2) float32
        Predicted a,b channels from the base colorizer (mutated in place
        and returned).
    class_map : (H, W) int
        Per-pixel ADE20K class id from the segmenter.
    L : (H, W) float32
        Lightness channel of the original image (0-100 scale).
    """

    class_ids = np.unique(class_map)

    for class_id in class_ids:

        mask = class_map == class_id

        pixel_count = int(mask.sum())

        if pixel_count < min_region_pixels:
            continue

        region_L = L[mask]

        median_L = float(np.median(region_L))

        if median_L <= 1e-3:
            continue

        shadow_threshold = median_L * shadow_ratio

        shadow_mask = mask & (L < shadow_threshold)

        if shadow_mask.sum() == 0:
            continue

        median_a = float(np.median(ab[:, :, 0][mask]))
        median_b = float(np.median(ab[:, :, 1][mask]))

        # How dark is each shadow pixel relative to the region's
        # typical brightness? Darker -> lower chroma scale (more
        # desaturated), mimicking how real shadows look.
        darkness_scale = np.clip(
            L[shadow_mask] / median_L,
            min_darkness_scale,
            1.0,
        )

        ab[:, :, 0][shadow_mask] = median_a * darkness_scale
        ab[:, :, 1][shadow_mask] = median_b * darkness_scale

    return ab


# -----------------------------------------------------------------
# Reflection consistency
# -----------------------------------------------------------------

def apply_reflection_consistency(
    ab,
    class_map,
    id2label,
    reflection_strength=0.55,
    reflection_desaturation=0.6,
    min_region_pixels=MIN_REGION_PIXELS,
):
    """
    For reflective classes (water, glass, mirrors), blend in a
    vertically-flipped, desaturated copy of the ab channels from the
    strip of image directly above each reflective region -- so a lake
    picks up the sky/treeline color above it, a window picks up
    whatever is in front of it, etc.

    Parameters
    ----------
    ab : (H, W, 2) float32
        Predicted a,b channels (mutated in place and returned).
    class_map : (H, W) int
        Per-pixel ADE20K class id.
    id2label : dict[int, str]
        Segmenter's class id -> class name mapping.
    """

    reflective_ids = _reflective_class_ids(id2label)

    if not reflective_ids:
        return ab

    height, width = class_map.shape

    combined_reflective_mask = np.isin(class_map, list(reflective_ids))

    if combined_reflective_mask.sum() < min_region_pixels:
        return ab

    # Treat each disconnected reflective blob separately, so a window
    # reflects what's in front of IT, not some other window's view.
    num_labels, components = cv.connectedComponents(
        combined_reflective_mask.astype(np.uint8)
    )

    for component_id in range(1, num_labels):

        component_mask = components == component_id

        pixel_count = int(component_mask.sum())

        if pixel_count < min_region_pixels:
            continue

        rows = np.where(component_mask.any(axis=1))[0]
        cols = np.where(component_mask.any(axis=0))[0]

        top_row = int(rows.min())
        bottom_row = int(rows.max())
        left_col = int(cols.min())
        right_col = int(cols.max())

        region_height = bottom_row - top_row + 1

        # Source strip: the area directly above this reflective region,
        # same width, similar height (capped so we don't reach off the
        # top of the image).
        source_height = min(region_height, top_row)

        if source_height <= 0:
            # Reflective region touches the top of the frame -- nothing
            # above it to reflect, leave the base prediction as-is.
            continue

        source_top = top_row - source_height
        source_bottom = top_row

        source_a = ab[source_top:source_bottom, left_col:right_col + 1, 0]
        source_b = ab[source_top:source_bottom, left_col:right_col + 1, 1]

        # Flip vertically -- reflections mirror what's above them.
        flipped_a = np.flipud(source_a)
        flipped_b = np.flipud(source_b)

        # Resize the flipped strip to match this component's bounding
        # box height exactly.
        target_shape = (right_col - left_col + 1, region_height)

        if flipped_a.size == 0:
            continue

        resized_a = cv.resize(
            flipped_a, target_shape, interpolation=cv.INTER_LINEAR
        )
        resized_b = cv.resize(
            flipped_b, target_shape, interpolation=cv.INTER_LINEAR
        )

        # Desaturate -- reflections are usually softer/less saturated
        # than the thing being reflected.
        resized_a *= reflection_desaturation
        resized_b *= reflection_desaturation

        # Blend into just this component's pixels within its bbox.
        bbox_component_mask = component_mask[
            top_row:bottom_row + 1, left_col:right_col + 1
        ]

        bbox_a = ab[top_row:bottom_row + 1, left_col:right_col + 1, 0]
        bbox_b = ab[top_row:bottom_row + 1, left_col:right_col + 1, 1]

        blended_a = (
            (1.0 - reflection_strength) * bbox_a
            + reflection_strength * resized_a
        )
        blended_b = (
            (1.0 - reflection_strength) * bbox_b
            + reflection_strength * resized_b
        )

        bbox_a[bbox_component_mask] = blended_a[bbox_component_mask]
        bbox_b[bbox_component_mask] = blended_b[bbox_component_mask]

        ab[top_row:bottom_row + 1, left_col:right_col + 1, 0] = bbox_a
        ab[top_row:bottom_row + 1, left_col:right_col + 1, 1] = bbox_b

    return ab


# -----------------------------------------------------------------
# Combined entry point
# -----------------------------------------------------------------

def apply_context_aware_consistency(ab, class_map, L, id2label):
    """Runs shadow consistency then reflection consistency in sequence."""

    ab = apply_shadow_consistency(ab, class_map, L)

    ab = apply_reflection_consistency(ab, class_map, id2label)

    return ab


if __name__ == "__main__":

    # Small smoke test with synthetic data -- no real model needed.
    print("Testing context_aware module with synthetic data...")

    H, W = 120, 160

    class_map = np.zeros((H, W), dtype=np.int64)
    class_map[:60, :] = 1          # "sky" (top half)
    class_map[60:, :] = 2          # "water" (bottom half, reflects sky)

    id2label = {0: "background", 1: "sky", 2: "water"}

    L = np.full((H, W), 70.0, dtype=np.float32)
    L[80:100, 40:120] = 20.0       # a dark patch inside the water region

    ab = np.zeros((H, W, 2), dtype=np.float32)
    ab[:60, :, 0] = -20.0          # sky: cool blue-ish a
    ab[:60, :, 1] = -40.0          # sky: cool blue-ish b
    ab[60:, :, 0] = 5.0            # water base guess: near neutral
    ab[60:, :, 1] = 5.0

    ab = apply_context_aware_consistency(ab.copy(), class_map, L, id2label)

    water_a_mean = ab[60:, :, 0].mean()
    water_b_mean = ab[60:, :, 1].mean()

    print(f"Water region mean (a,b) after reflection: "
          f"({water_a_mean:.2f}, {water_b_mean:.2f})")
    print("(Should have shifted toward the sky's cool blue tone)")

    shadow_patch_a = ab[80:100, 40:120, 0].mean()
    print(f"Shadow patch mean a: {shadow_patch_a:.2f} "
          f"(should be scaled down from the water region's own color)")

    print("Smoke test complete.")