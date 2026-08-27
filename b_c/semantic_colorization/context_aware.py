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
from scipy import ndimage

MIN_REGION_PIXELS = 250

# -----------------------------------------------------------------
# Global refinement parameters
# -----------------------------------------------------------------

EDGE_LOW_THRESHOLD = 75
EDGE_HIGH_THRESHOLD = 150

EDGE_PROTECTION = 0.35

SEMANTIC_SMOOTHING_KERNEL = 5

NEIGHBOR_BLEND = 0.25

ILLUMINATION_STRENGTH = 0.40

MAX_REFLECTION_STRENGTH = 0.75

MIN_REFLECTION_STRENGTH = 0.20

GAUSSIAN_SIGMA = 1.2


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
# Edge detection
# -----------------------------------------------------------------

def compute_edge_mask(L):
    """
    Detect strong intensity edges.

    Returns
    -------
    bool ndarray
    """

    gray = np.clip(L * 2.55, 0, 255).astype(np.uint8)

    edges = cv.Canny(
        gray,
        EDGE_LOW_THRESHOLD,
        EDGE_HIGH_THRESHOLD,
    )

    return edges > 0


# -----------------------------------------------------------------
# Region statistics
# -----------------------------------------------------------------

def compute_region_statistics(ab, mask):
    """
    Computes robust statistics for one semantic region.
    """

    a = ab[:, :, 0][mask]
    b = ab[:, :, 1][mask]

    return {
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "std_a": float(np.std(a)),
        "std_b": float(np.std(b)),
    }


# -----------------------------------------------------------------
# Connected semantic regions
# -----------------------------------------------------------------

def connected_region_masks(class_map):

    class_ids = np.unique(class_map)

    for cid in class_ids:

        binary = (class_map == cid).astype(np.uint8)

        count, labels = cv.connectedComponents(binary)

        for i in range(1, count):

            mask = labels == i

            if mask.sum() < MIN_REGION_PIXELS:
                continue

            yield cid, mask


# -----------------------------------------------------------------
# Semantic edge map
# -----------------------------------------------------------------

def semantic_boundary_map(class_map):

    kernel = np.ones((3, 3), np.uint8)

    dilated = cv.dilate(class_map.astype(np.uint8), kernel)

    eroded = cv.erode(class_map.astype(np.uint8), kernel)

    return dilated != eroded


# -----------------------------------------------------------------
# Semantic smoothing
# -----------------------------------------------------------------

def semantic_smoothing(
    ab,
    class_map,
    kernel_size=SEMANTIC_SMOOTHING_KERNEL,
):
    """
    Smooth colors INSIDE each semantic region only.

    Prevents noisy color predictions while avoiding
    color bleeding across semantic boundaries.
    """

    result = ab.copy()

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for class_id in np.unique(class_map):

        mask = (class_map == class_id)

        if mask.sum() < MIN_REGION_PIXELS:
            continue

        mask_uint8 = mask.astype(np.uint8)

        eroded = cv.erode(mask_uint8, kernel)

        if eroded.sum() < MIN_REGION_PIXELS:
            continue

        for c in range(2):

            channel = result[:, :, c]

            blurred = cv.GaussianBlur(
                channel,
                (5, 5),
                GAUSSIAN_SIGMA,
            )

            channel[eroded.astype(bool)] = blurred[
                eroded.astype(bool)
            ]

            result[:, :, c] = channel

    return result


# -----------------------------------------------------------------
# Edge preservation
# -----------------------------------------------------------------

def preserve_semantic_edges(
    ab,
    class_map,
    L,
):
    """
    Protect strong image edges from excessive smoothing.
    """

    edge_mask = compute_edge_mask(L)

    semantic_edges = semantic_boundary_map(class_map)

    preserve = edge_mask | semantic_edges

    result = ab.copy()

    blur_a = cv.bilateralFilter(
        result[:, :, 0].astype(np.float32),
        7,
        25,
        25,
    )

    blur_b = cv.bilateralFilter(
        result[:, :, 1].astype(np.float32),
        7,
        25,
        25,
    )

    result[:, :, 0][~preserve] = blur_a[~preserve]
    result[:, :, 1][~preserve] = blur_b[~preserve]

    return result


# -----------------------------------------------------------------
# Illumination consistency
# -----------------------------------------------------------------

def apply_illumination_consistency(
        ab,
        L,
    ):
        """
        Preserve global lighting.

        Bright regions remain saturated.

        Dark regions become slightly desaturated.

        This mimics real-world illumination without changing
        the semantic colors predicted by DDColor.
        """

        result = ab.copy()

        brightness = np.clip(L / 100.0, 0.15, 1.0)

        saturation_scale = (
            0.75
            + ILLUMINATION_STRENGTH * brightness
        )

        result[:, :, 0] *= saturation_scale
        result[:, :, 1] *= saturation_scale

        return result


# -----------------------------------------------------------------
# Shadow consistency
# -----------------------------------------------------------------

def apply_shadow_consistency(
    ab,
    class_map,
    L,
    shadow_ratio=0.65,
    min_darkness_scale=0.40,
):
    """
    Improved context-aware shadow refinement.

    Instead of treating every semantic class as one giant region,
    each connected semantic object is processed independently.

    This prevents one building from affecting another building,
    one tree affecting another tree, etc.
    """

    edge_mask = compute_edge_mask(L)

    for class_id, mask in connected_region_masks(class_map):

        region_pixels = mask.sum()

        if region_pixels < MIN_REGION_PIXELS:
            continue

        region_L = L[mask]

        median_L = float(np.median(region_L))
        mean_L = float(np.mean(region_L))
        std_L = float(np.std(region_L))

        if median_L < 1e-3:
            continue

        # Adaptive threshold
        shadow_threshold = median_L - 0.5 * std_L

        shadow_threshold = max(
            shadow_threshold,
            median_L * shadow_ratio,
        )

        shadow_mask = mask & (L < shadow_threshold)

        if shadow_mask.sum() == 0:
            continue

        stats = compute_region_statistics(ab, mask)

        median_a = stats["median_a"]
        median_b = stats["median_b"]

        darkness = np.clip(
            L[shadow_mask] / median_L,
            min_darkness_scale,
            1.0,
        )

        # Preserve chroma slightly better
        chroma_scale = 0.7 + 0.3 * darkness

        target_a = median_a * chroma_scale
        target_b = median_b * chroma_scale

        shadow_indices = np.where(shadow_mask)

        current_a = ab[:, :, 0][shadow_indices]
        current_b = ab[:, :, 1][shadow_indices]

        blend = darkness

        current_edges = edge_mask[shadow_indices]

        # Protect semantic boundaries
        blend[current_edges] *= EDGE_PROTECTION

        ab[:, :, 0][shadow_indices] = (
            blend * current_a
            + (1.0 - blend) * target_a
        )

        ab[:, :, 1][shadow_indices] = (
            blend * current_b
            + (1.0 - blend) * target_b
        )

    return ab


# -----------------------------------------------------------------
# Reflection consistency
# -----------------------------------------------------------------

def apply_reflection_consistency(
    ab,
    class_map,
    id2label,
    L,
):
    """
    Enhanced reflection consistency.

    Improvements
    ------------
    1. Adaptive reflection strength.
    2. Uses local brightness.
    3. Preserves edges.
    4. Supports water, glass and mirrors.
    5. Prevents unrealistic reflections.
    """

    reflective_ids = _reflective_class_ids(id2label)

    if not reflective_ids:
        return ab

    edge_mask = compute_edge_mask(L)

    combined_mask = np.isin(class_map, list(reflective_ids))

    if combined_mask.sum() < MIN_REGION_PIXELS:
        return ab

    num_labels, components = cv.connectedComponents(
        combined_mask.astype(np.uint8)
    )

    for component_id in range(1, num_labels):

        component = components == component_id

        if component.sum() < MIN_REGION_PIXELS:
            continue

        rows = np.where(component.any(axis=1))[0]
        cols = np.where(component.any(axis=0))[0]

        top = rows.min()
        bottom = rows.max()

        left = cols.min()
        right = cols.max()

        region_height = bottom - top + 1
        region_width = right - left + 1

        source_height = min(region_height, top)

        if source_height <= 0:
            continue

        source_top = top - source_height

        source_a = ab[
            source_top:top,
            left:right + 1,
            0,
        ]

        source_b = ab[
            source_top:top,
            left:right + 1,
            1,
        ]

        if source_a.size == 0:
            continue

        source_L = L[
            source_top:top,
            left:right + 1,
        ]

        flipped_a = np.flipud(source_a)
        flipped_b = np.flipud(source_b)
        flipped_L = np.flipud(source_L)

        resized_a = cv.resize(
            flipped_a,
            (region_width, region_height),
            interpolation=cv.INTER_LINEAR,
        )

        resized_b = cv.resize(
            flipped_b,
            (region_width, region_height),
            interpolation=cv.INTER_LINEAR,
        )

        resized_L = cv.resize(
            flipped_L,
            (region_width, region_height),
            interpolation=cv.INTER_LINEAR,
        )

        bbox_mask = component[
            top:bottom + 1,
            left:right + 1,
        ]

        bbox_edges = edge_mask[
            top:bottom + 1,
            left:right + 1,
        ]

        bbox_a = ab[
            top:bottom + 1,
            left:right + 1,
            0,
        ]

        bbox_b = ab[
            top:bottom + 1,
            left:right + 1,
            1,
        ]

        bbox_L = L[
            top:bottom + 1,
            left:right + 1,
        ]

        brightness_ratio = np.clip(
            bbox_L / (resized_L + 1e-6),
            0.35,
            1.0,
        )

        reflection_strength = (
            MIN_REFLECTION_STRENGTH
            + (MAX_REFLECTION_STRENGTH - MIN_REFLECTION_STRENGTH)
            * brightness_ratio
        )

        reflection_strength *= (
            1.0 - 0.20 * bbox_edges
        )

        reflection_strength = np.clip(
            reflection_strength,
            MIN_REFLECTION_STRENGTH,
            MAX_REFLECTION_STRENGTH,
        )

        resized_a *= 0.65
        resized_b *= 0.65

        blended_a = (
            bbox_a * (1 - reflection_strength)
            + resized_a * reflection_strength
        )

        blended_b = (
            bbox_b * (1 - reflection_strength)
            + resized_b * reflection_strength
        )

        bbox_a[bbox_mask] = blended_a[bbox_mask]
        bbox_b[bbox_mask] = blended_b[bbox_mask]

        ab[
            top:bottom + 1,
            left:right + 1,
            0,
        ] = bbox_a

        ab[
            top:bottom + 1,
            left:right + 1,
            1,
        ] = bbox_b

    return ab


# -----------------------------------------------------------------
# Combined entry point
# -----------------------------------------------------------------

def apply_context_aware_consistency(ab, class_map, L, id2label):
    """Runs shadow consistency then reflection consistency in sequence."""

    ab = apply_shadow_consistency(
        ab,
        class_map,
        L,
    )

    ab = apply_reflection_consistency(
        ab,
        class_map,
        id2label,
        L,
    )

    ab = semantic_smoothing(
        ab,
        class_map,
    )

    ab = preserve_semantic_edges(
        ab,
        class_map,
        L,
    )

    ab = apply_illumination_consistency(
        ab,
        L,
    )

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