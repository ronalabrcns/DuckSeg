"""
Per-frame preprocessing transforms for the cell-masking pipeline.

Every function in this module follows the same shape: it takes the
transform's parameters and returns a ``Callable[[numpy.ndarray],
numpy.ndarray]`` that applies the transform to a single frame. This makes
them composable in the ``transforms`` list passed to
:func:`DuckSeg.image_transformer.generate_masks_using_filters_and_transforms`
and :func:`DuckSeg.experiment_evaluator.evalute_batch`, e.g.::

    transforms = [convert_to_u8, set_brightness(20), threshold_image(20)]

:func:`convert_to_u8` is the one exception — it is applied directly to a
frame rather than being a transform factory, since it has no tunable
parameters.
"""

import numpy as np
import cv2

def convert_to_u8(frame):
    """
    Convert a frame to 8-bit unsigned integer format.

    The masking model (CellSAM) requires ``uint8`` input, so this is
    typically the first transform applied in a pipeline.

    Parameters
    ----------
    frame : numpy.ndarray
        Input frame, either ``uint8`` (returned unchanged) or ``uint16``
        (rescaled from the full 16-bit range to ``[0, 255]``).

    Returns
    -------
    numpy.ndarray
        The frame as ``uint8``.

    Raises
    ------
    AssertionError
        If ``frame.dtype`` is neither ``uint8`` nor ``uint16``.
    """
    if frame.dtype == np.uint8:
        return frame
    elif frame.dtype == np.uint16:
        return (frame/(2.**16-1)*255).astype(np.uint8)
    else:
        raise AssertionError(f"Cannot convert to u8. Unsupported type {frame.dtype}")

def threshold_image(threshold_value):
    """
    Build a transform that binary-thresholds a frame.

    Pixels above ``threshold_value`` are mapped to ``255``; the rest to
    ``0``. The result is then scaled by ``128`` (matching the value CellSAM
    expects for a strong binary mask signal, at the cost of no longer being
    a valid 8-bit image on overflow — intended purely as masking-model
    input, not for display).

    Parameters
    ----------
    threshold_value : int
        Pixel intensity threshold, 0-255.

    Returns
    -------
    Callable[[numpy.ndarray], numpy.ndarray]
        A function that thresholds a ``uint8`` grayscale frame.
    """
    def threshold_image_inner(frame):
        ret, thresh_img = cv2.threshold(frame, threshold_value, 255, cv2.THRESH_BINARY)
        return thresh_img*128
    return threshold_image_inner

def adjust_contrast(alpha: float = 1.0, beta: float = 0.0):
    """
    Returns a function that adjusts an image with:
      - cv2.convertScaleAbs for alpha >= 1 and beta >= 0
      - manual subtraction + saturation when beta < 0

    Args:
        alpha (float): Contrast multiplier.
        beta (float): Brightness shift (can be positive or negative).

    Returns:
        Callable[[np.ndarray], np.ndarray]: A function to adjust a uint8 grayscale image.
    """
    def adjust(image: np.ndarray) -> np.ndarray:
        if beta >= 0:
            # Standard case
            return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        else:
            # Negative beta (subtract), saturate at 0
            result = image.astype(np.int16) * alpha + beta
            result = np.clip(result, 0, 255)
            return result.astype(np.uint8)

    return adjust

def set_brightness(value: float):
    """
    Build a transform that rescales a frame to a target mean brightness.

    Parameters
    ----------
    value : float
        Target mean pixel intensity for the output frame.

    Returns
    -------
    Callable[[numpy.ndarray], numpy.ndarray]
        A function that rescales a frame so its mean intensity is
        approximately ``value``.
    """
    def set(image: np.ndarray) -> np.ndarray:
        alpha = value/np.mean(image)
        return cv2.convertScaleAbs(image, alpha=alpha)
    return set


