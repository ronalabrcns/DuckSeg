import numpy as np
import cv2

def convert_to_u8(frame):
    if frame.dtype == np.uint8:
        return frame
    elif frame.dtype == np.uint16:
        return (frame/(2.**16-1)*255).astype(np.uint8)
    else:
        raise AssertionError(f"Cannot convert to u8. Unsupported type {frame.dtype}")

def threshold_image(threshold_value):
    def threshold_image_inner(frame):
        ret, thresh_img = cv2.threshold(frame, threshold_value, 255, cv2.THRESH_BINARY)
        return thresh_img*128
    return threshold_image_inner

def adjust_contrast(alpha: float = 1.5, beta: float = 0.0):
    """
    Returns a function that applies cv2.convertScaleAbs
    with the specified alpha (contrast) and beta (brightness) values.
    
    Args:
        alpha (float): Contrast control (1.0 = no change).
        beta (float): Brightness adjustment (0 = no change).
    
    Returns:
        Callable[[np.ndarray], np.ndarray]: A function that takes
        a grayscale image (numpy array) and returns the adjusted image.
    """
    def adjust(image: np.ndarray) -> np.ndarray:
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return adjust


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
    def set(image: np.ndarray) -> np.ndarray:
        alpha = value/np.mean(image)
        return cv2.convertScaleAbs(image, alpha=alpha)
    return set


