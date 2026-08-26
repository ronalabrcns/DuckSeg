"""
Frame alignment and brightness normalization utilities.

This module provides two complementary ways of registering the frames of a
laser-irradiation time-lapse video to a common reference frame:

- Feature-based alignment (:func:`align_frame_to_reference`,
  :func:`align_and_adjust_frames`) using ORB keypoints and a homography.
- Brute-force alignment (:func:`align_frames_bruteforce` and friends) that
  searches a grid of small translations/rotations and keeps the one that
  maximizes a similarity metric (e.g. :func:`correlation_metric`) between
  consecutive frames. This is the alignment strategy used by
  :func:`align_cell_video` to stabilize individual cell crops before
  brightness scoring.
"""

import cv2
import numpy as np
import itertools

def calculate_brightness(frame):
    """
    Calculate the average brightness of a grayscale frame.
    
    Parameters:
        frame (ndarray): The grayscale frame image.
        
    Returns:
        float: The average brightness.
    """
    return np.mean(frame)

def adjust_brightness(frame, target_brightness):
    """
    Adjust the brightness of a grayscale frame to match a target brightness.
    
    Parameters:
        frame (ndarray): The input grayscale frame.
        target_brightness (float): The target average brightness level.
        
    Returns:
        ndarray: The brightness-adjusted frame.
    """
    current_brightness = calculate_brightness(frame)
    adjustment_factor = target_brightness / (current_brightness + 1e-6)
    print(adjustment_factor)
    
    # Scale pixel values by the adjustment factor and clip to valid range
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=adjustment_factor, beta=0)
    return adjusted_frame

def align_frame_to_reference(reference_frame, frame, transform):
    """
    Aligns a frame to a reference frame using keypoints and homography.

    Parameters:
        reference_frame (ndarray): The reference grayscale frame.
        frame (ndarray): The grayscale frame to be aligned to the reference frame.

    Returns:
        aligned_frame (ndarray): The aligned version of the frame.
    """
    # Use ORB to detect and describe features
    orb = cv2.ORB_create()
    keypoints_ref, descriptors_ref = orb.detectAndCompute(reference_frame, None)
    keypoints_frame, descriptors_frame = orb.detectAndCompute(frame, None)
    print(keypoints_ref, descriptors_ref)
    print(keypoints_frame, descriptors_frame)

    # Match features using the BFMatcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_ref, descriptors_frame)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    points_ref = np.float32([keypoints_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    print(points_ref)
    points_frame = np.float32([keypoints_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    print(points_frame)

    # Find homography matrix
    homography, _ = cv2.findHomography(points_frame, points_ref, cv2.RANSAC, 5.0)

    # Warp the frame to align with the reference frame
    height, width = reference_frame.shape
    aligned_frame = cv2.warpPerspective(frame, homography, (width, height))

    return aligned_frame

def align_and_adjust_frames(frames):
    """
    Aligns each frame in the list to the first frame and adjusts brightness to match the first frame.

    Parameters:
        frames (list of ndarray): List of grayscale frames to process.

    Returns:
        list of ndarray: List of aligned and brightness-adjusted frames.
    """
    if not frames:
        return []

    # Use the first frame as the reference
    reference_frame = frames[0]
    reference_brightness = calculate_brightness(reference_frame)

    # Process each frame
    aligned_frames = []
    for frame in frames:
        # Adjust the brightness of the current frame to match the reference brightness
        adjusted_frame = adjust_brightness(frame, reference_brightness)

        # Align the adjusted frame to the reference frame
        aligned_frame = align_frame_to_reference(reference_frame, adjusted_frame)

        # Append the aligned and adjusted frame to the list
        aligned_frames.append(aligned_frame)

    return aligned_frames

def pad_then_transform_2d(image, padding, shift0, shift1, rotation):
    """
    Zero-pad an image and then apply a 2D translation/rotation.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    padding : int or sequence of int
        Padding width(s) passed to :func:`numpy.pad`.
    shift0 : float
        Translation along the vertical axis (rows).
    shift1 : float
        Translation along the horizontal axis (columns).
    rotation : float
        Rotation angle in degrees (counter-clockwise).

    Returns
    -------
    numpy.ndarray
        The padded and transformed image.
    """
    padded = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
    return transform_2d(padded, shift0, shift1, rotation)

def transform_2d(image, shift0, shift1, rotation):
    """
    Transforms an image by applying translation and rotation in a single operation.
    
    Args:
        image (numpy.ndarray): Input image.
        shift0 (float): Translation along the vertical axis (rows).
        shift1 (float): Translation along the horizontal axis (columns).
        rotation (float): Rotation angle in degrees (counter-clockwise).
    
    Returns:
        numpy.ndarray: Transformed image.
    """
    rows, cols = image.shape[:2]
    center = (cols / 2, rows / 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation, scale=1.0)
    rotation_matrix_3x3 = np.vstack([rotation_matrix, [0, 0, 1]])

    translation_matrix = np.array([[1, 0, shift1], [0, 1, shift0], [0, 0, 1]], dtype=np.float32)
    combined_matrix = np.dot(rotation_matrix_3x3, translation_matrix)
    combined_matrix_2x3 = combined_matrix[:2, :]

    transformed_image = cv2.warpAffine(image, combined_matrix_2x3, (cols, rows))
    
    return transformed_image

def correlation_metric(img1, img2):
    """
    Calculate the similarity of two images based on correlation.
    If the images are not the same size, the smaller image is padded with zeros to match the larger image.
    
    Parameters:
        img1 (numpy.ndarray): The first image (grayscale or color).
        img2 (numpy.ndarray): The second image (grayscale or color).
    
    Returns:
        float: Correlation coefficient between the two images (value between -1 and 1).
    """

    h1, w1 = img1.shape
    h2, w2 = img2.shape

    max_h = max(h1, h2)
    max_w = max(w1, w2)

    padded_img1 = np.zeros((max_h, max_w), dtype=img1.dtype)
    padded_img2 = np.zeros((max_h, max_w), dtype=img2.dtype)

    padded_img1[:h1, :w1] = img1
    padded_img2[:h2, :w2] = img2

    flattened_img1 = padded_img1.flatten()
    flattened_img2 = padded_img2.flatten()

    correlation = np.corrcoef(flattened_img1, flattened_img2)[0, 1]

    return correlation

def align_to_reference_bruteforce(reference, align, rotation_range=(-10,10), rotation_step = 1, translation_range = (-5, 5), translation_step = 1, similarity_metric=None, translation_base = (0,0), rotation_base = 0):
    """
    Find the translation/rotation that best aligns one image to a reference.

    Exhaustively tries every combination of vertical shift, horizontal
    shift, and rotation in the given ranges (offset by ``translation_base``
    / ``rotation_base``), applies it to ``align`` with :func:`transform_2d`,
    and scores the result against ``reference`` using ``similarity_metric``.

    Parameters
    ----------
    reference : numpy.ndarray
        The reference image to align to.
    align : numpy.ndarray
        The image being aligned.
    rotation_range : tuple of int, optional
        ``(min, max)`` rotation offsets to search, in degrees.
    rotation_step : int, optional
        Step size between rotation candidates.
    translation_range : tuple of int, optional
        ``(min, max)`` translation offsets to search, in pixels, applied to
        both axes.
    translation_step : int, optional
        Step size between translation candidates.
    similarity_metric : Callable[[numpy.ndarray, numpy.ndarray], float]
        Function used to score how well an aligned candidate matches
        ``reference``. Higher is better, e.g. :func:`correlation_metric`.
    translation_base : tuple of int, optional
        ``(shift0, shift1)`` offset added to every candidate in
        ``translation_range``, typically the best transform found for the
        previous frame so the search stays centered on it.
    rotation_base : int, optional
        Rotation offset added to every candidate in ``rotation_range``.

    Returns
    -------
    tuple of (int, int, int)
        The ``(shift0, shift1, rotation)`` triple that maximizes
        ``similarity_metric``.
    """
    trans_range0 = range(translation_range[0] + translation_base[0], translation_range[1] + translation_base[0], translation_step)
    trans_range1 = range(translation_range[0] + translation_base[1], translation_range[1] + translation_base[0], translation_step)
    rot_range = range(rotation_range[0] + rotation_base, rotation_range[1] + rotation_base, rotation_step)
    scores = []
    for (shift0, shift1, rot) in itertools.product(trans_range0, trans_range1, rot_range):
        transformed = transform_2d(align, shift0, shift1, rot)
        score = similarity_metric(reference, transformed)
        scores.append(((shift0, shift1, rot), score))
        #print(scores[-1])
    return max(scores, key = lambda x: x[1])[0]

def calc_align_frames_transforms_bruteforce(frames, rotation_range = (-5,5), rotation_step = 1, translation_range = (-2, 2), translation_step = 1, similarity_metric=correlation_metric):
    """
    Compute a per-frame alignment transform chain via brute-force search.

    The first frame is left untransformed. Each subsequent frame is aligned
    to ``frames[0]`` with :func:`align_to_reference_bruteforce`, using the
    previous frame's transform as the search base so small drifts
    accumulate smoothly instead of being re-searched from scratch.

    Parameters
    ----------
    frames : list of numpy.ndarray
        Sequence of frames to align, in temporal order.
    rotation_range, rotation_step, translation_range, translation_step
        Search-space parameters forwarded to
        :func:`align_to_reference_bruteforce` for every frame pair.
    similarity_metric : Callable[[numpy.ndarray, numpy.ndarray], float], optional
        Scoring function, defaults to :func:`correlation_metric`.

    Returns
    -------
    list of tuple of (int, int, int)
        One ``(shift0, shift1, rotation)`` transform per input frame.
    """
    if len(frames) == 0:
        return []
    transforms = [(0, 0, 0)]
    for i in range(1, len(frames)):
        #print(f"aligning frame {i+1}/{len(frames)}")
        last_transform = transforms[-1]
        shift0, shift1, rotation = align_to_reference_bruteforce(frames[0], frames[i], rotation_range=rotation_range, rotation_step=rotation_step, translation_range=translation_range, translation_step=translation_step, similarity_metric=similarity_metric, translation_base = (last_transform[0], last_transform[1]), rotation_base = last_transform[2])
        transforms.append((shift0, shift1, rotation))
    return transforms

def align_frames_bruteforce(frames, transformed=None, rotation_range = (-5,5), rotation_step = 1, translation_range = (-2, 2), translation_step = 1, similarity_metric=correlation_metric):
    """
    Align a sequence of frames to their first frame via brute-force search.

    Transforms are computed from ``frames`` (typically a filtered/denoised
    version of the video used only to make the correlation search more
    robust) and then applied to ``transformed`` (typically the original,
    unfiltered frames), so alignment quality and output quality can be
    decoupled.

    Parameters
    ----------
    frames : list of numpy.ndarray
        Frames used to compute the alignment transforms.
    transformed : list of numpy.ndarray, optional
        Frames the computed transforms are applied to. Defaults to
        ``frames`` itself.
    rotation_range, rotation_step, translation_range, translation_step
        Search-space parameters forwarded to
        :func:`calc_align_frames_transforms_bruteforce`.
    similarity_metric : Callable[[numpy.ndarray, numpy.ndarray], float], optional
        Scoring function, defaults to :func:`correlation_metric`.

    Returns
    -------
    list of numpy.ndarray
        The aligned frames, same length and order as ``transformed``.
    """
    if transformed is None:
        transformed = frames
    if len(frames) == 0:
        return []
    assert len(frames) == len(transformed)
    transforms = calc_align_frames_transforms_bruteforce(frames, rotation_range = rotation_range, rotation_step = rotation_step, translation_range = translation_range, translation_step = translation_step, similarity_metric=similarity_metric)
    result = [transformed[0]]
    for i in range(1, len(frames)):
        shift0, shift1, rotation = transforms[i]
        result.append(transform_2d(transformed[i], shift0, shift1, rotation))
    return result

def BilateralFilter(a, b, c):
    """
    Build a bilateral-filter callable bound to fixed parameters.

    Convenience wrapper around :func:`cv2.bilateralFilter` for use as a
    transform in a pipeline (e.g. before brute-force alignment, where a
    denoised frame gives a more stable correlation score).

    Parameters
    ----------
    a : int
        Diameter of the pixel neighborhood used during filtering.
    b : float
        Filter sigma in the color space.
    c : float
        Filter sigma in the coordinate space.

    Returns
    -------
    Callable[[numpy.ndarray], numpy.ndarray]
        A function that applies ``cv2.bilateralFilter(image, a, b, c)``.
    """
    return lambda image: cv2.bilateralFilter(image, a,b,c)

def align_cell_video(cell_video):
    """
    Stabilize a :class:`~DuckSeg.experiment_evaluator.CellVideo` in place.

    Bilateral-filters the 8-bit frames to suppress noise, uses those
    filtered frames to compute a brute-force alignment (see
    :func:`align_frames_bruteforce`), and applies the resulting transforms
    to the video's original (non-filtered) frames.

    Parameters
    ----------
    cell_video : DuckSeg.experiment_evaluator.CellVideo
        The per-cell video to align.

    Returns
    -------
    list of numpy.ndarray
        The aligned original-resolution frames.
    """
    bolat = list(map(BilateralFilter(9, 75, 75), cell_video.frames_u8()))
    aligned = align_frames_bruteforce(bolat, transformed=cell_video.frames)
    return aligned
