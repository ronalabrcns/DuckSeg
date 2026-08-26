"""
Cell segmentation, per-cell tracking, and laser-ROI brightness scoring.

This is the core module of the DuckSeg pipeline. Its main entry point is
:func:`evalute_batch`, which segments every frame of every experiment in a
batch with CellSAM, tracks each detected cell across frames into a
:class:`CellVideo`, and caches the result to disk. From there:

- :class:`ExperimentEvaluator` drives the per-experiment segmentation and
  tracking (:meth:`~ExperimentEvaluator.create_starting_boxes` through
  :meth:`~ExperimentEvaluator.create_cell_video_inds3`) and is what
  :func:`evalute_batch` saves/loads from each experiment's cache.
- :class:`CellVideo` is a single tracked cell's cropped frames/masks over
  time, together with the laser ROI row range(s) it overlaps. Its
  :meth:`~CellVideo.calculate_brightness` family of methods compares
  fluorescence inside vs. outside the irradiated region — the core readout
  of a laser-irradiation ROI recruitment assay.
- The ``plot_*`` and ``calc_brightness_score``/``calculate_brightness_score_for_*``
  functions aggregate :class:`CellVideo` scores across many cells,
  experiments, or WT/KO batches for visualization.
- :class:`BoundingBox` is the shared rectangle type used throughout for
  CellSAM detections and cropping.

Frame-level mask generation itself (:func:`generate_mask`,
:func:`generate_mask_inner`) is CellSAM-backed and lazily loads the model
into ``cellsam_model`` on first use.
"""

import dataclasses
from copy import deepcopy
from DuckSeg.experiment_loader import ExperimentBatch, Experiment, ExperimentType
from DuckSeg.experiment_loader import generate_ROI_ranges as compute_roi_ranges
from typing import Optional
import numpy as np
from numpy import ndarray
import enum
from DuckSeg.aligner import transform_2d, BilateralFilter
import cv2
import torch

from cellSAM import segment_cellular_image, get_model
import matplotlib.pyplot as plt

cellsam_model = None


class ScoreType(enum.Enum):
    """
    How to combine in-ROI and out-of-ROI brightness into a single score.

    See :meth:`CellVideo.calculate_brightness_score`.
    """
    Diff = 0
    """Score as ``lasered - normal`` (absolute brightness difference)."""
    Ratio = 1
    """Score as ``lasered / normal`` (relative brightness fold-change)."""

@dataclasses.dataclass
class BoundingBox:
    """
    An axis-aligned rectangle, used for cell detections and image cropping.

    Coordinates follow the ``(row, column)`` convention used elsewhere in
    DuckSeg: ``x`` is the vertical/row axis (the axis the laser ROI ranges
    are defined along) and ``y`` is the horizontal/column axis. Instances
    are typically produced by :meth:`from_cellsam_bounding_box` from a raw
    CellSAM detection.

    Attributes
    ----------
    x_min, x_max : float
        Row range of the box.
    y_min, y_max : float
        Column range of the box.
    """
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def enlarged(self, percent: float):
        """
        Return a copy of this box enlarged symmetrically by a percentage.

        Parameters
        ----------
        percent : float
            Percentage to grow each dimension by, e.g. ``10`` grows width
            and height by 10% total (5% on each side).

        Returns
        -------
        BoundingBox
            The enlarged box, centered on the same point as ``self``.
        """
        x_diff = (self.x_max - self.x_min)*percent/100/2
        y_diff = (self.y_max - self.y_min)*percent/100/2
        return BoundingBox(self.x_min - x_diff, self.x_max + x_diff, self.y_min - y_diff, self.y_max + y_diff)

    def from_cellsam_bounding_box(box: list[float]):
        """
        Build a :class:`BoundingBox` from a raw CellSAM detection box.

        CellSAM returns boxes as ``[y_min, x_min, y_max, x_max]``; this
        reorders them into DuckSeg's ``(x_min, x_max, y_min, y_max)``
        convention.

        Parameters
        ----------
        box : list of float
            A CellSAM-format bounding box.

        Returns
        -------
        BoundingBox
        """
        return BoundingBox(box[1], box[3], box[0], box[2])

    def cellsam_bounding_box(self) -> list[float]:
        """Return this box in CellSAM's ``[y_min, x_min, y_max, x_max]`` format."""
        return [self.y_min, self.x_min, self.y_max, self.x_max]

    def cut(self, array):
        """
        Crop a 2D array to this box (clamped to non-negative indices).

        Parameters
        ----------
        array : numpy.ndarray
            The array to crop, e.g. a frame or mask.

        Returns
        -------
        numpy.ndarray
            The cropped region ``array[x_min:x_max, y_min:y_max]``.
        """
        x_min = max(0, int(self.x_min))
        x_max = max(0, int(self.x_max))
        y_min = max(0, int(self.y_min))
        y_max = max(0, int(self.y_max))
        return array[x_min:x_max,y_min:y_max]

    def as_indices(self):
        """
        Return a copy of this box with coordinates clamped to non-negative
        integers, suitable for direct use as array indices.

        Returns
        -------
        BoundingBox
        """
        x_min = max(0, int(self.x_min))
        x_max = max(0, int(self.x_max))
        y_min = max(0, int(self.y_min))
        y_max = max(0, int(self.y_max))
        return BoundingBox(x_min, x_max, y_min, y_max)

    def intersection_area(self, other: 'BoundingBox') -> float:
        """Return the overlapping area between this box and ``other`` (0 if disjoint)."""
        # Calculate the overlapping area
        x_overlap = max(0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        y_overlap = max(0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))

        # If the boxes don't overlap, either dimension will be zero
        return x_overlap * y_overlap
    def intersection_box(self, other: 'BoundingBox') -> 'BoundingBox':
        """Return the box representing the overlap between this box and ``other``."""
        x_min = max(self.x_min, other.x_min)
        x_max = min(self.x_max, other.x_max)
        y_min = max(self.y_min, other.y_min)
        y_max = min(self.y_max, other.y_max)
        return BoundingBox(x_min, x_max, y_min, y_max)

    def subtract_from(self, other: 'BoundingBox') -> 'BoundingBox':
        """
        Express ``other`` relative to this box's origin.

        Used to translate a box defined in full-frame coordinates (e.g. an
        intersection box) into coordinates local to a crop taken with
        :meth:`cut`, so it can index into the cropped array.

        Parameters
        ----------
        other : BoundingBox
            A box in the same coordinate space as ``self``.

        Returns
        -------
        BoundingBox
            ``other``'s coordinates minus this box's ``(x_min, y_min)`` origin.
        """
        x_min = other.x_min - self.x_min
        x_max = other.x_max - self.x_min
        y_min = other.y_min - self.y_min
        y_max = other.y_max - self.y_min
        return BoundingBox(x_min, x_max, y_min, y_max)


    def intersects_y_range(self, y_min: int, y_max: int) -> bool:
        """
        Test whether this box's row range overlaps ``(y_min, y_max)``.

        Despite the parameter names, this compares against ``self.x_min``/
        ``self.x_max`` (the row axis) — used to test whether a cell's
        bounding box overlaps a laser ROI range, which is itself expressed
        as a row range (see
        :func:`DuckSeg.experiment_loader.generate_ROI_ranges`).

        Parameters
        ----------
        y_min, y_max : int
            The row range to test against, e.g. one ROI range.

        Returns
        -------
        bool
        """
        return self.x_max > y_min and y_max > self.x_min

    def range_to_relative(self, range: tuple[float]) -> tuple[float]:
        """
        Express a row range relative to this box's ``x_min``.

        Parameters
        ----------
        range : tuple of float
            A ``(start, end)`` row range in full-frame coordinates.

        Returns
        -------
        tuple of float
            The same range, shifted so ``0`` corresponds to this box's
            ``x_min`` (i.e. the top row of a crop taken with :meth:`cut`).
        """
        return (range[0] - self.x_min, range[1] - self.x_min)

def evalute_batch(batch: ExperimentBatch, start = 1, transforms=None, use_cache=True):
    """
    Segment, track, and cache every experiment in a batch.

    For each experiment, this is the top-level driver of the DuckSeg
    pipeline: it runs :func:`evalute_experiment1` (CellSAM segmentation +
    cell tracking) and saves the resulting :class:`ExperimentEvaluator` to
    the experiment's on-disk cache via
    :meth:`~DuckSeg.experiment_loader.Experiment.save_data`. This is
    typically the slowest step in the notebook, since it runs the
    segmentation model on every frame of every experiment.

    Parameters
    ----------
    batch : DuckSeg.experiment_loader.ExperimentBatch
        The batch of experiments to process.
    start : int, optional
        Unused (reserved).
    transforms : list of Callable[[numpy.ndarray], numpy.ndarray], optional
        Preprocessing transforms applied to each frame before masking, e.g.
        from :mod:`DuckSeg.image_transforms`. Forwarded to
        :func:`evalute_experiment1`.
    use_cache : bool, optional
        If ``True`` (default), experiments that already have a cached
        :class:`ExperimentEvaluator` are loaded from disk instead of being
        re-processed.
    """
    for experiment in batch.experiments():
        print(f"Evaluating {experiment.name}")
        if use_cache:
            try:
                wt_eval = ExperimentEvaluator.load_experiment(experiment)
                print("Loaded from cache")
                continue
            except:
                pass
        wt_eval = evalute_experiment1(experiment, transforms=transforms)
        wt_eval.save(force=True)
        
def print_counts(arr):
    """Print each unique value in ``arr`` alongside its occurrence count. Debug helper."""
    unique_elements, counts = np.unique(arr, return_counts=True)

    # Print each element and its count
    for element, count in zip(unique_elements, counts):
        print(f"Element: {element}, Count: {count}")

def get_mask_num(arr):
    """
    Return the most common non-zero label in a labeled mask array.

    CellSAM masks encode each detected cell as a distinct integer label,
    with ``0`` reserved for background. Used to identify which label
    corresponds to a given cell after cropping the full-frame mask to its
    bounding box (see :meth:`ExperimentEvaluator.create_masked_cells`).

    Parameters
    ----------
    arr : numpy.ndarray
        A labeled mask array (e.g. a cropped region of a CellSAM mask).

    Returns
    -------
    int
        The most frequently occurring non-zero label, or ``0`` if the crop
        contains no non-zero pixels.
    """
    unique_elements, counts = np.unique(arr, return_counts=True)
    max_count = 0
    max_value = 0
    for element, count in zip(unique_elements, counts):
        if max_count < count and element != 0:
            max_count = count
            max_value = element
    return max_value

def generate_mask_inner(frame, bounding_boxes=None):
    """
    Run CellSAM segmentation on a single frame.

    Lazily loads and caches the CellSAM model in the module-level
    ``cellsam_model`` on first call. On a CUDA out-of-memory error, clears
    the GPU cache and retries once.

    Parameters
    ----------
    frame : numpy.ndarray
        A ``uint8`` grayscale frame (or an already-thresholded/transformed
        version of one).
    bounding_boxes : list, optional
        Previously known detection boxes (CellSAM format) to guide
        segmentation, e.g. from a prior frame.

    Returns
    -------
    tuple of (numpy.ndarray, list)
        The labeled segmentation mask and the list of detected bounding
        boxes (CellSAM ``[y_min, x_min, y_max, x_max]`` format).
    """
    global cellsam_model
    if cellsam_model is None:
        cellsam_model = get_model()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask, embedding, bounding_boxes = segment_cellular_image(frame, device=device.type, model = cellsam_model, bounding_boxes=bounding_boxes, bbox_threshold=0.7)
    except torch.cuda.OutOfMemoryError:
        """
        torch can run out of memory on the gpu, reset the session and try again
        """
        print("Out of memory. Restting cuda session!")
        #numba.cuda.select_device(0)
        #numba.cuda.close()
        #cellsam_model = get_model()
        torch.cuda.empty_cache()
        mask, embedding, bounding_boxes = segment_cellular_image(frame, device=device.type, model = cellsam_model, bounding_boxes=bounding_boxes)
    return mask, bounding_boxes

def generate_mask_transform(frame):
    """
    Segment a frame and return only its mask (drop the bounding boxes).

    A convenience wrapper around :func:`generate_mask_inner` with the
    ``Callable[[numpy.ndarray], numpy.ndarray]`` signature expected by a
    transform pipeline, so it can be passed directly to
    :func:`DuckSeg.image_transformer.apply_transform`.

    Parameters
    ----------
    frame : numpy.ndarray
        A ``uint8`` grayscale (or pre-thresholded) frame.

    Returns
    -------
    numpy.ndarray
        The labeled segmentation mask.
    """
    mask, _ = generate_mask_inner(frame)
    return mask

def generate_mask(frame, threshold_value = 1, bounding_boxes=None, transforms=None):
    """
    Preprocess a frame and run CellSAM segmentation on it.

    Parameters
    ----------
    frame : numpy.ndarray
        The frame to segment.
    threshold_value : int, optional
        If ``transforms`` is not given, the frame is binary-thresholded at
        this value (see :func:`DuckSeg.image_transforms.threshold_image`)
        before masking. Ignored if ``transforms`` is provided.
    bounding_boxes : list, optional
        Previously known detection boxes to guide segmentation, forwarded
        to :func:`generate_mask_inner`.
    transforms : list of Callable[[numpy.ndarray], numpy.ndarray], optional
        If given, these are applied to a copy of ``frame`` in order
        instead of the default thresholding step.

    Returns
    -------
    tuple of (numpy.ndarray, list)
        The labeled segmentation mask and detected bounding boxes; see
        :func:`generate_mask_inner`.
    """
    if transforms is not None:
        #print(f"Applying {len(transforms)} transforms")
        frame = deepcopy(frame)
        for transform in transforms:
            frame = transform(frame)
    elif threshold_value is not None:
        ret, thresh_img = cv2.threshold(frame, threshold_value, 255, cv2.THRESH_BINARY)
        frame = thresh_img*128
    return generate_mask_inner(frame, bounding_boxes=bounding_boxes)

@dataclasses.dataclass
class ExperimentEvaluator:
    """
    Runs and holds the per-frame segmentation and per-cell tracking results
    for a single :class:`~DuckSeg.experiment_loader.Experiment`.

    The pipeline stages are meant to be run in order — each populates
    fields consumed by the next:

    1. :meth:`create_starting_boxes` — segment the first frame.
    2. :meth:`filter_starting_boxes` — keep only cells overlapping the
       laser ROI.
    3. :meth:`create_boxes` — segment every frame of the video.
    4. :meth:`create_masked_cells` — crop and mask each detected cell in
       each frame.
    5. :meth:`create_cell_video_inds3` — track each starting cell across
       frames by matching masks.
    6. :meth:`cell_videos` — assemble the tracked cells into
       :class:`CellVideo` objects.

    :func:`evalute_experiment1` runs stages 1-5 in this order;
    :func:`evalute_batch` runs it per-experiment and persists the result
    via :meth:`save`/:meth:`load_experiment` so the notebook can reload
    results without re-running segmentation.

    Attributes
    ----------
    experiment : DuckSeg.experiment_loader.Experiment
        The experiment being evaluated.
    starting_boxes : list of BoundingBox, optional
        Every cell detected in the first frame. Set by
        :meth:`create_starting_boxes`.
    starting_mask : numpy.ndarray, optional
        The labeled segmentation mask of the first frame. Set by
        :meth:`create_starting_boxes`.
    filtered_starting_boxes : list of BoundingBox, optional
        The subset of ``starting_boxes`` overlapping a laser ROI range —
        the cells actually tracked and scored. Set by
        :meth:`filter_starting_boxes`.
    filtered_starting_inds : list of int, optional
        Indices into ``starting_boxes`` corresponding to
        ``filtered_starting_boxes``. Set by :meth:`filter_starting_boxes`.
    boxes : list of list of BoundingBox, optional
        Detected cell boxes for every frame, ``boxes[frame_index]`` being
        the boxes found in that frame. Set by :meth:`create_boxes`.
    masks : list of numpy.ndarray, optional
        The full-frame labeled segmentation mask for every frame. Set by
        :meth:`create_boxes`.
    cell_image_lists : list of list of numpy.ndarray, optional
        Cropped, masked cell images per frame, indexed the same way as
        ``boxes``. Set by :meth:`create_masked_cells`.
    cell_mask_lists : list of list of numpy.ndarray, optional
        Cropped boolean cell masks per frame, indexed the same way as
        ``boxes``. Set by :meth:`create_masked_cells`.
    cell_video_inds : list of list of tuple of (int, int), optional
        For each tracked starting cell, a per-frame ``(frame_box_list_index,
        box_index)`` pair identifying which detection in ``boxes``/
        ``cell_mask_lists`` corresponds to that cell in that frame. Set by
        :meth:`create_cell_video_inds3` (or the superseded
        :meth:`create_cell_video_inds`/:meth:`create_cell_video_inds2`).
    """
    experiment: Experiment
    # boxes for the starting frame
    starting_boxes: Optional[list[BoundingBox]] = None
    starting_mask: Optional[ndarray] = None
    # starting boxes intersecting the lines
    filtered_starting_boxes: Optional[list[BoundingBox]] = None
    filtered_starting_inds: Optional[list[int]] = None
    # Filtered (only ones intersecting the lines) boxes for each frame
    boxes: Optional[list[list[BoundingBox]]] = None
    masks: Optional[list[ndarray]] = None
    # Cell images grouped by frame
    cell_image_lists: Optional[list[list[ndarray]]] = None
    cell_mask_lists: Optional[list[list[ndarray]]] = None
    cell_video_inds: Optional[list[list[int]]] = None

    def create_starting_boxes(self, threshold_value = 1, transforms=None):
        """
        Segment the experiment's first frame and record the detected cells.

        Populates ``starting_boxes`` and ``starting_mask``. This defines
        the set of cells eligible for tracking — only cells present (and,
        after :meth:`filter_starting_boxes`, ROI-overlapping) in the first
        frame are followed through the rest of the video.

        Parameters
        ----------
        threshold_value : int, optional
            Forwarded to :func:`generate_mask` (ignored if ``transforms``
            is given).
        transforms : list of Callable[[numpy.ndarray], numpy.ndarray], optional
            Preprocessing transforms to apply before segmentation,
            forwarded to :func:`generate_mask`.
        """
        frame = self.experiment.first_frame_u8()
        mask, bounding_boxes = generate_mask(frame, threshold_value=threshold_value, transforms=transforms)
        self.starting_boxes = [BoundingBox.from_cellsam_bounding_box(box) for box in bounding_boxes]
        self.starting_mask = mask

    def filter_starting_boxes(self):
        """
        Restrict tracking to cells overlapping the laser ROI.

        Populates ``filtered_starting_boxes`` and ``filtered_starting_inds``
        from ``starting_boxes``, keeping only cells whose bounding box
        overlaps at least one of ``experiment.ROI_ranges()``. Must be
        called after :meth:`create_starting_boxes`.
        """
        ranges = self.experiment.ROI_ranges()
        self.filtered_starting_boxes = [box for box in self.starting_boxes if any(box.intersects_y_range(*laser_range) for laser_range in ranges)]
        self.filtered_starting_inds = [i for i, box in enumerate(self.starting_boxes) if any(box.intersects_y_range(*laser_range) for laser_range in ranges)]

    def create_boxes(self, limit = None, transforms=None):
        """
        Segment every frame of the experiment's video.

        Populates ``boxes`` and ``masks``, one entry per frame. This is
        typically the slowest step of the pipeline, since it runs CellSAM
        on every frame.

        Parameters
        ----------
        limit : int, optional
            If given, and ``boxes`` already has at least this many entries,
            return immediately without re-processing (cheap way to resume
            a partially completed run). Also stops segmentation after this
            many frames on a fresh run.
        transforms : list of Callable[[numpy.ndarray], numpy.ndarray], optional
            Preprocessing transforms to apply to each frame before
            segmentation, forwarded to :func:`generate_mask`.
        """
        experiment = self.experiment
        num_frames = len(experiment.frames_u8())
        if self.boxes is not None and limit is not None and len(self.boxes) >= limit:
            return
        boxes = []
        masks = []
        self.boxes = boxes
        self.masks = masks
        #last_boxes = self.filtered_starting_boxes
        for (i, frame) in enumerate(experiment.frames_u8()):
            if limit is not None and i == limit:
                break
            print(f"Running frame {i+1}/{num_frames}")
            # enlarge the last bounding boxes by 10% and pass it to cellsam
            #enlarged = [box.enlarged(10).cellsam_bounding_box() for box in last_boxes]
            mask, bounding_boxes = generate_mask(frame, bounding_boxes=None, transforms=transforms)
            masks.append(mask)
            boxes.append([BoundingBox.from_cellsam_bounding_box(box) for box in bounding_boxes])
            #last_boxes = boxes[-1]
    def create_masked_cells(self):
        """
        Crop and mask every detected cell in every frame.

        For each detected box in ``boxes``, crops the frame and mask to
        that box, resolves which mask label belongs to that specific cell
        with :func:`get_mask_num` (in case multiple labels fall within one
        crop), and stores the masked cell image and boolean mask.
        Populates ``cell_image_lists`` and ``cell_mask_lists``. Must be
        called after :meth:`create_boxes`.
        """
        self.cell_image_lists = []
        self.cell_mask_lists = []
        for i, (boxes, mask, frame) in enumerate(zip(self.boxes, self.masks, self.experiment.frames())):
            cell_images = []
            cell_masks = []
            for box in boxes:
                boxed_frame = box.cut(frame)
                boxed_mask = box.cut(mask)
                #print_counts(boxed_mask)
                mask_val = get_mask_num(boxed_mask)
                boxed_mask = boxed_mask == mask_val
                masked_boxed_frame = boxed_frame * boxed_mask
                cell_images.append(masked_boxed_frame)
                cell_masks.append(boxed_mask)
            self.cell_image_lists.append(cell_images)
            self.cell_mask_lists.append(cell_masks)
    def generate_ROI_ranges(self):
        """
        Compute the laser ROI row ranges local to each detected cell's crop.

        For every detected box in every frame, crops the ROI mask to that
        box and re-derives the ROI row ranges within the crop's own
        coordinate system. Populates ``cell_ROI_image_list`` and
        ``cell_ROI_range_list``. Not part of the main pipeline used by
        :func:`evalute_experiment1` — :class:`CellVideo` instead expands
        the experiment-level ROI ranges per cell on demand (see
        :meth:`CellVideo.expanded_ranges`).
        """
        self.cell_ROI_image_list = []
        self.cell_ROI_range_list = []
        for i, (boxes, mask, frame) in enumerate(zip(self.boxes, self.masks, self.experiment.frames())):
            image_list = []
            range_list = []
            for box in boxes:
                boxed_roi = box.cut(self.experiment.ROI_frame())
                boxed_roi_ranges = compute_roi_ranges(boxed_roi)
                image_list.append(boxed_roi)
                range_list.append(boxed_roi_ranges)
            self.cell_ROI_image_list.append(image_list)
            self.cell_ROI_range_list.append(range_list)
    def create_cell_video_inds(self):
        """
        Track each starting cell across frames by fixed-box overlap.

        For every ``filtered_starting_boxes`` entry, picks in each frame
        the detected box with the largest overlap against the *original*
        starting box. Superseded by :meth:`create_cell_video_inds3`, which
        tracks against the previous frame's position instead and is
        therefore far more robust to cell movement; kept for reference.

        Populates ``cell_video_inds`` with one list of per-frame box
        indices per tracked cell.
        """
        self.cell_video_inds = []
        for box in self.filtered_starting_boxes:
            inds = []
            for boxes in self.boxes:
                ind = np.argmax([float(box.intersection_area(next_box)) for next_box in boxes])
                inds.append(ind)
            self.cell_video_inds.append(inds)
    def create_cell_video_inds2(self):
        """
        Track each starting cell across frames by rolling-box overlap.

        Like :meth:`create_cell_video_inds`, but re-anchors the overlap
        comparison to the previously matched box each frame instead of the
        original starting box, so tracking follows cell movement. Superseded
        by :meth:`create_cell_video_inds3`, which uses mask overlap instead
        of box overlap for more accurate matching; kept for reference.

        Populates ``cell_video_inds`` with one list of per-frame box
        indices per tracked cell.
        """
        self.cell_video_inds = []
        for box in self.filtered_starting_boxes:
            curr_box = box
            inds = []
            for boxes in self.boxes:
                ind = np.argmax([float(curr_box.intersection_area(next_box)) for next_box in boxes])
                inds.append(ind)
                curr_box = boxes[ind]
            self.cell_video_inds.append(inds)
    def create_cell_video_inds3(self):
        """
        Track each starting cell across frames by rolling mask overlap.

        This is the tracking strategy used by :func:`evalute_experiment1`
        (the main pipeline). For every cell detected in the first frame
        that overlaps a laser ROI range, follows it frame by frame by
        picking, in each subsequent frame, the detection whose mask
        differs least from the previously matched mask (see
        :func:`mask_diff`) — more accurate than :meth:`create_cell_video_inds2`'s
        box-overlap heuristic since it accounts for the cell's actual
        shape, not just its bounding box. If no detection in a frame is a
        good match (mask difference exceeds 60% of the current mask's
        area), the previous frame's match is reused rather than jumping to
        an unrelated cell.

        Populates ``cell_video_inds`` with one list per tracked cell, each
        entry a ``(frame_box_list_index, box_index)`` pair used by
        :meth:`cell_video` to look up ``boxes``/``cell_mask_lists``.
        Must be called after :meth:`create_masked_cells`.
        """
        self.cell_video_inds = []
        ranges = self.experiment.ROI_ranges()
        i = 0
        for box, mask in zip(self.boxes[0], self.cell_mask_lists[0]):
            i += 1
            print(f"calculating {i}/{len(self.boxes[0])}")
            if not any(box.intersects_y_range(*laser_range) for laser_range in ranges):
                continue
            curr_box = box
            curr_mask = mask
            inds = []
            for (boxes, masks) in zip(self.boxes, self.cell_mask_lists):
                diffs = [float(mask_diff(curr_box, curr_mask, next_box, next_mask)) for next_box, next_mask in zip(boxes, masks)]
                ind = np.argmin(diffs)
                # if we didn't create a mask for a cell in this frame, use the prvious one
                if diffs[ind] > np.sum(curr_mask) * 0.6:
                    inds.append(inds[-1])
                else:
                    i_j = (len(inds), ind)
                    inds.append(i_j)
                    curr_box = boxes[ind]
                    curr_mask = masks[ind]
            self.cell_video_inds.append(inds)
    def show_cell_video(self, ind: int):
        """
        Display every frame of one tracked cell with matplotlib, in sequence.

        Parameters
        ----------
        ind : int
            Index into the tracked cells (i.e. into ``filtered_starting_boxes``
            / ``cell_video_inds``).

        Note
        ----
        Relies on :meth:`cell_frames`, which expects ``cell_video_inds``
        entries to be plain per-frame indices — the format produced by the
        superseded :meth:`create_cell_video_inds`/:meth:`create_cell_video_inds2`,
        not the ``(i, j)`` pairs produced by :meth:`create_cell_video_inds3`
        that the main pipeline actually uses. Prefer :meth:`cell_video`
        followed by :meth:`CellVideo.play_video` after running the standard
        pipeline.
        """
        for frame in self.cell_frames(ind):
            plt.imshow(frame)
            plt.show()

    def cell_frames(self, ind):
        """
        Return the cropped, masked frames of one tracked cell.

        Parameters
        ----------
        ind : int
            Index into the tracked cells.

        Returns
        -------
        list of numpy.ndarray

        Note
        ----
        See the note on :meth:`show_cell_video` regarding
        ``cell_video_inds`` format compatibility.
        """
        return [cell_image_list[j] for cell_image_list, j in zip(self.cell_image_lists, self.cell_video_inds[ind])]
    def play_cell_video(self, ind: int):
        """
        Play one tracked cell as an inline notebook video.

        Parameters
        ----------
        ind : int
            Index into the tracked cells.

        Note
        ----
        See the note on :meth:`show_cell_video` regarding
        ``cell_video_inds`` format compatibility.
        """
        frames = self.cell_frames(ind)
        play_nb_video(frames)
    def cell_video(self, ind: int) -> 'CellVideo':
        """
        Assemble one tracked cell's frames/masks/boxes into a :class:`CellVideo`.

        Parameters
        ----------
        ind : int
            Index into the tracked cells (i.e. into
            ``filtered_starting_boxes`` / ``cell_video_inds``).

        Returns
        -------
        CellVideo
            The cell's full-resolution, mask-multiplied frames across
            every tracked timepoint, with ID
            ``f"{experiment.name}_{ind}"``.
        """
        masks = []
        frames = []
        boxes = []
        exp_frames = self.experiment.frames()
        for real_ind, (i, j) in enumerate(self.cell_video_inds[ind]):
            frame = exp_frames[real_ind]
            box = self.boxes[i][j]
            boxed_mask = self.cell_mask_lists[i][j]
            boxed_frame = box.cut(frame)
            masked_boxed_frame = boxed_frame * boxed_mask
            frames.append(masked_boxed_frame)
            #frames.append(self.cell_image_lists[i][j])
            masks.append(self.cell_mask_lists[i][j])
            boxes.append(self.boxes[i][j])
        return CellVideo(id=f"{self.experiment.name}_{ind}", frames=frames, boxes=boxes, masks=masks, laser_ranges=self.experiment.ROI_ranges())
    def cell_videos(self) -> list['CellVideo']:
        """
        Assemble every tracked cell into a :class:`CellVideo`.

        This is the usual way analysis code and notebooks consume an
        :class:`ExperimentEvaluator`'s results — see the outlier-filtering
        and scoring stages of the example notebook.

        Returns
        -------
        list of CellVideo
            One :class:`CellVideo` per tracked cell; see :meth:`cell_video`.
        """
        return [self.cell_video(i) for i in range(len(self.cell_video_inds))]

    @staticmethod
    def from_other(other):
        """
        Copy the data fields of another :class:`ExperimentEvaluator`.

        Used by :meth:`load_experiment` to rebuild a class instance loaded
        from an older pickle, dropping any stale bound methods the pickle
        may carry and re-binding them to the current class definition.

        Parameters
        ----------
        other : ExperimentEvaluator
            The instance (e.g. freshly unpickled) to copy data from.

        Returns
        -------
        ExperimentEvaluator
            A new instance with ``other``'s non-method attributes copied
            over.
        """
        import types
        new = ExperimentEvaluator(other.experiment)
        for name, value in other.__dict__.items():
            # Check if the attribute is not a method
            if not isinstance(value, types.FunctionType):
                # Set the attribute in the destination class
                setattr(new, name, value)
        return new

    def save(self, force=False):
        """
        Persist this evaluator to the experiment's on-disk cache.

        Parameters
        ----------
        force : bool, optional
            If ``False`` (default), raises rather than overwriting an
            existing cached result. See
            :meth:`DuckSeg.experiment_loader.Experiment.save_data`.
        """
        self.experiment.save_data("evaluator", self, force=force)

    @staticmethod
    def load_experiment(experiment: Experiment):
        """
        Load a previously cached :class:`ExperimentEvaluator` for an experiment.

        This is how the example notebook reloads segmentation/tracking
        results computed by an earlier :func:`evalute_batch` call without
        re-running CellSAM.

        Parameters
        ----------
        experiment : DuckSeg.experiment_loader.Experiment
            The experiment to load cached results for. Note this is a
            fresh ``Experiment`` instance (e.g. from
            ``batch.experiments()``); the loaded evaluator's own
            ``experiment`` reference is replaced with it.

        Returns
        -------
        ExperimentEvaluator
            The cached evaluator, rebuilt via :meth:`from_other` to ensure
            its methods match the currently loaded class definition.
        """
        old = experiment.load_data("evaluator")
        old.experiment = experiment
        #old.cell_videos = ExperimentEvaluator.cell_videos
        return ExperimentEvaluator.from_other(old)

def mask_diff(box1, mask1, box2, mask2):
    """
    Measure how different two (possibly differently cropped) cell masks are.

    Computes the symmetric difference between the two masks in their
    overlapping region, plus the area of each mask that falls entirely
    outside the overlap. Used by
    :meth:`ExperimentEvaluator.create_cell_video_inds3` to decide which
    detection in the next frame is the same cell as the current match —
    the detection with the smallest ``mask_diff`` wins.

    Parameters
    ----------
    box1 : BoundingBox
        Bounding box of the first mask, in full-frame coordinates.
    mask1 : numpy.ndarray
        Boolean mask cropped to ``box1``.
    box2 : BoundingBox
        Bounding box of the second mask, in full-frame coordinates.
    mask2 : numpy.ndarray
        Boolean mask cropped to ``box2``.

    Returns
    -------
    int
        A dissimilarity score; ``0`` for identical masks at the same
        position, larger for masks that differ more in shape, position, or
        size.
    """
    box1 = box1.as_indices()
    box2 = box2.as_indices()
    intersection_box = box1.intersection_box(box2)
    intersection_in_mask1 = box1.subtract_from(intersection_box).cut(mask1)
    intersection_in_mask2 = box2.subtract_from(intersection_box).cut(mask2)
    common = np.sum(intersection_in_mask1 ^ intersection_in_mask2)
    m1 = np.sum(mask1) - np.sum(intersection_in_mask1)
    m2 = np.sum(mask2) - np.sum(intersection_in_mask2)
    return common + m1 + m2
    
def evalute_experiment1(experiment: Experiment, limit = None, transforms=None):
    """
    Run the full segmentation + tracking pipeline on one experiment.

    Runs, in order: :meth:`~ExperimentEvaluator.create_starting_boxes`,
    :meth:`~ExperimentEvaluator.filter_starting_boxes`,
    :meth:`~ExperimentEvaluator.create_boxes`,
    :meth:`~ExperimentEvaluator.create_masked_cells`, and
    :meth:`~ExperimentEvaluator.create_cell_video_inds3`. This is what
    :func:`evalute_batch` calls per experiment; call it directly to
    evaluate a single experiment without touching the on-disk cache.

    Parameters
    ----------
    experiment : DuckSeg.experiment_loader.Experiment
        The experiment to process.
    limit : int, optional
        Maximum number of frames to segment, forwarded to
        :meth:`~ExperimentEvaluator.create_boxes`. Useful for a quick
        partial run while tuning ``transforms``.
    transforms : list of Callable[[numpy.ndarray], numpy.ndarray], optional
        Preprocessing transforms applied before segmentation, e.g. from
        :mod:`DuckSeg.image_transforms`.

    Returns
    -------
    ExperimentEvaluator
        The populated evaluator, ready for :meth:`~ExperimentEvaluator.cell_videos`
        or :meth:`~ExperimentEvaluator.save`.
    """
    evaluator = ExperimentEvaluator(experiment)
    evaluator.create_starting_boxes(transforms=transforms)
    evaluator.filter_starting_boxes()
    evaluator.create_boxes(limit = limit, transforms=transforms)
    evaluator.create_masked_cells()
    #evaluator.create_cell_video_inds()
    #evaluator.create_cell_video_inds2()
    evaluator.create_cell_video_inds3()
    return evaluator

def zero_pad_bottom_right_2d(img, target_shape):
    """
    Zero-pad a 2D array on its bottom/right edges to reach a target shape.

    Parameters
    ----------
    img : numpy.ndarray
        2D array to pad.
    target_shape : tuple of (int, int)
        Desired ``(height, width)``. Must be at least as large as
        ``img.shape`` in each dimension.

    Returns
    -------
    numpy.ndarray
        ``img`` padded with zeros to ``target_shape``.
    """
    H, W = img.shape
    target_H, target_W = target_shape
    pad_bottom = max(0, target_H - H)
    pad_right = max(0, target_W - W)
    return np.pad(img, ((0, pad_bottom), (0, pad_right)), mode='constant')
    
@dataclasses.dataclass
class CellVideo:
    """
    One tracked cell's cropped frames and masks across a video, plus the
    laser ROI range(s) it overlaps.

    This is the unit of analysis for laser-irradiation brightness scoring:
    :meth:`calculate_brightness` and its ``calculate_brightness_*``/
    ``plot_brightness_*`` siblings compare fluorescence intensity inside
    the ROI row range against the rest of the cell, over time, which is
    the core readout of a laser-ROI recruitment assay. Instances are
    normally produced by :meth:`ExperimentEvaluator.cell_video` /
    :meth:`~ExperimentEvaluator.cell_videos`, not constructed directly.

    Attributes
    ----------
    id : str
        Identifier, conventionally ``f"{experiment.name}_{index}"``. Used
        throughout the pipeline (outlier filters, CSV export, plot labels)
        to refer to this specific cell.
    frames : list of numpy.ndarray
        Cropped, mask-multiplied frames, one per tracked timepoint.
    boxes : list of BoundingBox
        The full-frame bounding box this cell was cropped from, one per
        frame.
    masks : list of numpy.ndarray
        Boolean cell masks, aligned with ``frames``.
    laser_ranges : list of tuple of (float, float)
        Laser ROI row range(s), in full-frame coordinates, that this
        cell's first-frame box overlaps. Set from
        ``experiment.ROI_ranges()`` and filtered down to just the
        overlapping ranges in :meth:`__post_init__`.
    """
    id: str
    frames: list[ndarray]
    boxes: list[BoundingBox]
    masks: list[ndarray]
    laser_ranges: list[tuple[float]]

    def map(self, fn) -> 'CellVideo':
        """
        Return a copy of this cell video with a function applied to every frame.

        Parameters
        ----------
        fn : Callable[[numpy.ndarray], numpy.ndarray]
            Function applied to a copy of each frame.

        Returns
        -------
        CellVideo
            A new instance with transformed frames; ``masks``, ``boxes``,
            and ``laser_ranges`` are carried over unchanged.
        """
        return CellVideo(id=self.id, frames=[fn(frame.copy()) for frame in self.frames], boxes=self.boxes, masks = self.masks, laser_ranges = self.laser_ranges)

    def homogenize_size(self) -> 'CellVideo':
        """
        Return a copy of this cell video with every frame/mask zero-padded
        to the same (maximum) size.

        Different frames can have slightly different crop sizes since each
        frame's bounding box is detected independently. Padding to a
        common size is required before frames can be stacked into a single
        array, e.g. for :func:`combine_cell_videos` or
        :func:`DuckSeg.cell_video_visualizer.normalize_videos_for_display`.

        Returns
        -------
        CellVideo
        """
        shape0 = max(frame.shape[0] for frame in self.frames)
        shape1 = max(frame.shape[1] for frame in self.frames)
        shape = (shape0, shape1)
        return CellVideo(id=self.id, frames = [zero_pad_bottom_right_2d(frame, shape) for frame in self.frames], masks = [zero_pad_bottom_right_2d(mask, shape) for mask in self.masks], laser_ranges = self.laser_ranges, boxes = self.boxes)

    def transform_2d(self, transforms: list[tuple[float, float, float]]) -> 'CellVideo':
        """
        Return a copy of this cell video with a per-frame 2D transform applied.

        Parameters
        ----------
        transforms : list of tuple of (float, float, float)
            One ``(shift0, shift1, rotation)`` triple per frame, as
            produced by e.g. :func:`DuckSeg.aligner.calc_align_frames_transforms_bruteforce`.
            See :func:`DuckSeg.aligner.transform_2d`.

        Returns
        -------
        CellVideo
            A new instance with transformed ``frames`` and ``masks``.
        """
        new_frames = [transform_2d(frame, *transform) for (frame, transform) in zip(self.frames, transforms)]
        new_masks = [transform_2d(mask.astype(np.uint8), *transform) > 0 for (mask, transform) in zip(self.masks, transforms)]
        return CellVideo(self.id, new_frames, self.boxes, new_masks, self.laser_ranges)

    def calc_max_mask_diff(self) -> int:
        """
        Return the largest frame-to-frame mask change (XOR pixel count)
        anywhere in this video.

        A quality metric: a large jump usually indicates a tracking or
        segmentation failure between two frames. See
        :meth:`max_relative_mask_diff` for a size-normalized version, and
        :func:`border_draw_mask_diff` for using this to flag suspect
        videos visually.

        Returns
        -------
        int
        """
        diffs = []
        for i in range(len(self.masks)-1):
            m1, m2 = self.masks[i], self.masks[i+1]
            size0 = max((m1.shape[0], m2.shape[0]))
            size1 = max((m1.shape[1], m2.shape[1]))
            arr1 = np.zeros((size0, size1),np.bool_)
            arr1[:m1.shape[0], :m1.shape[1]] = m1
            arr2 = np.zeros((size0, size1),np.bool_)
            arr2[:m2.shape[0], :m2.shape[1]] = m2
            diffs.append(np.sum(arr1^arr2))
        return max(diffs)
    def calc_min_corr(self, blur=None) -> float:
        """
        Return the lowest frame-to-frame correlation anywhere in this video.

        An alternative quality metric to :meth:`calc_max_mask_diff`: frames
        are blurred (bilateral filter by default, to reduce noise
        sensitivity) and correlated pairwise; a low correlation between
        consecutive frames suggests misalignment or a tracking failure.

        Parameters
        ----------
        blur : Callable[[numpy.ndarray], numpy.ndarray], optional
            Denoising filter applied to each frame before correlating.
            Defaults to ``DuckSeg.aligner.BilateralFilter(9, 75, 75)``.

        Returns
        -------
        float
            The minimum pairwise correlation coefficient between
            consecutive frames.
        """
        corrs = []
        if blur is None:
            blur = BilateralFilter(9, 75, 75)
        for i in range(len(self.masks)-1):
            m1, m2 = blur(self.frames[i]), blur(self.frames[i+1])
            size0 = max((m1.shape[0], m2.shape[0]))
            size1 = max((m1.shape[1], m2.shape[1]))
            arr1 = np.zeros((size0, size1),np.bool_)
            arr1[:m1.shape[0], :m1.shape[1]] = m1
            arr2 = np.zeros((size0, size1),np.bool_)
            arr2[:m2.shape[0], :m2.shape[1]] = m2
            corrs.append(np.corrcoef(arr1.flatten(), arr2.flatten()))
        return min(corrs)

    def max_element_count(self) -> int:
        """Return the size (pixel count) of the largest mask in this video."""
        return max(mask.size for mask in self.masks)
    def frame_u8(self, i: int):
        """Return frame ``i`` rescaled from 16-bit to 8-bit grayscale."""
        return (self.frames[i]/(2.**16-1)*255).astype(np.uint8)
    def frames_u8(self):
        """Return every frame rescaled from 16-bit to 8-bit grayscale."""
        return [self.frame_u8(i) for i in range(len(self.frames))]

    def max_relative_mask_diff(self) -> float:
        """
        Return :meth:`calc_max_mask_diff` normalized by :meth:`max_element_count`.

        A size-independent quality metric in ``[0, 1]``-ish range, making
        it comparable across cells of different sizes; used by
        :func:`border_draw_mask_diff` with a fixed ``0.3`` threshold to
        flag likely tracking failures.

        Returns
        -------
        float
        """
        return self.calc_max_mask_diff() / self.max_element_count()
    def frame_count(self) -> int:
        """Return the number of frames in this video."""
        return len(self.frames)

    def play_video(self):
        """Play this cell video as an inline notebook video widget."""
        frames = self.frames
        play_nb_video(self.frames)

    def __post_init__(self):
        """Restrict ``laser_ranges`` to the ranges the first frame's box actually overlaps."""
        self.laser_ranges = [range for range in self.laser_ranges if self.boxes[0].intersects_y_range(*range)]
        assert len(self.laser_ranges) != 0, "ROI ranges modified after evaluate_batch!"

    def expanded_ranges(self, modifier):
        """
        Return ``laser_ranges`` symmetrically widened by a multiplicative factor.

        Widening the ROI range compensates for imprecision in the drawn
        ROI mask and for cell/laser-spot movement, so brightness scoring
        (see :meth:`calculate_brightness`) captures the full irradiated
        region even if it drifted slightly from the mask.

        Parameters
        ----------
        modifier : float
            Multiplicative width factor; ``1`` leaves ranges unchanged,
            ``> 1`` widens them (e.g. ``4`` used by default throughout the
            scoring functions), ``< 1`` narrows them.

        Returns
        -------
        list of tuple of (float, float)
        """
        res = []
        for range in self.laser_ranges:
            diff = (range[1] - range[0])*(modifier-1)/2
            res.append((range[0]-diff, range[1]+diff))
        return res
    def video_with_ranges(self, range_modifier=1):
        """
        Return copies of every frame with the laser ROI row range blacked out.

        Visualization helper for confirming the ROI range used in scoring
        lines up with the visible laser damage; see :meth:`play_video_with_ranges`.

        Parameters
        ----------
        range_modifier : float, optional
            Width factor forwarded to :meth:`expanded_ranges`.

        Returns
        -------
        list of numpy.ndarray
            Frame copies with rows inside the (expanded) ROI range set to 0.
        """
        ranges = self.expanded_ranges(range_modifier)
        copied_frames = [frame.copy() for frame in self.frames]
        for range in ranges:
            range = self.boxes[0].range_to_relative(range)
            for frame in copied_frames:
                frame[[max([int(range[0]), 0]),min([int(range[1]), frame.shape[0]-1])],:] = 0
        return copied_frames

    def play_video_with_ranges(self, range_modifier=1):
        """
        Play :meth:`video_with_ranges` as an inline notebook video widget.

        Parameters
        ----------
        range_modifier : float, optional
            Forwarded to :meth:`video_with_ranges`.
        """
        frames = self.video_with_ranges(range_modifier=range_modifier)
        play_nb_video(frames)

    def calculate_brightness(self, range_modifier=1):
        """
        Compute per-frame mean fluorescence inside vs. outside the laser ROI.

        This is the core measurement of a laser-irradiation recruitment
        assay: it compares the average pixel intensity within the
        (expanded) irradiated row range against the rest of the cell, for
        every frame, giving a time series of whether a fluorescent protein
        is accumulating at the damage site.

        Only the first ``laser_ranges`` entry is used — this method assumes
        (but does not enforce beyond a no-op check) that a cell overlaps a
        single laser ROI range.

        Parameters
        ----------
        range_modifier : float, optional
            Width factor forwarded to :meth:`expanded_ranges`; wider values
            include more of the cell margin around the nominal ROI.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(frame_count, 2)``; column 0 is the
            mean intensity outside the ROI ("normal"), column 1 is the
            mean intensity inside the ROI ("lasered"), per frame. Values
            are normalized by mask pixel count, so they are directly
            comparable to :meth:`calculate_brightness_diff` /
            :meth:`calculate_brightness_ratio` regardless of cell size.
        """
        ranges = self.expanded_ranges(range_modifier)
        #assert len(ranges) == 1
        if len(ranges) != 1:
            pass
            #print(f"Multiple ranges {ranges}")
        range = self.boxes[0].range_to_relative(ranges[0])
        start, end = (int(range[0]), int(range[1]))
        brighnesses = []
        for mask, frame in zip(self.masks, self.frames):
            lasered = np.sum(frame[start:end,:]) / np.sum(mask[start:end,:])
            normal = (np.sum(frame[:start,:]) + np.sum(frame[end:,:])) / (np.sum(mask[:start,:]) + np.sum(mask[end:,:]))
            brighnesses.append((normal, lasered))
        return np.array(brighnesses)
    def plot_brightness(self, range_modifier=1):
        """
        Plot the raw ``(normal, lasered)`` brightness time series with matplotlib.

        Parameters
        ----------
        range_modifier : float, optional
            Forwarded to :meth:`calculate_brightness`.
        """
        plt.plot(self.calculate_brightness(range_modifier=range_modifier))
        plt.show()
    def plot_brightness_diff(self, range_modifier=1):
        """
        Plot :meth:`calculate_brightness_diff` with matplotlib.

        Parameters
        ----------
        range_modifier : float, optional
            Forwarded to :meth:`calculate_brightness_diff`.
        """
        diff = self.calculate_brightness_diff(range_modifier=range_modifier)
        plt.plot(diff)
        plt.show()
    def calculate_brightness_diff(self, range_modifier=1):
        """
        Compute the per-frame ``lasered - normal`` brightness difference.

        Parameters
        ----------
        range_modifier : float, optional
            Forwarded to :meth:`calculate_brightness`.

        Returns
        -------
        numpy.ndarray
            1D array, one value per frame.
        """
        vals = self.calculate_brightness(range_modifier=range_modifier)
        diff = vals[:,1] - vals[:,0]
        return diff
    def calculate_brightness_ratio(self, range_modifier=1):
        """
        Compute the per-frame ``lasered / normal`` brightness ratio.

        Parameters
        ----------
        range_modifier : float, optional
            Forwarded to :meth:`calculate_brightness`.

        Returns
        -------
        numpy.ndarray
            1D array, one value per frame.
        """
        vals = self.calculate_brightness(range_modifier=range_modifier)
        diff = vals[:,1]/vals[:,0]
        return diff
    def calculate_brightness_score(self, score: ScoreType, range_modifier=1):
        """
        Compute the per-frame brightness score using the requested combination.

        Parameters
        ----------
        score : ScoreType
            ``ScoreType.Diff`` for :meth:`calculate_brightness_diff` or
            ``ScoreType.Ratio`` for :meth:`calculate_brightness_ratio`.
        range_modifier : float, optional
            Forwarded to :meth:`calculate_brightness`.

        Returns
        -------
        numpy.ndarray
            1D array, one value per frame.

        Raises
        ------
        AssertionError
            If ``score`` is not a recognized :class:`ScoreType`.
        """
        if score == ScoreType.Diff:
            return self.calculate_brightness_diff(range_modifier=range_modifier)
        elif score == ScoreType.Ratio:
            return self.calculate_brightness_ratio(range_modifier=range_modifier)
        else:
            raise AssertionError

    def plot_brightness_ratio(self, range_modifier=1):
        """
        Plot :meth:`calculate_brightness_ratio` with matplotlib.

        Parameters
        ----------
        range_modifier : float, optional
            Forwarded to :meth:`calculate_brightness_ratio`.
        """
        diff = self.calculate_brightness_ratio(range_modifier=range_modifier)
        plt.plot(diff)
        plt.show()
    def max_size(self) -> tuple[int, int]:
        """Return the ``(height, width)`` of the largest frame in this video."""
        return (max(frame.shape[0] for frame in self.frames), max(frame.shape[1] for frame in self.frames))


def combine_cell_videos(videos: list[CellVideo], output_width = 1024, extra_draw=None, cell_filter=lambda x: True):
    """
    Tile multiple cell videos into a single video, packed left-to-right/top-to-bottom.

    Cells are packed greedily in the order given, wrapping to a new row
    once ``output_width`` would be exceeded (a simple shelf/bin-packing
    layout, not size-optimized). Used by :func:`combine_batch_videos` to
    build one overview video per experiment.

    Parameters
    ----------
    videos : list of CellVideo
        Cell videos to combine. All are included; filter the list before
        calling if only a subset should appear (``cell_filter`` is
        accepted for API symmetry with :func:`combine_batch_videos` but is
        not applied here).
    output_width : int, optional
        Maximum pixel width of a row before wrapping.
    extra_draw : Callable[[numpy.ndarray, CellVideo], None], optional
        If given, called for every ``(frame_region, video)`` pair after
        placing it, e.g. :func:`border_draw_mask_diff` to flag low-quality
        tracking with a border.
    cell_filter : Callable[[CellVideo], bool], optional
        Unused in this function; accepted for signature symmetry with
        :func:`combine_batch_videos`.

    Returns
    -------
    numpy.ndarray
        A ``(frame_count, output_width, output_height)`` ``uint16`` array
        containing every video's frames placed on the tiled canvas.
    """
    sizes = [vid.max_size() for vid in videos]
    y = 0
    x = 0
    next_y = 0
    frame_count = max(vid.frame_count() for vid in videos)
    positions = []
    y_sizes = [0]
    y_inds = [0]
    for (x_size, y_size) in sizes:
        assert output_width > x_size
        if x + x_size < output_width:
            positions.append((x, y))
            x += x_size
            next_y = max((next_y, y + y_size))
            y_sizes[-1] = max((y_sizes[-1], y_size))
        else:
            y = next_y
            x = x_size
            next_y = next_y + y_size
            positions.append((0, y))
            y_sizes.append(y_size)
        y_inds.append(len(y_sizes)-1)
    output_height = next_y
    frames = np.zeros((frame_count, output_width, output_height), np.uint16)
    for i in range(len(frames)):
        for (vid, pos, y_ind) in zip(videos, positions, y_inds):
            y_size = y_sizes[y_ind]
            if i >= vid.frame_count():
                continue
            vid_frame = vid.frames[i]
            frames[i, pos[0]:pos[0]+vid_frame.shape[0], pos[1]:pos[1]+vid_frame.shape[1]] = vid_frame
            if extra_draw is not None:
                extra_draw(frames[i, pos[0]:pos[0]+vid_frame.shape[0], pos[1]:pos[1]+y_size], vid)
    return frames

def text_to_ndarray(text, array_size, dtype, font_scale=1, thickness=2, font_color=None):
    """
    Generate a grayscale ndarray with specified text.

    Parameters:
        text (str): The text to be added to the ndarray.
        array_size (tuple): Size of the array (height, width).
        font_scale (int, optional): Scale of the font. Defaults to 1.
        thickness (int, optional): Thickness of the font. Defaults to 2.

    Returns:
        ndarray: Grayscale image array with text.
    """
    # Step 1: Create a blank grayscale image with the specified size
    blank_image = np.zeros((array_size[0], array_size[1]), dtype=dtype)
    
    # Step 2: Define font and position for text
    font = cv2.FONT_HERSHEY_SIMPLEX
    if font_color is None:
        font_color = np.iinfo(dtype).max  # White color in grayscale

    # Calculate text size to center the text
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (blank_image.shape[1] - text_size[0]) // 2
    text_y = (blank_image.shape[0] + text_size[1]) // 2
    position = (text_x, text_y)

    # Step 3: Add text to the blank image
    cv2.putText(blank_image, text, position, font, font_scale, font_color, thickness)
    
    return blank_image

def add_title_to_video(array, title: str, font_scale=1, thickness=2, height=30) -> np.ndarray:
    """
    Prepend a text title banner to every frame of a video array.

    Parameters
    ----------
    array : numpy.ndarray
        Video array of shape ``(frame_count, height, width)``, e.g. from
        :func:`combine_cell_videos`.
    title : str
        Text to render, centered in the banner.
    font_scale : float, optional
        Text scale, forwarded to :func:`text_to_ndarray`.
    thickness : int, optional
        Text stroke thickness, forwarded to :func:`text_to_ndarray`.
    height : int, optional
        Height in pixels of the title banner.

    Returns
    -------
    numpy.ndarray
        A new array of shape ``(frame_count, height + array.shape[1],
        width)`` with the title banner stacked above every frame.
    """
    array_size = (height, array.shape[2])
    text = text_to_ndarray(title, array_size, array.dtype, font_scale=font_scale, thickness=thickness, font_color=int(np.max(array)))
    return np.stack([np.concatenate((text, frame), axis=0) for frame in array], axis=0)

def border_draw_mask_diff(frame, vid):
    """
    Draw a warning border on ``frame`` if ``vid``'s tracking looks unstable.

    An ``extra_draw`` callback for :func:`combine_cell_videos`/
    :func:`combine_batch_videos`: flags cells whose
    :meth:`CellVideo.max_relative_mask_diff` exceeds ``0.3`` (a fixed
    threshold), which usually indicates a segmentation or tracking failure
    between two frames, worth a human's attention when scanning the
    combined overview video.

    Parameters
    ----------
    frame : numpy.ndarray
        The frame region to draw on, modified in place.
    vid : CellVideo
        The cell video ``frame`` belongs to.
    """
    if vid.max_relative_mask_diff() > 0.3:
        draw_border(frame, 3, intensity=np.max(frame))

def draw_border(array, border_width, intensity=255):
    """
    Draw a border with configurable width around a 2D ndarray.
    
    Parameters:
        array (ndarray): The 2D array where the border will be drawn.
        border_width (int): The width of the border.
        intensity (int): The intensity (brightness) of the border. Default is 255 (white).
        
    Returns:
        ndarray: The array with the border drawn.
    """
    # Ensure the border width is valid
    rows, cols = array.shape
    if border_width * 2 > rows or border_width * 2 > cols:
        raise ValueError("Border width is too large for the given array dimensions.")

    # Set the top and bottom border
    array[:border_width, :] = intensity           # Top border
    array[-border_width:, :] = intensity          # Bottom border

    # Set the left and right border
    array[:, :border_width] = intensity           # Left border
    array[:, -border_width:] = intensity          # Right border

    return array

def combine_batch_videos(batch: ExperimentBatch, extra_draw=None, cell_filter=lambda x: True):
    """
    Build one tiled overview video per experiment and concatenate them side by side.

    Loads each experiment's cached :class:`ExperimentEvaluator` (see
    :meth:`ExperimentEvaluator.load_experiment`), tiles its cell videos
    with :func:`combine_cell_videos`, and labels the result with the
    experiment's :meth:`~DuckSeg.experiment_loader.Experiment.in_batch_id`.

    Parameters
    ----------
    batch : DuckSeg.experiment_loader.ExperimentBatch
        The batch to visualize. Requires every experiment to already have
        a cached evaluator (i.e. :func:`evalute_batch` has been run).
    extra_draw : Callable[[numpy.ndarray, CellVideo], None], optional
        Forwarded to :func:`combine_cell_videos`, e.g.
        :func:`border_draw_mask_diff`.
    cell_filter : Callable[[CellVideo], bool], optional
        Predicate used to compute a filtered cell-video list per
        experiment (kept for future use).

    Returns
    -------
    numpy.ndarray
        A single video array with every experiment's tiled overview
        concatenated along the width axis.
    """
    experiments = batch.experiments()
    exp_videos = []
    for exp in experiments:
        print(exp.name)
        cell_videos = ExperimentEvaluator.load_experiment(exp).cell_videos()
        cell_videos = [video for video in cell_videos if cell_filter(video)]
        vid = combine_cell_videos(ExperimentEvaluator.load_experiment(exp).cell_videos(), extra_draw=extra_draw, cell_filter=cell_filter)
        titled_vid = add_title_to_video(vid, exp.in_batch_id(), height=100)
        exp_videos.append(titled_vid)
    #exp_videos = [combine_cell_videos(ExperimentEvaluator.load_experiment(exp).cell_videos()) for exp in experiments[:2]]
    vid = np.concatenate(exp_videos, axis=2)
    #play_nb_video(vid)
    return vid

def calc_brightness_score(batch: ExperimentBatch, score_type: ScoreType = ScoreType.Ratio, range_modifier=4, cell_filter=lambda x: True, experiment_type = None, print_progress=False):
    """
    Compute per-cell brightness scores for a batch, optionally restricted
    to one experimental group.

    Parameters
    ----------
    batch : DuckSeg.experiment_loader.ExperimentBatch
        The batch to score. Requires cached evaluators (see
        :func:`evalute_batch`).
    score_type : ScoreType, optional
        How to combine in-ROI/out-of-ROI brightness; see
        :meth:`CellVideo.calculate_brightness_score`.
    range_modifier : float, optional
        ROI width factor forwarded to
        :meth:`CellVideo.calculate_brightness`.
    cell_filter : Callable[[CellVideo], bool], optional
        Predicate selecting which cells to include, e.g. the result of
        :func:`DuckSeg.outlier_filters.outlier_vids` inverted into a
        keep-filter.
    experiment_type : DuckSeg.experiment_loader.ExperimentType, optional
        If given, restricts to :meth:`~DuckSeg.experiment_loader.ExperimentBatch.wt_experiments`
        or :meth:`~DuckSeg.experiment_loader.ExperimentBatch.ko_experiments`;
        otherwise every experiment in the batch is used.
    print_progress : bool, optional
        If ``True``, print each experiment's name as it's processed.

    Returns
    -------
    list of list of numpy.ndarray
        One list per experiment, each containing one score time series per
        (filtered) cell.
    """
    if experiment_type == ExperimentType.WT:
        experiments = batch.wt_experiments()
    elif experiment_type == ExperimentType.KO:
        experiments = batch.ko_experiments()
    else:
        experiments = batch.experiments()
    return calculate_brightness_score_for_experiments(
            experiments,
            score_type=score_type,
            range_modifier=range_modifier,
            cell_filter=cell_filter,
            print_progress=print_progress,
        )
def calculate_brightness_score_for_experiments(experiments: list[Experiment], score_type: ScoreType = ScoreType.Ratio, range_modifier=4, cell_filter=lambda x: True, print_progress=False):
    """
    Compute per-cell brightness scores for an explicit list of experiments.

    Loads each experiment's cached :class:`ExperimentEvaluator` and scores
    every cell that passes ``cell_filter``. Called by
    :func:`calc_brightness_score` after it resolves which experiments to
    include; call this directly to score an arbitrary experiment subset.

    Parameters
    ----------
    experiments : list of DuckSeg.experiment_loader.Experiment
        Experiments to score. Requires cached evaluators.
    score_type : ScoreType, optional
        See :meth:`CellVideo.calculate_brightness_score`.
    range_modifier : float, optional
        ROI width factor forwarded to
        :meth:`CellVideo.calculate_brightness`.
    cell_filter : Callable[[CellVideo], bool], optional
        Predicate selecting which cells to include.
    print_progress : bool, optional
        If ``True``, print each experiment's name as it's processed.

    Returns
    -------
    list of list of numpy.ndarray
        One list per experiment, each containing one score time series per
        (filtered) cell.
    """
    all_scores = []
    all_diffs = []
    for experiment in experiments:
        if print_progress:
            print(experiment.name)
        eval = ExperimentEvaluator.load_experiment(experiment)
        diffs = []
        for cell_video in eval.cell_videos():
            if not cell_filter(cell_video):
                continue
            diffs.append(cell_video.calculate_brightness_score(score_type, range_modifier=range_modifier))
        all_diffs.append(diffs)
    return all_diffs

def calculate_brightness_score_for_videos(videos: list[CellVideo], score_type: ScoreType = ScoreType.Ratio, range_modifier=4, cell_filter=lambda x: True, print_progress=False):
    """
    Compute brightness scores for an explicit list of already-loaded cell videos.

    Unlike :func:`calc_brightness_score`/:func:`calculate_brightness_score_for_experiments`,
    this does not load anything from disk — used by the example notebook
    on the ``vids`` list already collected across the whole batch, so
    per-cell IDs can be tracked alongside their scores (e.g. for
    :func:`plot_all_vid_scores`, which labels each line by ID).

    Parameters
    ----------
    videos : list of CellVideo
        Cell videos to score.
    score_type : ScoreType, optional
        See :meth:`CellVideo.calculate_brightness_score`.
    range_modifier : float, optional
        ROI width factor forwarded to
        :meth:`CellVideo.calculate_brightness`.
    cell_filter : Callable[[CellVideo], bool], optional
        Predicate selecting which cells to include.
    print_progress : bool, optional
        Unused; accepted for signature symmetry with
        :func:`calc_brightness_score`.

    Returns
    -------
    tuple of (list of str, list of numpy.ndarray)
        The IDs of the videos that passed ``cell_filter``, and their
        corresponding score time series, in matching order.
    """
    diffs = []
    vid_ids = []
    for vid in videos:
        if cell_filter(vid):
            vid_ids.append(vid.id)
            diffs.append(vid.calculate_brightness_score(score_type, range_modifier=range_modifier))
    return vid_ids, diffs

def score_avg(diffb):
    """
    Compute the mean and standard error of a collection of score time series.

    NaN-containing series (e.g. from a division by zero when a mask has no
    pixels on one side of the ROI) are dropped before averaging.

    Parameters
    ----------
    diffb : list of numpy.ndarray
        Per-cell score time series, all the same length, e.g. from
        :func:`calculate_brightness_score_for_videos`.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        The per-timepoint mean and standard error across series (NaN
        series excluded).

    Raises
    ------
    ValueError
        If every series is NaN-only, or series have inconsistent lengths
        (raised by :func:`numpy.vstack`).
    """
    for diffa in diffb:
        try:
            np.any(np.isnan(diffa))
        except:
            print(f"Diffa {diffa}")
            print(f"Diffb {diffb}")
            raise
    diffs = np.vstack([diffa for diffa in diffb if not np.any(np.isnan(diffa))])
    mean = np.mean(diffs, axis=0)
    std = np.std(diffs, ddof=1, axis=0)
    se = std/diffs.shape[0]**0.5
    return (mean, se)
def plot_score_avg(diffs, show = True, color='blue', label=None):
    """
    Plot the mean +/- SEM of a collection of score time series.

    Parameters
    ----------
    diffs : list of numpy.ndarray
        Per-cell score time series; see :func:`score_avg`.
    show : bool, optional
        If ``True`` (default), call ``plt.show()`` after plotting. Set to
        ``False`` when overlaying multiple series on one figure (see
        :func:`plot_scores_avg`).
    color : str, optional
        Line and shaded-region color.
    label : str, optional
        Legend label for this series.
    """
    mean, sem = score_avg(diffs)
    x = range(len(mean))
    plt.plot(x, mean, label=label, color=color)
    plt.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.3)
    plt.legend()
    if show:
        plt.show()

def plot_brightness_score(batch: ExperimentBatch, score_type: ScoreType = ScoreType.Ratio, range_modifier=4, cell_filter=lambda x: True, color='blue'):
    """
    Plot the batch-averaged brightness score over time for a single batch.

    Combines :func:`calc_brightness_score` and :func:`plot_score_avg`; this
    is the top-level plot used in the example notebook to visualize
    recruitment kinetics for one batch of experiments.

    Parameters
    ----------
    batch : DuckSeg.experiment_loader.ExperimentBatch
        The batch to plot.
    score_type : ScoreType, optional
        See :meth:`CellVideo.calculate_brightness_score`.
    range_modifier : float, optional
        ROI width factor forwarded to
        :meth:`CellVideo.calculate_brightness`.
    cell_filter : Callable[[CellVideo], bool], optional
        Predicate selecting which cells to include, e.g. to exclude
        detected outliers.
    color : str, optional
        Line color.
    """
    diffs = calc_brightness_score(batch, score_type = score_type, range_modifier=range_modifier, cell_filter=cell_filter)
    flattened = [diff for exp_diff in diffs for diff in exp_diff]
    plt.figure()
    plot_score_avg(flattened, color=color)

def plot_brightness_score_double(ctrl: ExperimentBatch, cnnd: ExperimentBatch, score_type: ScoreType = ScoreType.Ratio, range_modifier=4, cell_filter=lambda x: True):
    """
    Plot batch-averaged brightness scores of two batches overlaid for comparison.

    Parameters
    ----------
    ctrl : DuckSeg.experiment_loader.ExperimentBatch
        First batch (e.g. a control condition), labeled with ``ctrl.name``.
    cnnd : DuckSeg.experiment_loader.ExperimentBatch
        Second batch to compare against, labeled with ``cnnd.name``.
    score_type : ScoreType, optional
        See :meth:`CellVideo.calculate_brightness_score`.
    range_modifier : float, optional
        ROI width factor forwarded to
        :meth:`CellVideo.calculate_brightness`.
    cell_filter : Callable[[CellVideo], bool], optional
        Predicate selecting which cells to include, applied to both
        batches.
    """
    ctrl_diffs = calc_brightness_score(ctrl, score_type = score_type, range_modifier=range_modifier, cell_filter=cell_filter)
    cnnd_diffs = calc_brightness_score(cnnd, score_type = score_type, range_modifier=range_modifier, cell_filter=cell_filter)
    ctrl_flattened = [diff for exp_diff in ctrl_diffs for diff in exp_diff]
    cnnd_flattened = [diff for exp_diff in cnnd_diffs for diff in exp_diff]
    plot_score_double(ctrl_flattened, ctrl.name, cnnd_flattened, cnnd.name)

def plot_brightness_scores(batches: list[ExperimentBatch], score_type: ScoreType = ScoreType.Ratio, range_modifier=4, cell_filter=lambda x: True):
    """
    Plot batch-averaged brightness scores of any number of batches overlaid.

    Generalizes :func:`plot_brightness_score_double` to more than two
    batches (up to the 6 colors :func:`plot_scores_avg` cycles through).

    Parameters
    ----------
    batches : list of DuckSeg.experiment_loader.ExperimentBatch
        Batches to compare, each labeled with its ``name``.
    score_type : ScoreType, optional
        See :meth:`CellVideo.calculate_brightness_score`.
    range_modifier : float, optional
        ROI width factor forwarded to
        :meth:`CellVideo.calculate_brightness`.
    cell_filter : Callable[[CellVideo], bool], optional
        Predicate selecting which cells to include, applied to every
        batch.
    """
    scores_names = []
    for batch in batches:
        diffs = calc_brightness_score(batch, score_type = score_type, range_modifier=range_modifier, cell_filter=cell_filter)
        flattened = [diff for exp_diff in diffs for diff in exp_diff]
        scores_names.append((flattened, batch.name))
    plot_scores_avg(scores_names)


def plot_brightness_score_wt_ko(batch: ExperimentBatch, score_type: ScoreType = ScoreType.Ratio, range_modifier=4, cell_filter=lambda x: True, print_progress=False):
    """
    Plot WT vs. KO batch-averaged brightness scores from a single batch.

    Splits ``batch`` into its WT and KO experiments (see
    :meth:`~DuckSeg.experiment_loader.ExperimentBatch.wt_experiments` /
    :meth:`~DuckSeg.experiment_loader.ExperimentBatch.ko_experiments`) and
    overlays their averaged score time series — the standard genotype
    comparison plot for a laser-ROI recruitment assay.

    Parameters
    ----------
    batch : DuckSeg.experiment_loader.ExperimentBatch
        Batch containing both WT and KO experiments.
    score_type : ScoreType, optional
        See :meth:`CellVideo.calculate_brightness_score`.
    range_modifier : float, optional
        ROI width factor forwarded to
        :meth:`CellVideo.calculate_brightness`.
    cell_filter : Callable[[CellVideo], bool], optional
        Predicate selecting which cells to include, applied to both
        groups.
    print_progress : bool, optional
        If ``True``, print each experiment's name as it's processed.
    """
    wt_diffs = calc_brightness_score(batch, score_type = score_type, range_modifier=range_modifier, cell_filter=cell_filter, experiment_type = ExperimentType.WT, print_progress=print_progress)
    ko_diffs = calc_brightness_score(batch, score_type = score_type, range_modifier=range_modifier, cell_filter=cell_filter, experiment_type = ExperimentType.KO, print_progress=print_progress)
    wt_flattened = [diff for exp_diff in wt_diffs for diff in exp_diff]
    ko_flattened = [diff for exp_diff in ko_diffs for diff in exp_diff]
    plot_score_double(wt_flattened, "wt", ko_flattened, "ko")

def plot_score_double(score1, score1_name, score2, score2_name):
    """
    Plot two labeled score-series collections overlaid via :func:`plot_scores_avg`.

    Parameters
    ----------
    score1 : list of numpy.ndarray
        First collection of per-cell score time series.
    score1_name : str
        Legend label for ``score1``.
    score2 : list of numpy.ndarray
        Second collection of per-cell score time series.
    score2_name : str
        Legend label for ``score2``.
    """
    plot_scores_avg([(score1, score1_name), (score2, score2_name)])
    #plot_score_avg(score1, label = score1_name, show=False)
    #plot_score_avg(score2, label = score2_name, color = "red")

def plot_scores_avg(scores_names):
    """
    Plot several labeled score-series collections overlaid on one figure.

    Parameters
    ----------
    scores_names : list of tuple of (list of numpy.ndarray, str)
        ``(score_series_list, label)`` pairs, one per group to overlay.
        Up to 6 groups are supported (colors cycle through
        red/blue/green/purple/yellow/black).
    """
    colors = ['red','blue','green','purple','yellow','black']
    for i, (score, name) in enumerate(scores_names):
        plot_score_avg(score, label = name, show=i+1 == len(scores_names), color=colors[i])

def plot_brightness_score_series(batch: ExperimentBatch, score_type: ScoreType = ScoreType.Ratio, range_modifier=4, cell_filter=lambda x: True):
    """
    Plot how the distribution of per-cell brightness scores evolves over time.

    Unlike :func:`plot_brightness_score` (which shows only the mean and
    SEM), this renders an interactive 3D Plotly surface of the full score
    distribution at every timepoint via
    :func:`plotly_time_series_distribution_3d`, useful for spotting
    bimodal or skewed responses that a mean would hide.

    Parameters
    ----------
    batch : DuckSeg.experiment_loader.ExperimentBatch
        The batch to plot.
    score_type : ScoreType, optional
        See :meth:`CellVideo.calculate_brightness_score`.
    range_modifier : float, optional
        ROI width factor forwarded to
        :meth:`CellVideo.calculate_brightness`.
    cell_filter : Callable[[CellVideo], bool], optional
        Predicate selecting which cells to include.
    """
    # what do I want to plot?
    # the distribution of the score over time
    diffs = calc_brightness_score(batch, score_type = score_type, range_modifier=range_modifier, cell_filter=cell_filter)
    flattened = [diff for exp_diff in diffs for diff in exp_diff]
    plotly_time_series_distribution_3d(flattened)

import plotly.graph_objs as go

def plotly_time_series_distribution_3d(
    list_of_series,
    num_bins=20,
    title='Distribution of Scores Over Time',
    xaxis_title='Time Step',
    yaxis_title='Score Value',
    zaxis_title='Density',
    colorscale='Viridis',
    density=True
):
    """
    Creates an interactive 3D surface plot using Plotly showing how the distribution
    of values in time series evolves over time.

    Parameters:
    - list_of_series: list of lists or 2D NumPy array
    - num_bins: number of histogram bins
    - title: plot title
    - xaxis_title, yaxis_title, zaxis_title: axis labels
    - colorscale: Plotly color scale name
    - density: whether to normalize histogram (density=True) or show raw counts
    """

    data = np.array(list_of_series)
    data = data[~np.isnan(data).any(axis=1)]  # Remove NaN-containing series

    if data.size == 0:
        raise ValueError("No valid time series to plot after removing NaN values.")

    time_steps = data.shape[1]
    min_val, max_val = np.min(data), np.max(data)
    bins = np.linspace(min_val, max_val, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Build histogram matrix Z
    Z = []
    for t in range(time_steps):
        values_at_t = data[:, t]
        hist, _ = np.histogram(values_at_t, bins=bins, density=density)
        Z.append(hist)
    Z = np.array(Z).T  # shape: (bins, time_steps)

    # Create surface plot
    surface = go.Surface(
        z=Z,
        x=np.arange(time_steps),  # time steps
        y=bin_centers,            # bin centers (score values)
        colorscale=colorscale
    )

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title=xaxis_title),
            yaxis=dict(title=yaxis_title),
            zaxis=dict(title=zaxis_title),
        ),
        autosize=False,
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig = go.Figure(data=[surface], layout=layout)
    fig.show()

def plot_all_scores(diffs, outliers=None, max_per_fig = 40, sort_by=lambda y: -1000000 if np.any(np.isnan(y)) else np.mean(y)):
    """
    Plot every per-cell score series from a nested per-experiment score list.

    Parameters
    ----------
    diffs : list of list of numpy.ndarray
        Per-experiment lists of per-cell score series, e.g. from
        :func:`calc_brightness_score`.
    outliers : optional
        Unused; accepted for signature symmetry with :func:`plot_all_vid_scores`.
    max_per_fig : int, optional
        Maximum number of lines per figure, forwarded to
        :func:`plot_sorted_3d_lines`.
    sort_by : Callable[[numpy.ndarray], float], optional
        Sort key applied to each series, forwarded to
        :func:`plot_sorted_3d_lines`.
    """
    flattened = [diff for exp_diff in diffs for diff in exp_diff]
    return plot_sorted_3d_lines(flattened, max_per_fig=max_per_fig)

def plot_all_vid_scores(scores, ids, outliers=None, max_per_fig = 40, sort_by=lambda y: -1000000 if np.any(np.isnan(y)) else np.mean(y)):
    """
    Plot every per-cell score series, labeled by cell video ID.

    The example notebook uses this (with
    :func:`calculate_brightness_score_for_videos`) as a final diagnostic:
    a 3D plot of every individual cell's score trace, sorted and labeled
    by ID, to spot-check cells beyond the batch-averaged summary plots.

    Parameters
    ----------
    scores : list of numpy.ndarray
        Per-cell score series, e.g. from
        :func:`calculate_brightness_score_for_videos`.
    ids : list of str
        Cell video IDs, matching ``scores`` element-for-element.
    outliers : optional
        Unused; accepted for signature symmetry with :func:`plot_all_scores`.
    max_per_fig : int, optional
        Maximum number of lines per figure, forwarded to
        :func:`plot_sorted_3d_lines`.
    sort_by : Callable[[numpy.ndarray], float], optional
        Sort key applied to each series, forwarded to
        :func:`plot_sorted_3d_lines`.
    """
    scores = scores[:]
    return plot_sorted_3d_lines(scores, max_per_fig=max_per_fig, sort_by=sort_by, ids=ids)

def plot_sorted_3d_lines(lines, max_per_fig=None, sort_by=lambda y: -1000000 if np.any(np.isnan(y)) else np.mean(y), ids = None):
    """
    Plot 1-D lines (Y only) in 3-D, sorted by a key, batched across figures.

    Each input line is drawn as a 3D line trace with its rank (post-sort
    position) as the Z axis, giving a stacked/waterfall view of many score
    series at once. Used by :func:`plot_all_scores`/:func:`plot_all_vid_scores`
    to inspect every individual cell's score trace.

    Parameters
    ----------
    lines : list of array-like
        Each element is a 1D sequence of Y values (e.g. one cell's score
        time series).
    max_per_fig : int or None, optional
        Maximum number of lines per figure. If ``None`` or non-positive,
        all lines go in one figure.
    sort_by : Callable[[array-like], float], optional
        Sort key applied to each line; lines are drawn in ascending key
        order. Defaults to the line's mean, with NaN-containing lines
        sorted first.
    ids : list of str, optional
        Per-line labels shown on the Z axis, matching ``lines``
        element-for-element.

    Returns
    -------
    None
        Displays one or more interactive Plotly figures; does not return a
        value.
    """
    from math import ceil
    # 1. Sort by mean
    if ids is not None:
        assert len(ids) == len(lines)
        idd_lines = list(zip(lines, ids))
        sorted_lines, sorted_ids = list(zip(*sorted(idd_lines, key= lambda k: sort_by(k[0]))))
    else:
        sorted_lines = sorted(lines, key=sort_by)
        sorted_ids = None
    n_lines = len(sorted_lines)

    # 2. Figure batching
    if max_per_fig is None or max_per_fig <= 0:
        max_per_fig = n_lines
    n_figs = ceil(n_lines / max_per_fig)
    max_per_fig = ceil(n_lines / n_figs)

    for fig_idx in range(n_figs):
        start = fig_idx * max_per_fig
        end   = min(start + max_per_fig, n_lines)
        batch = sorted_lines[start:end]

        fig = go.Figure()
        for local_idx, y in enumerate(batch):
            x = np.arange(len(y))
            z = np.full_like(x, local_idx+start)           # stack within the batch
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                name=f'Line {start+local_idx}',
                line=dict(width=4)
            ))

        if sorted_ids is not None:
            zaxis = dict(
                tickvals=list(range(start, end)),
                ticktext=[sorted_ids[i] for i in range(start, end)]
            )
        else:
            zaxis=None
        fig.update_layout(
            title=f'3‑D Lines {start}–{end-1} (sorted by mean)',
            scene=dict(
                xaxis_title='Index',
                yaxis_title='Y',
                zaxis_title='Line rank within figure',
                zaxis=zaxis
            ),
            width=800, height=600,
            showlegend=False
        )
        fig.show()


#import numpy as np
import imageio.v3 as iio
import base64, io, uuid
from IPython.display import display, HTML
import ipywidgets as widgets
import tempfile
import os

def encode_mp4_temp(frames, fps=24, plugin="FFMPEG"):
    """
    Encode uint8 RGB frames → MP4 bytes using a real temp file
    (needed for plugins such as ffmpeg that can’t handle BytesIO).

    Returns
    -------
    bytes  : the MP4 file’s contents
    """
    # create a temporary file that survives until we manually delete
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_name = tmp.name          # path to pass into imageio
    try:
        # write video to that path
        iio.imwrite(tmp_name, frames, plugin=plugin, fps=fps)

        # read the finished file back into memory
        with open(tmp_name, "rb") as f:
            mp4_bytes = f.read()
    finally:
        os.remove(tmp_name)          # clean up the temp file

    return mp4_bytes

def video_widget(frames, fps=20, max_width='800px'):
    """
    Interactive HTML5 video player with:
      • Scroll‑pan & Ctrl+wheel zoom
      • Playback‑rate slider (0.1×–10×)
    
    Parameters
    ----------
    frames : iterable of uint8 RGB frames, shape (H,W,3)
    fps    : int, encoding frame‑rate
    max_width : CSS width for the video container
    """
    # --- Encode frames to MP4 in memory ---
    b64 = base64.b64encode(encode_mp4_temp(frames, fps=fps)).decode()
    
    # Unique IDs so multiple widgets coexist
    vid_id  = f"v{uuid.uuid4().hex}"
    slid_id = f"s{uuid.uuid4().hex}"
    
    # --- HTML video, wrapped in a scrollable div for panning ---
    video_html = f"""
    <div style="overflow:auto; border:1px solid #ccc; width:{max_width};">
      <video id="{vid_id}" src="data:video/mp4;base64,{b64}" 
             controls style="width:100%; display:block;"></video>
    </div>
    """
    
    # --- Playback‑rate slider (log scale 0.1–10) ---
    slider = widgets.FloatLogSlider(value=1, base=10, min=-1, max=1,
                                    description='Speed ×', readout_format='.2f')
    slider.layout.width = max_width
    slider.add_class(slid_id)
    
    # --- JS: link slider → playbackRate, add Ctrl‑wheel zoom ---
    js = f"""
    <script>
    (function() {{
        const video  = document.getElementById("{vid_id}");
        const slider = document.querySelector(".{slid_id} input");
        
        // Sync slider → video
        const updateRate = () => {{ video.playbackRate = parseFloat(slider.value); }};
        slider.addEventListener('input', updateRate);
        updateRate();  // initial
        
        // Ctrl+wheel zoom
        let scale = 1;
        video.addEventListener('wheel', e => {{
            if(!e.ctrlKey) return;
            e.preventDefault();
            scale *= (e.deltaY < 0 ? 1.1 : 0.9);
            scale = Math.max(0.2, Math.min(scale, 10));
            video.style.transformOrigin = '0 0';
            video.style.transform = `scale(${{scale}})`;
        }}, {{passive:false}});
    }})();
    </script>
    """
    
    display(widgets.VBox([widgets.HTML(video_html + js), slider]))
