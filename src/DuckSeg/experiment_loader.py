"""
Loading raw microscopy videos and their laser-ROI masks from disk.

This module is the entry point of the DuckSeg pipeline. An
:class:`ExperimentBatch` represents a directory of paired video files
(``.nd2`` or ``.tif``/``.tiff``) and ROI mask images belonging to one
biological batch; it discovers the individual :class:`Experiment` objects
on disk. Each :class:`Experiment` lazily loads its own video frames and ROI
mask, and derives the pixel row ranges struck by the laser
(:func:`generate_ROI_ranges`) from the mask.

Expected file naming convention, per experiment, inside a batch directory::

    <NAME>_<WT|KO>_<...>.nd2        # or .tif / .tiff
    <NAME>_<WT|KO>_<...>_ROI.tif    # binary mask, same frame size

where the group label (``WT``/``KO``) anywhere in the file name is used by
:meth:`ExperimentBatch.wt_experiments` / :meth:`ExperimentBatch.ko_experiments`
to split a batch into conditions.
"""

import nd2
import os
import os.path
import dataclasses
import numpy as np
from typing import Optional
from numpy import ndarray
import matplotlib.animation as animation
from IPython.display import HTML, display
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import enum

import tifffile #add to dependencies also

def nd2_u8_frames(file_name):
    """
    Read every frame of an ``.nd2`` file as 8-bit grayscale.

    Parameters
    ----------
    file_name : str
        Path to the ``.nd2`` video file.

    Returns
    -------
    list of numpy.ndarray
        One ``uint8`` frame per sequence position in the file.
    """
    with nd2.ND2File(file_name) as f:
        num_frames = f.attributes.sequenceCount
        return [get_frame_u8(f, i) for i in range(num_frames)]

def nd2_frames(file_name):
    """
    Read every frame of an ``.nd2`` file at its native bit depth.

    Parameters
    ----------
    file_name : str
        Path to the ``.nd2`` video file.

    Returns
    -------
    list of numpy.ndarray
        One frame per sequence position in the file, in its original dtype
        (typically ``uint16``).
    """
    with nd2.ND2File(file_name) as f:
        num_frames = f.attributes.sequenceCount
        return [np.array(f.read_frame(i)) for i in range(num_frames)]

def get_frame_u8(f: nd2.ND2File, i: int) -> np.ndarray:
    """
    Read a single frame from an open ``.nd2`` file as 8-bit grayscale.

    Parameters
    ----------
    f : nd2.ND2File
        An already-open ND2 file handle.
    i : int
        Frame index to read.

    Returns
    -------
    numpy.ndarray
        The frame, rescaled from 16-bit to ``uint8``.
    """
    return (f.read_frame(i)/(2.**16-1)*255).astype(np.uint8)

def tiff_frames(file_name):
    """
    Read every frame of a multi-page ``.tif``/``.tiff`` file.

    Parameters
    ----------
    file_name : str
        Path to the TIFF video file.

    Returns
    -------
    list of numpy.ndarray
        One frame per page, in its original dtype.
    """
    # tifffile returns a numpy array of shape (frames, height, width)
    video = tifffile.imread(file_name)
    return [frame for frame in video]

def tiff_u8_frames(file_name):
    """
    Read every frame of a multi-page ``.tif``/``.tiff`` file as 8-bit.

    Parameters
    ----------
    file_name : str
        Path to the TIFF video file.

    Returns
    -------
    list of numpy.ndarray
        One ``uint8`` frame per page. ``uint16`` input is rescaled from the
        full 16-bit range; any other dtype is rescaled by its own maximum
        value.
    """
    video = tifffile.imread(file_name)
    # Convert to 8-bit safely depending on the original data type
    if video.dtype == np.uint16:
        return [(frame / (2.**16 - 1) * 255).astype(np.uint8) for frame in video]
    elif video.dtype == np.uint8:
        return [frame for frame in video]
    else:
        # Fallback for weird float/32-bit formats
        return [(frame / frame.max() * 255).astype(np.uint8) for frame in video]

def generate_ROI_ranges(roi) -> list[tuple[float]]:
    """
    Extract the laser-irradiated row ranges from a binary ROI mask.

    Scans the mask's center column top to bottom and returns each
    contiguous run of non-zero rows as a ``(start, end)`` range. DuckSeg
    represents the laser-struck region(s) of an image this way rather than
    as a full 2D mask, since the ROI is drawn as one or more horizontal
    strips.

    Parameters
    ----------
    roi : numpy.ndarray
        2D binary (or near-binary) mask image, non-zero where the laser
        irradiated the sample.

    Returns
    -------
    list of tuple of (float, float)
        ``(start_row, end_row)`` for each contiguous irradiated strip,
        ``start_row`` inclusive and ``end_row`` exclusive.
    """
    in_roi = False
    ranges = []
    last_start = None
    j = roi.shape[1]//2
    for i in range(roi.shape[0]):
        if in_roi:
             if roi[i,j] == 0:
                 in_roi = False
                 ranges.append((last_start, i))
        else:
            if roi[i,j] != 0:
                in_roi = True
                last_start = i
    if in_roi:
        ranges.append((last_start, roi.shape[1]))
    return ranges

@dataclasses.dataclass
class Experiment:
    """
    A single laser-irradiation video and its associated ROI mask.

    Instances are normally created by :meth:`ExperimentBatch.experiments`,
    not constructed directly. Video frames and the ROI mask are loaded
    lazily on first access and then cached on the instance (``loaded_video``,
    ``loaded_video_u8``, ``loaded_ROI``), so repeated calls are cheap.

    Attributes
    ----------
    name : str
        Experiment identifier, derived from the video file name without
        its extension (e.g. ``"MSH2_WT_00123"``).
    path : str
        Path to the video file (``.nd2`` or ``.tif``/``.tiff``).
    ROI_path : str
        Path to the corresponding ROI mask image.
    max_len : int or None
        If set, truncates loaded videos to this many frames. Useful for
        aligning experiments in a batch that were recorded for different
        durations.
    """
    name: str
    path: str
    ROI_path: str
    max_len: Optional[int]
    loaded_video: Optional[list[ndarray]] = None
    loaded_video_u8: Optional[list[ndarray]] = None
    loaded_ROI: Optional[ndarray] = None
    _ROI_ranges: Optional[list[tuple[int]]] = None
    def first_frame_u8(self):
        """Return the first frame of the video as 8-bit grayscale."""
        return self.frame_u8(0)

    def frames(self):
        """
        Return the video frames at their native bit depth, loading and
        caching them on first call.

        Returns
        -------
        list of numpy.ndarray
            Frames in temporal order, truncated to ``max_len`` if set.

        Raises
        ------
        ValueError
            If ``path`` does not end in ``.nd2``, ``.tif``, or ``.tiff``.
        """
        if self.loaded_video is None:
            if self.path.lower().endswith(".nd2"):
                self.loaded_video = nd2_frames(self.path)
            elif self.path.lower().endswith((".tif", ".tiff")):
                self.loaded_video = tiff_frames(self.path)
            else:
                raise ValueError(f"Unsupported video file format: {self.path}")
                
        if self.max_len is not None:
            self.loaded_video = self.loaded_video[:self.max_len]
        return self.loaded_video

    def frames_u8(self):
        """
        Return the video frames as 8-bit grayscale, loading and caching
        them on first call.

        Returns
        -------
        list of numpy.ndarray
            ``uint8`` frames in temporal order, truncated to ``max_len`` if
            set.

        Raises
        ------
        ValueError
            If ``path`` does not end in ``.nd2``, ``.tif``, or ``.tiff``.
        """
        if self.loaded_video_u8 is None:
            if self.path.lower().endswith(".nd2"):
                self.loaded_video_u8 = nd2_u8_frames(self.path)
            elif self.path.lower().endswith((".tif", ".tiff")):
                self.loaded_video_u8 = tiff_u8_frames(self.path)
            else:
                raise ValueError(f"Unsupported video file format: {self.path}")
                
        if self.max_len is not None:
            self.loaded_video_u8 = self.loaded_video_u8[:self.max_len]
        return self.loaded_video_u8
        
    def frame_u8(self, i: int):
        """Return frame ``i`` of the video as 8-bit grayscale."""
        return self.frames_u8()[i]
    def plot_first_frame(self):
        """Display the first frame with matplotlib."""
        plt.imshow(self.first_frame_u8())
        plt.show()
    def plot_frame(self, i: int):
        """Display frame ``i`` with matplotlib."""
        plt.imshow(self.frame_u8(i))
        plt.show()
    def play_video(self):
        """Display the full video as an inline matplotlib animation."""
        frames = self.frames_u8()
        fig, ax = plt.subplots()
        im = ax.imshow(frames[0])
        plt.close()
        def update(frame):
            im.set_data(frame)
            return [im]
        ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=50)
        
        # Display the animation in the notebook
        a = HTML(ani.to_jshtml())
        display(a)
    def corrected_roi_path(self) -> str:
        """
        Path where a manually corrected ROI mask is looked up / saved.

        See :func:`DuckSeg.roi_editor.show_roi_editor`, which writes to
        this path when a user adjusts ROI boundaries interactively.

        Returns
        -------
        str
            ``ROI_path`` with ``_corrected_ROI`` inserted before the
            extension.
        """
        base, ext = os.path.splitext(self.ROI_path)
        return base + '_corrected_ROI' + ext

    def ROI_frame(self):
        """
        Return the ROI mask image, loading and caching it on first call.

        Prefers the corrected mask at :meth:`corrected_roi_path` if one
        exists (e.g. saved by the ROI editor), otherwise falls back to the
        original ``ROI_path``.

        Returns
        -------
        numpy.ndarray
            The ROI mask image.
        """
        if self.loaded_ROI is None:
            corrected = self.corrected_roi_path()
            path = corrected if os.path.exists(corrected) else self.ROI_path
            self.loaded_ROI = np.array(Image.open(path))
        return self.loaded_ROI

    def plot_ROI(self):
        """Display the ROI mask with matplotlib."""
        plt.imshow(self.ROI_frame())
        plt.show()

    def plot_frame_with_roi_masked(self, frame_idx: int = None):
        """
        Display a frame with the laser-irradiated rows blacked out.

        Useful for visually verifying the detected ROI ranges (see
        :meth:`ROI_ranges`) line up with the visible laser damage.

        Parameters
        ----------
        frame_idx : int, optional
            Index of the frame to display. Defaults to the middle frame of
            the video.
        """
        frames = self.frames_u8()
        idx = len(frames) // 2 if frame_idx is None else frame_idx
        frame = frames[idx].copy()
        for start, end in self.ROI_ranges():
            frame[int(start):int(end), :] = 0
        fig, ax = plt.subplots()
        ax.imshow(frame, cmap='viridis')
        ax.set_title(self.name)
        plt.show()


    def ROI_ranges(self) -> list[tuple[float]]:
        """
        Return the laser-irradiated row ranges, computing and caching them
        on first call.

        Returns
        -------
        list of tuple of (float, float)
            ``(start_row, end_row)`` per irradiated strip; see
            :func:`generate_ROI_ranges`.
        """
        if self._ROI_ranges is None:
            self._ROI_ranges = generate_ROI_ranges(self.ROI_frame())
        return self._ROI_ranges

    def cache_dir(self) -> str:
        """Return the directory used to cache derived data for this experiment."""
        return f'{self.path}_cache'
    def cache_path(self, name: str) -> str:
        """Return the cache file path for a named piece of derived data."""
        return os.path.join(self.cache_dir(), name)
    def ensure_cache_dir_exists(self):
        """Create :meth:`cache_dir` if it does not already exist."""
        if not os.path.exists(self.cache_dir()):
            os.makedirs(self.cache_dir())
    def save_data(self, name: str, data, force: bool = False):
        """
        Pickle ``data`` to this experiment's cache directory.

        Used by :meth:`DuckSeg.experiment_evaluator.ExperimentEvaluator.save`
        to persist segmentation results so they don't need to be
        recomputed on every notebook run.

        Parameters
        ----------
        name : str
            Cache entry name (e.g. ``"evaluator"``).
        data : object
            Picklable object to store.
        force : bool, optional
            If ``False`` (default), raises rather than overwriting an
            existing cache entry.

        Raises
        ------
        AssertionError
            If the cache entry already exists and ``force`` is ``False``.
        """
        cache_path = self.cache_path(name)
        self.ensure_cache_dir_exists()
        assert force or not os.path.exists(cache_path), f"{cache_path} already exists. Use force=True to overwrite"
        with open(cache_path, 'wb') as file:
            pickle.dump(data, file)
    def load_data(self, name: str, force: bool = False):
        """
        Load a previously pickled cache entry for this experiment.

        Parameters
        ----------
        name : str
            Cache entry name, as passed to :meth:`save_data`.
        force : bool, optional
            Unused; kept for symmetry with :meth:`save_data`.

        Returns
        -------
        object or None
            The unpickled data, or ``None`` if no such cache entry exists.
        """
        try:
            with open(self.cache_path(name), 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None

    def in_batch_id(self) -> str:
        """
        Return a short display label distinguishing this experiment within
        its batch, e.g. ``"WT 3"`` or ``"KO 1"``.

        Derived from the ``WT``/``KO`` group label and the last character
        of ``name`` (typically a per-condition sequence number).

        Returns
        -------
        str or None
            The display label, or ``None`` if ``name`` contains neither
            ``"WT"`` nor ``"KO"``.
        """
        if 'WT' in self.name:
            return f'WT {self.name[-1]}'
        if 'KO' in self.name:
            return f'KO {self.name[-1]}'

class ExperimentType(enum.Enum):
    """Experimental group label: wild-type or knock-out."""
    WT = 0
    KO = 1

@dataclasses.dataclass
class ExperimentBatch:
    """
    A directory of paired video/ROI files belonging to one biological batch.

    Discovers its :class:`Experiment` members from ``path`` on demand (see
    :meth:`experiments`) based on the file naming convention described in
    the module docstring; it does not scan the directory eagerly at
    construction time.

    Attributes
    ----------
    name : str
        Batch identifier, used for labeling in plots.
    path : str
        Directory containing the video and ROI files.
    max_len : int or None, optional
        If set, propagated to every :class:`Experiment` created by this
        batch to truncate loaded videos to this many frames.
    """
    name: str
    path: str
    max_len: Optional[int] = None
    def experiment_files(self):
        """Return the names of all files in the batch directory."""
        return os.listdir(self.path)
    def experiments(self) -> list[Experiment]:
        """
        Discover the experiments in this batch.

        Pairs each video file (``.nd2``/``.tif``/``.tiff``, excluding
        ``*_ROI.tif`` files) with its ``<basename>_ROI.tif`` mask.

        Returns
        -------
        list of Experiment
            One :class:`Experiment` per video file found, sorted by name.

        Raises
        ------
        AssertionError
            If a discovered video file has no corresponding file on disk
            (this should not normally happen, since the video itself is
            what was discovered).
        """
        all_files = self.experiment_files()
        # Grab videos: Must be nd2, tif, or tiff AND must NOT be the ROI files
        videos = [
            f for f in all_files 
            if f.lower().endswith((".nd2", ".tif", ".tiff")) 
            and not f.endswith("_ROI.tif")
        ]
        experiments = []
        for v in videos:
            # Safely split the base name from the extension to build the ROI filename
            base_name = os.path.splitext(v)[0]
            roi_name = base_name + "_ROI.tif"
            
            experiment = Experiment(
                name=base_name,
                path=os.path.join(self.path, v),
                ROI_path=os.path.join(self.path, roi_name),
                max_len=self.max_len
            )
            assert os.path.exists(experiment.path), f"Video file {experiment.path} not found."
            experiments.append(experiment)
            
        return sorted(experiments, key=lambda exp: exp.name)
        
    def ko_experiment(self, i: int):
        """Return the KO experiment whose path ends in ``"00{i}"``."""
        end = f"00{i}"
        return [e for e in self.experiments() if "_KO_" in e.path and e.path.endswith(end)][0]
    def wt_experiment(self, i: int):
        """Return the WT experiment whose path ends in ``"00{i}"``."""
        end = f"00{i}"
        return [e for e in self.experiments() if "_WT_" in e.path and e.path.endswith(end)][0]
    def wt_experiments(self):
        """Return every experiment whose path contains ``"_WT_"``."""
        return [e for e in self.experiments() if "_WT_" in e.path]
    def ko_experiments(self):
        """Return every experiment whose path contains ``"_KO_"``."""
        return [e for e in self.experiments() if "_KO_" in e.path]


    def experiment(self, i: int, type: ExperimentType) -> Experiment:
        """
        Return experiment ``i`` of the given group.

        Parameters
        ----------
        i : int
            Sequence number, forwarded to :meth:`wt_experiment` /
            :meth:`ko_experiment`.
        type : ExperimentType
            Which group to look in.

        Returns
        -------
        Experiment
        """
        if type == ExperimentType.WT:
            return self.wt_experiment(i)
        elif type == ExperimentType.KO:
            return self.ko_experiment(i)

def experiment_batches(data_dir):
    """
    Discover experiment batches from a directory of batch subdirectories.

    Each immediate subdirectory of ``data_dir`` is treated as one
    :class:`ExperimentBatch`, named after its directory name with the
    first 11 characters stripped (matching a common
    ``"YYYYMMDD_NN_"``-style timestamp prefix in this lab's directory
    naming convention).

    Parameters
    ----------
    data_dir : str
        Directory containing one subdirectory per batch.

    Returns
    -------
    dict of str to ExperimentBatch
        Batches keyed by their derived ``name``.
    """
    experiments = os.listdir(data_dir)
    out = {}
    for d in experiments:
        full_dir = os.path.join(data_dir, d)
        name = d[11:]
        out[name] = ExperimentBatch(name, full_dir)
    return out
