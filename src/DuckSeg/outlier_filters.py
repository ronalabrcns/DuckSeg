"""
Filters for excluding mis-segmented cells from downstream analysis.

CellSAM occasionally produces masks that merge adjacent cells, lose track
of a cell partway through a video, or otherwise vary in area in ways
inconsistent with a single tracked cell. Rather than fixing the
segmentation, DuckSeg flags these
:class:`~DuckSeg.experiment_evaluator.CellVideo` instances as outliers and
excludes them from brightness scoring.

A filter is any ``Callable[[CellVideo], bool]`` that returns ``True`` when
a cell video should be *kept*. :func:`area_between`,
:func:`max_area_change_between_frames`, :func:`max_total_area_change`, and
:func:`id_starts_with` are factories that build such filters; combine
several of them in a list and pass it to :func:`outlier_vids` to compute
the union of everything they reject, alongside any manually chosen
outlier IDs. New filters can be added following the same
``Callable[[CellVideo], bool]``-factory pattern.
"""

import numpy as np

def outlier_vids(vids, manual, filters):
    """
    Compute the set of cell-video IDs to exclude from analysis.

    Parameters
    ----------
    vids : list of DuckSeg.experiment_evaluator.CellVideo
        Cell videos to check against ``filters``.
    manual : Iterable of str
        IDs to always include in the result, regardless of what the
        filters decide (hand-picked outliers).
    filters : list of Callable[[DuckSeg.experiment_evaluator.CellVideo], bool]
        Filter functions; a video failing *any* filter (returning
        ``False``) is added to the outlier set. See :func:`area_between`,
        :func:`max_area_change_between_frames`,
        :func:`max_total_area_change`, and :func:`id_starts_with`.

    Returns
    -------
    set of str
        The union of ``manual`` and every video ID rejected by ``filters``.
    """
    outliers = set(manual)
    for vid in vids:
        for filter in filters:
            if not filter(vid):
                outliers.add(vid.id)
    return outliers

area_cache = {}
def area(vid: 'CellVideo') -> int:
    """
    Return the per-frame mask area (pixel count) of a cell video, cached.

    Results are memoized in the module-level ``area_cache`` dict, keyed by
    ``vid.id``, since the same video's area is typically queried by
    multiple filters.

    Parameters
    ----------
    vid : DuckSeg.experiment_evaluator.CellVideo
        The cell video to measure.

    Returns
    -------
    list of int
        Number of foreground (``True``) pixels in ``vid.masks[i]`` for each
        frame ``i``.
    """
    if vid.id not in area_cache:
        area_cache[vid.id] = [np.sum(mask) for mask in vid.masks]
    return area_cache[vid.id]

def area_between(start, end):
    """
    Build a filter that rejects cells whose mask area ever leaves a range.

    Parameters
    ----------
    start : int
        Minimum acceptable mask area, in pixels (inclusive).
    end : int
        Maximum acceptable mask area, in pixels (exclusive).

    Returns
    -------
    Callable[[DuckSeg.experiment_evaluator.CellVideo], bool]
        ``True`` if every frame's mask area falls within ``[start, end)``.
    """
    def filter(vid):
        return start <= min(area(vid)) and max(area(vid)) < end
    return filter

def max_area_change_between_frames(max_ratio):
    """
    Build a filter that rejects cells with abrupt frame-to-frame area jumps.

    Useful for catching segmentation failures such as a mask suddenly
    merging with a neighboring cell or losing most of the cell body.

    Parameters
    ----------
    max_ratio : float
        Maximum allowed relative change in mask area between two
        consecutive frames (e.g. ``0.2`` allows up to a 20% change).

    Returns
    -------
    Callable[[DuckSeg.experiment_evaluator.CellVideo], bool]
        ``True`` if no consecutive-frame area ratio deviates from 1 by more
        than ``max_ratio``.
    """
    def filter(vid):
        ratios = [area(vid)[i+1]/area(vid)[i] for i in range(len(vid.frames)-1)]
        return all(abs(1-ratio) < max_ratio for ratio in ratios)
    return filter

def max_total_area_change(ratio):
    """
    Build a filter that rejects cells whose area drifts too much overall.

    Unlike :func:`max_area_change_between_frames`, this compares the
    smallest and largest mask area seen anywhere in the video, catching
    slow drift that no single frame-to-frame step would flag.

    Parameters
    ----------
    ratio : float
        Maximum allowed relative difference between the minimum and
        maximum mask area over the whole video.

    Returns
    -------
    Callable[[DuckSeg.experiment_evaluator.CellVideo], bool]
        ``True`` if ``1 - min(area)/max(area) < ratio``.
    """
    def filter(vid):
        return 1 - min(area(vid))/max(area(vid)) < ratio
    return filter

def id_starts_with(s):
    """
    Build a filter that keeps only cell videos whose ID has a given prefix.

    Useful for restricting analysis to a specific experiment or condition
    by ID (cell video IDs are typically ``f"{experiment.name}_{index}"``).

    Parameters
    ----------
    s : str
        Required prefix.

    Returns
    -------
    Callable[[DuckSeg.experiment_evaluator.CellVideo], bool]
        ``True`` if ``vid.id.startswith(s)``.
    """
    def filter(vid):
        return vid.id.startswith(s)
    return filter
