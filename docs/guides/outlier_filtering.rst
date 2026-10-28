=======================
Review and filter cells
=======================

Not every tracked cell is trustworthy. Combine automated filters with a
visual review, then keep only the cells that pass both.

1. Automated filters
====================

A filter is a function that takes a
:class:`~DuckSeg.experiment_evaluator.CellVideo` and returns ``True`` if
the cell should be **kept**. The ready-made filters in
:mod:`DuckSeg.outlier_filters` are factories:

``area_between(start, end)``
    Keep cells whose mask area (in pixels) stays within ``[start, end)``
    in every frame.

``max_area_change_between_frames(ratio)``
    Keep cells whose area never changes by more than ``ratio`` (``0.2`` is
    20 %) from one frame to the next.

``max_total_area_change(ratio)``
    Keep cells whose smallest area is within ``ratio`` of the largest over
    the whole video.

``id_starts_with(prefix)``
    Keep only cells whose id starts with ``prefix``. Cell ids begin with
    the experiment (file) name, so this selects experiments.

.. code-block:: python

   from DuckSeg.outlier_filters import area_between, max_area_change_between_frames, outlier_vids

   filters = [area_between(10, 5000), max_area_change_between_frames(0.2)]
   extra_manual_outliers = {"MSH2_WT_001_7"}       # ids you always want to exclude
   outliers = outlier_vids(vids, extra_manual_outliers, filters)

``outliers`` is the set of ids that failed any filter, plus the manual ids.

2. Visual review
================

The review grid shows all cells side by side with synchronized playback.
The irradiated band is drawn into each cell for orientation, so use the
same ``range_modifier`` as later in the plots.

.. code-block:: python

   from DuckSeg.cell_video_visualizer import normalize_videos_for_display, show_video_grid_jupyter

   range_modifier = 5
   vids2 = normalize_videos_for_display(vids, range_modifier=range_modifier)

   manual_outliers = set()          # filled by the checkboxes
   update = show_video_grid_jupyter(vids2[:30], manual_outliers=manual_outliers)
   update(outliers, extra_manual_outliers)      # mark the automatic outliers

Tick the checkbox of a cell to exclude it. Cells already flagged get a red
border and can be hidden with *Show outliers*. For large batches show the
cells in chunks (the example notebook uses 30 per grid) and collect one
``update`` function per grid.

Cell ids have the form ``<experiment name>_<number>`` and are shown above
each cell.

3. Use the result
=================

.. code-block:: python

   cell_filter = lambda vid: vid.id not in outliers and vid.id not in manual_outliers

Pass ``cell_filter`` to the plotting functions (:doc:`compare_conditions`).
The CSV export does not filter by itself (:doc:`export_results`).

Optional: inspect the individual traces
=======================================

Interactive 3D plots stack the score traces of all cells, sorted by their
mean, which makes unusual traces easy to spot:

.. code-block:: python

   from DuckSeg.experiment_evaluator import (
       ScoreType, calculate_brightness_score_for_videos, plot_all_vid_scores)

   ids, scores = calculate_brightness_score_for_videos(
       vids, score_type=ScoreType.Diff, range_modifier=range_modifier, cell_filter=cell_filter)
   plot_all_vid_scores(scores, ids)

Write your own filter
=====================

.. code-block:: python

   import numpy as np
   from DuckSeg.outlier_filters import area

   def min_frames(n):
       def keep(vid):
           return vid.frame_count() >= n
       return keep

   def mean_area_above(pixels):
       def keep(vid):
           return np.mean(area(vid)) > pixels
       return keep
