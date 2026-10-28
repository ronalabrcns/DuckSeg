========
Concepts
========

This page explains what DuckSeg measures and how. See
:doc:`workflow` for the order in which the steps are run.

.. contents::
   :local:
   :depth: 1

The measurement
===============

In a laser micro-irradiation experiment a thin stripe across each nucleus
is damaged, and a fluorescently tagged protein may accumulate there. The
ROI mask of an experiment marks the irradiated rows of the image. For each
tracked cell and each frame DuckSeg computes

* :math:`I_\mathrm{ROI}`, the mean intensity of the cell's mask pixels
  inside the irradiated band, and
* :math:`I_\mathrm{ref}`, the mean intensity of the remaining mask pixels
  of the same cell (above and below the band),

and from them two scores (:class:`~DuckSeg.experiment_evaluator.ScoreType`):

.. math::

   \mathrm{Diff} = I_\mathrm{ROI} - I_\mathrm{ref}, \qquad
   \mathrm{Ratio} = I_\mathrm{ROI} / I_\mathrm{ref}

Both use the original (non-preprocessed) fluorescence values; the
preprocessing transforms are only used to help the segmentation model.
Using the rest of the nucleus as the reference means no separate
background measurement is needed, which makes the scores suitable for a
quick first look rather than a final kinetic analysis.

The implementation is
:meth:`~DuckSeg.experiment_evaluator.CellVideo.calculate_brightness`.

.. note::

   By default the scores are the raw values. Pass ``baseline_correct=True``
   to the plot, score and export functions to subtract the first time
   point of every cell so that the curves start at zero (see
   :doc:`guides/compare_conditions` and :doc:`guides/export_results`).
   Photobleaching is not corrected.

The irradiated band and ``range_modifier``
------------------------------------------

The band is derived from the ROI mask: the rows where the mask, read
along the middle column of the image, is non-zero. Several stripes in one
image are supported; a cell uses the stripes that intersect its bounding
box in the first frame, and the score uses the first of them.

``range_modifier`` multiplies the *width* of the band around its centre.
``1`` is the stripe as drawn in the mask, ``4`` (the default of the plot
functions) uses a band four times as wide, which captures protein
spreading beyond the nominal stripe. Rows outside the widened band form
the reference. Use the same value in the review grid, the plots and the
CSV export.

Mean and error band
-------------------

The recruitment curve of a condition is the mean across all retained
cells at each frame; the shaded band is the standard error of the mean
(sample standard deviation divided by the square root of the number of
cells, :func:`~DuckSeg.experiment_evaluator.score_avg`). Cells whose score
series contains NaN (for example because the mask has no pixels on one
side of the band) are dropped before averaging.

Segmentation and tracking
=========================

Segmentation
------------

Every frame of an experiment is segmented independently with CellSAM
after the preprocessing transforms were applied. The result for a frame is
a labelled mask and one bounding box per detected nucleus.

Cells that are followed
-----------------------

Only nuclei whose bounding box intersects an irradiated band **in the
first frame** are followed
(:meth:`~DuckSeg.experiment_evaluator.ExperimentEvaluator.filter_starting_boxes`).

Linking cells across frames
---------------------------

:meth:`~DuckSeg.experiment_evaluator.ExperimentEvaluator.create_cell_video_inds3`
links a cell from one frame to the next by comparing masks. For each
candidate in the next frame it counts the pixels in which the two masks
differ (:func:`~DuckSeg.experiment_evaluator.mask_diff`, after aligning
the bounding boxes) and picks the candidate with the smallest difference.
If even the best candidate differs from the current mask by more than
60 % of the current mask's area, the cell is considered lost in that frame
and the previous assignment is kept.

Quality control
===============

Segmentation and linking can fail (merged nuclei, a cell jumping to a
neighbour). DuckSeg therefore offers

* automated filters on the mask area
  (:mod:`DuckSeg.outlier_filters`), and
* a visual review grid with a checkbox per cell
  (:func:`~DuckSeg.cell_video_visualizer.show_video_grid_jupyter`).

Both end up as a set of cell ids that are excluded from the plots; see
:doc:`guides/outlier_filtering`.

Caching
=======

Segmentation and tracking is by far the slowest step, so its result is
stored next to each video and reused (:doc:`guides/caching`). The raw
frames are not stored; they are re-read from the video file.

Limitations
===========

* The irradiated stripe is assumed to be **horizontal** (a set of image
  rows).
* Photobleaching and background variation are not corrected. Baseline
  correction (subtracting the first frame) is optional.
* Only the first stripe that intersects a cell is used for its score.
* Tracking relies on mask similarity between consecutive frames, so
  fast-moving, overlapping or dividing nuclei may be mislinked. Review the
  cells before drawing conclusions.
* DuckSeg is meant for fast, exploratory assessment of experiments rather
  than detailed kinetic modelling.
