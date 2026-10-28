==================
Compare conditions
==================

Every plot shows the mean score of the retained cells at each frame with
the standard error of the mean as a shaded band (see :doc:`../concepts`).
Choose the score with :class:`~DuckSeg.experiment_evaluator.ScoreType`:
``Ratio`` (in-band over reference) or ``Diff`` (in-band minus reference).

One directory per condition
===========================

Create one :class:`~DuckSeg.experiment_loader.ExperimentBatch` per
condition, with the condition as its name. The name is used as the label
in the legend.

.. code-block:: python

   from DuckSeg.experiment_loader import ExperimentBatch
   from DuckSeg.experiment_evaluator import ScoreType, plot_brightness_scores

   batches = [
       ExperimentBatch("EV_siNT",  "/data/MSH2/EV_siNT"),
       ExperimentBatch("EV_siUTR", "/data/MSH2/EV_siUTR"),
   ]
   for b in batches:
       b.max_len = 60            # same length everywhere

   plot_brightness_scores(batches, score_type=ScoreType.Ratio,
                          range_modifier=5, cell_filter=cell_filter)

Run :func:`~DuckSeg.experiment_evaluator.evaluate_batch` for every batch
first (:doc:`caching`); the plots read the stored results. Colors are
assigned in the order red, blue, green, purple, yellow, black, so at most
six conditions can be plotted at once.

Other plots
===========

Single batch
    :func:`~DuckSeg.experiment_evaluator.plot_brightness_score` pools all
    experiments of one batch.

Two batches
    :func:`~DuckSeg.experiment_evaluator.plot_brightness_score_double`
    takes a control and a second batch.

WT versus KO inside one batch
    :func:`~DuckSeg.experiment_evaluator.plot_brightness_score_wt_ko`
    splits one batch by ``_WT_`` and ``_KO_`` in the file names.

Score distribution over time
    :func:`~DuckSeg.experiment_evaluator.plot_brightness_score_series`
    draws an interactive 3D surface of how the scores are distributed at
    each frame.

``cell_filter`` is optional in all of them; without it every cell is used.

Baseline correction and saving a figure
=======================================

All the mean-curve plots accept two options:

``baseline_correct=True``
    Subtract the first frame's value from the series of every cell, so
    that all curves start at zero. The default (``False``) plots the raw
    scores. ``plot_brightness_score_series`` accepts it too.

``save_path="recruitment.png"``
    Also save the figure to this file (300 dpi; the format follows the
    extension, for example ``.png``, ``.pdf`` or ``.svg``). The figure is
    still displayed. Not available for the interactive 3D plot.

.. code-block:: python

   plot_brightness_scores(batches, score_type=ScoreType.Ratio, range_modifier=5,
                          cell_filter=cell_filter, baseline_correct=True,
                          save_path="recruitment_ratio.png")
