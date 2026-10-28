========================
Segment, track and cache
========================

Segmenting every frame is the slowest step, so it is done once and stored.

.. code-block:: python

   from DuckSeg.experiment_evaluator import evaluate_batch

   vids = evaluate_batch(batch, transforms=transforms)   # use_cache=True

The function returns the list of tracked
:class:`~DuckSeg.experiment_evaluator.CellVideo` objects of the whole
batch. Older notebooks call it ``evalute_batch`` (sic); that name still
works as an alias.

Where results are stored
========================

Next to every video ``VIDEO.nd2`` a folder ``VIDEO.nd2_cache`` is created
holding a file ``evaluator``. It contains the masks, boxes and cell
tracks, but not the raw frames, which are re-read from the video file.
Cache files are Python pickles: only load caches you created yourself.

When the cache is used
======================

``use_cache=True`` (default)
    A saved result is loaded instead of running CellSAM again. Experiments
    without a cache are segmented.

ROI lines changed
    If you moved the stripe with the :doc:`ROI editor <roi_editor>` after
    the cache was written, the cells are re-tracked **once** on the new
    stripe and the cache is updated. CellSAM is not run again, and a
    message tells you that cell ids may now refer to other cells.

Different number of frames
    If ``batch.max_len`` was lowered, the cached result is cut to the new
    length and the cells are re-tracked (CellSAM is not run again). Only
    when more frames are needed than the cache holds is that experiment
    segmented again.

Changed transforms
    The cache does **not** notice a change of ``transforms``. After
    editing them, run the batch once with ``use_cache=False``:

    .. code-block:: python

       vids = evaluate_batch(batch, transforms=transforms, use_cache=False)

Load results later
==================

In a new session, load the stored cells without segmenting:

.. code-block:: python

   from DuckSeg.experiment_evaluator import ExperimentEvaluator

   vids = []
   for experiment in batch.experiments():
       evaluator = ExperimentEvaluator.load_experiment(experiment)
       if evaluator is not None:       # None means: no cache yet
           vids += evaluator.cell_videos()

The plotting functions (:func:`~DuckSeg.experiment_evaluator.plot_brightness_score`
and friends) load the cache themselves.
