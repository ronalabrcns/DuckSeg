====================
Prepare your data
====================

DuckSeg reads one **batch directory** per condition. A batch holds the
videos and, for every video, a binary mask of the irradiated region.

File names
==========

For each experiment put two files in the batch directory::

    MSH2_001.nd2             # the video (.nd2, .tif or .tiff)
    MSH2_001_ROI.tif         # the laser ROI mask

* The ROI file must be called ``<video name without extension>_ROI.tif``.
  Files ending in ``_ROI.tif`` are never treated as videos; a mask saved as
  ``_ROI.tiff`` would be.
* The mask has the same height and width as the video. Non-zero pixels
  mark the irradiated rows; the rows are read along the middle column of
  the image, so the stripe has to cross it.

Load a batch
============

.. code-block:: python

   from DuckSeg.experiment_loader import ExperimentBatch

   batch = ExperimentBatch("MSH2", "/path/to/batch_directory")
   print(len(batch.experiments()))

``batch.experiments()`` lists the experiments sorted by name. The videos
are not read until they are needed.

Use the same number of frames everywhere
========================================

To compare experiments, cut them all to the length of the shortest one:

.. code-block:: python

   batch.max_len = min(e.frame_count() for e in batch.experiments())

``frame_count()`` reads only the file header, not the frames.
``max_len`` is applied to every experiment that ``batch.experiments()``
returns afterwards. Lowering it later reuses the cached segmentation
(cut to the new length); raising it segments the missing frames again.
The plots stop with a clear message if the videos of a plot have
different lengths.

Bit depth
=========

The segmentation model gets 8-bit images. ``.nd2`` frames are scaled from
16 bit (divided by 65535). TIFF frames that are 8 bit stay as they are,
16-bit frames are scaled like ``.nd2``, and other types are scaled by
their maximum. The measurement itself always uses the original values.
