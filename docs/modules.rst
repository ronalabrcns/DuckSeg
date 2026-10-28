
API Reference
=============

The modules below are grouped by where they sit in the DuckSeg pipeline,
in the order a typical analysis uses them. See :doc:`workflow` for a
narrative walkthrough of the same pipeline as it is used in the example
notebook.

Data loading and ROI definition
--------------------------------

.. toctree::
   :maxdepth: 1

   api/experiment_loader
   api/roi_editor

Preprocessing
-------------

.. toctree::
   :maxdepth: 1

   api/image_transformer
   api/image_transforms

Segmentation, tracking, and scoring
-------------------------------------

.. toctree::
   :maxdepth: 1

   api/experiment_evaluator
   api/aligner

Review, filtering, and export
------------------------------

.. toctree::
   :maxdepth: 1

   api/cell_video_visualizer
   api/outlier_filters
   api/video_exporter
