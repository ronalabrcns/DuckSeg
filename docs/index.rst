================================
DuckSeg Documentation
================================

.. **DuckSeg: Deep-learning Utility for Cell masKing and Segmentation**

.. image:: /_static/duckseg_logo_nobg.png
   :alt: DuckSeg logo
   :align: left
   :width: 400px
   :class: dark-light

DuckSeg is a Python toolkit for quantifying protein recruitment to sites
of laser-induced DNA damage in live-cell microscopy time-lapse videos. It
combines deep-learning cell segmentation (`CellSAM
<https://github.com/vanvalenlab/cellSAM>`_) with automated per-cell
tracking and region-of-interest (ROI) brightness scoring, replacing what
is otherwise a slow, manual, and reviewer-dependent image-analysis
workflow.

.. raw:: html

   <p>
     <a href="https://colab.research.google.com/github/ronalabrcns/DuckSeg/blob/main/notebooks/DuckSeg_Colab.ipynb">
       <img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
     </a>
     <a href="https://github.com/ronalabrcns/DuckSeg">
       <img alt="GitHub" src="https://img.shields.io/badge/source-GitHub-181717?logo=github">
     </a>
     <a href="https://opensource.org/licenses/MIT">
       <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green">
     </a>
   </p>

.. admonition:: Try it now, no installation required
   :class: tip

   The fastest way to see DuckSeg in action — including on your own
   microscopy data — is the example notebook on Google Colab:

   `Open DuckSeg_Colab.ipynb in Google Colab
   <https://colab.research.google.com/github/ronalabrcns/DuckSeg/blob/main/notebooks/DuckSeg_Colab.ipynb>`_

   It clones the repository, installs dependencies (including CellSAM),
   and walks through the full pipeline end to end on a GPU runtime
   provided free of charge by Colab.

Background
==========

Laser micro-irradiation assays induce a localized DNA damage stripe
across a nucleus and image, over time, whether a fluorescently tagged
protein of interest (e.g. a DNA repair or mismatch-repair factor)
accumulates at that stripe relative to the rest of the cell. Answering
this question at scale — across many cells, many fields of view, and
multiple genotypes (e.g. wild-type vs. knock-out) — requires:

1. segmenting individual cells in every frame of every video,
2. tracking each cell across the time-lapse despite movement and
   segmentation noise,
3. measuring fluorescence intensity inside the irradiated ROI vs. the
   rest of the cell, per frame, and
4. aggregating and comparing that measurement across cells, experiments,
   and conditions.

DuckSeg automates all four steps, with a human-in-the-loop review stage
(:func:`~DuckSeg.cell_video_visualizer.show_video_grid_jupyter`) and
configurable automated filters
(:mod:`~DuckSeg.outlier_filters`) for catching cells where segmentation
or tracking failed, so results stay reproducible without sacrificing
quality control.

Key features
============

- **Deep-learning cell segmentation** for every frame of every video via
  `CellSAM <https://github.com/vanvalenlab/cellSAM>`_, with a
  configurable preprocessing pipeline
  (:mod:`~DuckSeg.image_transforms`, :mod:`~DuckSeg.image_transformer`)
  to tune masking quality per dataset.
- **Automated per-cell tracking** across a time-lapse using mask-overlap
  matching (:meth:`~DuckSeg.experiment_evaluator.ExperimentEvaluator.create_cell_video_inds3`),
  robust to modest cell movement between frames.
- **Laser-ROI brightness scoring** — per-frame, per-cell comparison of
  in-ROI vs. out-of-ROI fluorescence, as both an absolute difference and
  a ratio (:class:`~DuckSeg.experiment_evaluator.ScoreType`), with
  batch-level and WT-vs-KO comparison plots built in.
- **Interactive ROI correction** (:mod:`~DuckSeg.roi_editor`) and
  **interactive outlier review** (:mod:`~DuckSeg.cell_video_visualizer`)
  as Jupyter widgets, plus configurable automated outlier filters
  (:mod:`~DuckSeg.outlier_filters`).
- **CSV export** (:mod:`~DuckSeg.video_exporter`) of per-cell brightness
  measurements for downstream statistics or supplementary data.
- **Runs anywhere Jupyter runs** — a local installation or, with no setup
  at all, the bundled Google Colab notebook.

How it works
============

See :doc:`workflow` for a full stage-by-stage walkthrough of the
pipeline, matching the example notebook, with links into the API
reference for every function and class involved. In brief:

.. code-block:: text

    ExperimentBatch (load videos + ROI masks)
        -> roi_editor (optional manual ROI correction)
        -> image_transforms / image_transformer (preprocessing, tuned interactively)
        -> evalute_batch -> ExperimentEvaluator (CellSAM segmentation + per-cell tracking)
        -> cell_video_visualizer + outlier_filters (review and filter tracked cells)
        -> CellVideo.calculate_brightness_score + plot_brightness_score* (quantify + visualize)
        -> video_exporter (CSV export)

Installation
============

DuckSeg targets Python 3.10+. For a local installation with GPU support::

    git clone https://github.com/ronalabrcns/DuckSeg.git
    cd DuckSeg
    python -m venv myenv
    source myenv/bin/activate
    python -m pip install -r requirements.txt
    pip install git+https://github.com/vanvalenlab/cellSAM.git
    pip install -e .

Then launch Jupyter and open a notebook from ``notebooks/``::

    python -m jupyter lab

No local GPU or environment setup? Use the
`Google Colab notebook <https://colab.research.google.com/github/ronalabrcns/DuckSeg/blob/main/notebooks/DuckSeg_Colab.ipynb>`_
instead — it installs everything it needs, including CellSAM, in the
first few cells.

Quickstart
==========

The essential shape of a DuckSeg analysis, abbreviated from the example
notebook (see :doc:`workflow` for the full walkthrough):

.. code-block:: python

    from DuckSeg.experiment_loader import ExperimentBatch
    from DuckSeg.image_transforms import convert_to_u8, set_brightness, threshold_image
    from DuckSeg.experiment_evaluator import (
        evalute_batch, ExperimentEvaluator, ScoreType, plot_brightness_score,
    )

    batch = ExperimentBatch("MSH2", "/path/to/videos")

    transforms = [convert_to_u8, set_brightness(20), threshold_image(20)]
    evalute_batch(batch, transforms=transforms)  # segment + track (slow, cached)

    vids = [v for exp in batch.experiments()
              for v in ExperimentEvaluator.load_experiment(exp).cell_videos()]

    plot_brightness_score(batch, score_type=ScoreType.Ratio, range_modifier=4)

Data expected on disk, per experiment, inside a batch directory::

    <NAME>_<WT|KO>_<...>.nd2        # or .tif / .tiff
    <NAME>_<WT|KO>_<...>_ROI.tif    # binary laser-ROI mask, same frame size

Citing DuckSeg
==============

If DuckSeg is useful in your research, please cite the associated
Bioinformatics Advances publication (details to be added upon
publication) and/or link to the `GitHub repository
<https://github.com/ronalabrcns/DuckSeg>`_.

License
=======

DuckSeg is released under the MIT License.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   workflow
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
