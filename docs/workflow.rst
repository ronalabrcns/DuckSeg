=====================
Pipeline Walkthrough
=====================

This page walks through the DuckSeg pipeline stage by stage, in the same
order used by the `example Colab notebook
<https://colab.research.google.com/github/ronalabrcns/DuckSeg/blob/main/notebooks/DuckSeg_Colab.ipynb>`_.
Each stage links to the API reference entry for the functions and classes
involved, so this page doubles as a map from "what the notebook does" to
"where that lives in the code."

.. contents::
   :local:
   :depth: 1

1. Load an experiment batch
============================

A :class:`~DuckSeg.experiment_loader.ExperimentBatch` represents one
directory of paired video (``.nd2``/``.tif``/``.tiff``) and ROI mask
files. Discovering its individual experiments is lazy and file-name
driven — see the :mod:`~DuckSeg.experiment_loader` module docstring for
the expected naming convention. ::

    from DuckSeg.experiment_loader import ExperimentBatch
    batch = ExperimentBatch(name, data_dir)

Key API:

- :class:`DuckSeg.experiment_loader.ExperimentBatch`
- :class:`DuckSeg.experiment_loader.Experiment`
- :func:`DuckSeg.experiment_loader.generate_ROI_ranges`

2. Correct ROI boundaries (optional)
======================================

Automatic ROI-range detection from the mask image can be imprecise. The
interactive editor overlays the detected laser ROI band(s) on a
scrubbable view of the video; adjustments are saved next to the original
ROI file and picked up transparently by every later stage. ::

    from DuckSeg.roi_editor import show_roi_editors
    show_roi_editors(batch.experiments())

Key API:

- :func:`DuckSeg.roi_editor.show_roi_editor`
- :func:`DuckSeg.roi_editor.show_roi_editors`

3. Tune the masking preprocessing
====================================

Before running full-batch segmentation, the notebook previews cell masks
on a handful of experiments/frames to tune the preprocessing transform
chain (contrast, brightness, thresholding). ::

    from DuckSeg.image_transformer import generate_masks_using_filters_and_transforms
    from DuckSeg.image_transforms import convert_to_u8, threshold_image, set_brightness

    transforms = [convert_to_u8, set_brightness(20), threshold_image(20)]
    generate_masks_using_filters_and_transforms(
        batch, experiment_filter, frame_filter, transforms, show_after_transform
    )

Key API:

- :func:`DuckSeg.image_transformer.generate_masks_using_filters_and_transforms`
- :mod:`DuckSeg.image_transforms` (:func:`~DuckSeg.image_transforms.convert_to_u8`,
  :func:`~DuckSeg.image_transforms.threshold_image`,
  :func:`~DuckSeg.image_transforms.adjust_contrast`,
  :func:`~DuckSeg.image_transforms.set_brightness`)

4. Segment and track cells across the batch
==============================================

With a preprocessing pipeline chosen, :func:`~DuckSeg.experiment_evaluator.evalute_batch`
runs CellSAM segmentation on every frame of every experiment and tracks
each ROI-overlapping cell across time, caching the result to disk. This
is the most compute-intensive step. ::

    from DuckSeg.experiment_evaluator import evalute_batch
    evalute_batch(batch, transforms=transforms)

Key API:

- :func:`DuckSeg.experiment_evaluator.evalute_batch`
- :class:`DuckSeg.experiment_evaluator.ExperimentEvaluator` — see its
  class docstring for the full stage-by-stage pipeline
  (:meth:`~DuckSeg.experiment_evaluator.ExperimentEvaluator.create_starting_boxes`
  through
  :meth:`~DuckSeg.experiment_evaluator.ExperimentEvaluator.create_cell_video_inds3`)
- :class:`DuckSeg.experiment_evaluator.CellVideo` — one tracked cell's
  frames, masks, and laser ROI range

5. Review tracked cells and filter outliers
==============================================

Segmentation and tracking occasionally fail for individual cells. DuckSeg
combines automated filters with a manual visual review grid: ::

    from DuckSeg.cell_video_visualizer import normalize_videos_for_display, show_video_grid_jupyter
    from DuckSeg.outlier_filters import area_between, outlier_vids

    vids2 = normalize_videos_for_display(vids, range_modifier=range_modifier)
    update_outliers = show_video_grid_jupyter(vids2, manual_outliers=manual_outliers)

    filters = [area_between(10, 5000)]
    outliers = outlier_vids(vids, extra_manual_outliers, filters)

Key API:

- :mod:`DuckSeg.outlier_filters` (:func:`~DuckSeg.outlier_filters.outlier_vids`,
  :func:`~DuckSeg.outlier_filters.area_between`,
  :func:`~DuckSeg.outlier_filters.max_area_change_between_frames`,
  :func:`~DuckSeg.outlier_filters.max_total_area_change`,
  :func:`~DuckSeg.outlier_filters.id_starts_with`)
- :func:`DuckSeg.cell_video_visualizer.normalize_videos_for_display`
- :func:`DuckSeg.cell_video_visualizer.show_video_grid_jupyter`

6. Score and visualize laser-ROI brightness
==============================================

The scientific readout: for every retained cell, compare fluorescence
intensity inside the laser-irradiated ROI row range against the rest of
the cell, over time, then aggregate and plot across a batch or across
WT/KO groups. ::

    from DuckSeg.experiment_evaluator import ScoreType, plot_brightness_score
    plot_brightness_score(batch, score_type=ScoreType.Ratio,
                           range_modifier=range_modifier, cell_filter=cell_filter)

Key API:

- :meth:`DuckSeg.experiment_evaluator.CellVideo.calculate_brightness` and
  its ``calculate_brightness_diff``/``calculate_brightness_ratio``/
  ``calculate_brightness_score`` siblings
- :class:`DuckSeg.experiment_evaluator.ScoreType`
- :func:`DuckSeg.experiment_evaluator.plot_brightness_score`
- :func:`DuckSeg.experiment_evaluator.plot_brightness_score_wt_ko`
- :func:`DuckSeg.experiment_evaluator.plot_all_vid_scores`

7. Export results
===================

Per-cell, per-frame brightness values can be exported to CSV for
statistics or supplementary data. ::

    from DuckSeg.video_exporter import export_cell_videos
    export_cell_videos(vids, "output/brightness_scores.csv", range_modifier=range_modifier)

Key API:

- :func:`DuckSeg.video_exporter.export_cell_videos`

See also
========

- :doc:`index` for installation and an overview of the tool.
- :doc:`modules` for the complete, module-by-module API reference.
