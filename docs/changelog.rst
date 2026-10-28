=========
Changelog
=========

Unreleased
==========

* Segmentation results are cached per video. Moving the ROI lines in the
  ROI editor re-tracks the cells without running CellSAM again
  (:doc:`guides/caching`).
* Interactive ROI editor (:mod:`DuckSeg.roi_editor`) and CSV export
  (:mod:`DuckSeg.video_exporter`).
* New documentation site: installation, concepts, how-to guides,
  troubleshooting and a NumPy-style API reference.
* Fixed ``evalute_batch`` failing because its result list was not
  initialised, ``generate_ROI_ranges`` ending an open stripe at the image
  width instead of its height, ``wt_experiment``/``ko_experiment`` never
  finding an experiment, the unused ``transform`` argument of
  ``align_frame_to_reference``, and ``normalize_videos_for_display``
  ignoring ``range_modifier``.

0.1.0
=====

* First version of the package.
