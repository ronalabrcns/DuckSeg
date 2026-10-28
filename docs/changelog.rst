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
* Optional ``baseline_correct=True`` for the score, plot and CSV export
  functions, and ``save_path=`` to save plot figures.
* ``evaluate_batch`` is the correctly spelled name of ``evalute_batch``,
  which remains as an alias.
* Each module now defines its public API with ``__all__``; internal
  helpers are no longer listed in the API reference.
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
