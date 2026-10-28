==============
Export results
==============

Write the per-cell, per-frame numbers to a CSV file for statistics or
plotting in other software.

.. code-block:: python

   import os
   from DuckSeg.video_exporter import export_cell_videos

   os.makedirs("output", exist_ok=True)
   kept = [vid for vid in vids if cell_filter(vid)]       # drop the outliers
   export_cell_videos(kept, "output/brightness_scores.csv", range_modifier=range_modifier)

:func:`~DuckSeg.video_exporter.export_cell_videos` exports exactly the
cells you pass in. Filter them first, as above, otherwise rejected cells
end up in the file.

File layout
===========

One row per frame (shorter videos are padded with empty cells) and four
columns per cell, where ``<id>`` is the cell id:

========================  ======================================================
Column                    Content
========================  ======================================================
``<id>_normal``           mean intensity outside the irradiated band
``<id>_ROI``              mean intensity inside the band
``<id>_diff``             ``ROI`` minus ``normal``
``<id>_ratio``            ``ROI`` divided by ``normal``
========================  ======================================================

These are the values behind the ``Diff`` and ``Ratio`` plots, computed
with the ``range_modifier`` you pass.

Baseline-corrected curves
=========================

By default the values are raw. With ``baseline_correct=True`` the
``_diff`` and ``_ratio`` columns are shifted so that each curve starts at
zero (the first frame's value is subtracted). The ``_normal`` and ``_ROI``
columns always stay raw.

.. code-block:: python

   export_cell_videos(kept, "output/brightness_scores.csv",
                      range_modifier=range_modifier, baseline_correct=True)
