===============
Troubleshooting
===============

CellSAM fails to start or asks for authentication
    CellSAM downloads its model with a DeepCell access token. Make sure
    ``DEEPCELL_ACCESS_TOKEN`` is set in the environment of the notebook
    kernel (in Colab: Secrets, see :doc:`installation`). Restart the
    kernel after setting it.

``CUDA out of memory``
    DuckSeg frees the GPU cache and retries once. If it keeps failing,
    restart the kernel, close other GPU programs, or work on fewer
    experiments at a time (the cache lets you continue where you stopped,
    see :doc:`guides/caching`).

``No experiments after applying `experiment_filter```
    The filter in :doc:`guides/preprocessing` matched nothing. Check the
    index list or the string against the file names.

``Unsupported video file format``
    Only ``.nd2``, ``.tif`` and ``.tiff`` videos are read. A mask called
    ``..._ROI.tiff`` is treated as a video; name masks ``..._ROI.tif``
    (:doc:`guides/data_preparation`).

The ROI editor or review grid shows nothing
    The widgets need an interactive backend. Run ``%matplotlib widget``
    (package ``ipympl``) before the ROI editor. In Colab also run
    ``from google.colab import output; output.enable_custom_widget_manager()``.

Videos in the review grid cannot be encoded
    ``video_widget`` encodes MP4 through ``imageio-ffmpeg``. Install it with
    ``pip install imageio-ffmpeg``.

Changed the preprocessing but the results look the same
    Cached results do not notice new ``transforms``. Rerun
    :func:`~DuckSeg.experiment_evaluator.evalute_batch` with
    ``use_cache=False`` (:doc:`guides/caching`).

``ROI ranges modified after evaluate_batch!``
    A cell no longer touches any stripe of the current ROI. Rerun
    :func:`~DuckSeg.experiment_evaluator.evalute_batch`; unchanged
    segmentation is reused and the cells are re-tracked.

The plots are missing a cell or show a gap
    Cells whose score contains NaN are dropped from the averages. This
    happens when a mask has no pixels inside or outside the band for some
    frame, for example when ``range_modifier`` makes the band cover the
    entire cell. Lower ``range_modifier``.

Experiments cannot be compared because they have different lengths
    Set ``batch.max_len`` to the shortest recording
    (:doc:`guides/data_preparation`). A cache made with another length is
    discarded automatically.
