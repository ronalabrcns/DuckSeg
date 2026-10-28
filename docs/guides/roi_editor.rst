==========================
Correct the irradiated ROI
==========================

If the stripe in the ROI mask does not sit exactly on the irradiated
region of the video, shift it with the interactive editor. This is
optional and best done before segmentation, but changes made later are
picked up too (see :doc:`caching`).

.. code-block:: python

   %matplotlib widget          # needs the ipympl package
   from DuckSeg.roi_editor import show_roi_editors

   show_roi_editors(batch.experiments())

In Google Colab run ``from google.colab import output;
output.enable_custom_widget_manager()`` first.

One editor is shown per experiment: the video with a slider and play
button, and the detected stripe(s) as a translucent white band.

* The **▲ / ▼** buttons next to a stripe move that stripe by one pixel.
* **▲ all / ▼ all** move all stripes together.
* **Reset** restores the original mask and deletes the corrected file.

Every change is saved immediately as ``<name>_ROI_corrected_ROI.tif`` next
to the original mask (the original is never overwritten). When it exists,
it is used instead of the original mask by all later steps. The editor
moves stripes only; it does not change their height.

.. note::

   Cell ids (``<experiment name>_<number>``) count only the cells that
   touch a stripe, so moving a stripe can change which cell an id refers
   to. After you change a stripe, re-check any ids you noted as outliers.
