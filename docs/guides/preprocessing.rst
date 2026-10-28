===========================
Tune the preprocessing
===========================

CellSAM works best on clean, contrasty 8-bit images. A list of *transforms*
is applied to every frame before segmentation; the goal is that most
nuclei are outlined correctly. Tuning is quick because it only processes a
few frames.

.. code-block:: python

   from DuckSeg.image_transformer import generate_masks_using_filters_and_transforms
   from DuckSeg.image_transforms import convert_to_u8, set_brightness, threshold_image

   transforms = [
       convert_to_u8,        # required: the model cannot handle 16 bit
       set_brightness(20),   # rescale each frame to a mean brightness of 20
       threshold_image(20),  # optional: binary image for a strong signal
   ]

   generate_masks_using_filters_and_transforms(
       batch,
       experiment_filter=[0],        # which experiments
       frame_filter=2,               # which frames
       transforms=transforms,
       show_after_transform=[0, 2],  # also show intermediate results
   )

Parameters
==========

``experiment_filter``
    ``None`` for all experiments, a list of indices such as ``[0, 2, 4]``,
    or a string: only experiments whose name contains it (``'siNT'``).

``frame_filter``
    An integer ``N`` samples ``N`` frames evenly (``1``: first frame,
    ``2``: first and last, ``3``: first, middle, last), or a list of frame
    indices such as ``[0, 10]``.

``show_after_transform``
    Indices into ``transforms``; the frames are displayed after each of
    those steps. Negative indices count from the end.

Available transforms
====================

All are in :mod:`DuckSeg.image_transforms`:

* :func:`~DuckSeg.image_transforms.convert_to_u8` – 16 bit to 8 bit.
* :func:`~DuckSeg.image_transforms.set_brightness` – scale a frame so its
  mean brightness equals the given value.
* :func:`~DuckSeg.image_transforms.adjust_contrast` – linear contrast
  (``alpha``) and brightness (``beta``) adjustment.
* :func:`~DuckSeg.image_transforms.threshold_image` – pixels above the
  threshold become 128, the rest 0.

Write your own
==============

A transform is any function that takes a frame (a NumPy array) and returns
a frame. A factory that returns such a function lets you add parameters:

.. code-block:: python

   import cv2

   def gaussian_blur(sigma):
       def blur(frame):
           return cv2.GaussianBlur(frame, (0, 0), sigma)
       return blur

   transforms = [convert_to_u8, gaussian_blur(1.5), set_brightness(20)]

Use the final list in :func:`~DuckSeg.experiment_evaluator.evaluate_batch`.
Remember that cached results do not notice a change of transforms
(:doc:`caching`).
