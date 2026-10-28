"""
Batch-level orchestration for previewing the masking pipeline.

The single entry point most users need is
:func:`generate_masks_using_filters_and_transforms`, which is what the
example notebook calls to preview cell masks for a subset of experiments
and frames before committing to a full :func:`DuckSeg.experiment_evaluator.evalute_batch`
run. The remaining functions are the building blocks it is assembled from.
"""

import matplotlib.pyplot as plt
from copy import deepcopy
from DuckSeg.experiment_evaluator import generate_mask_transform

def apply_experiment_filter(experiments, filter):
    """
    Select a subset of experiments to process.

    Parameters
    ----------
    experiments : list of DuckSeg.experiment_loader.Experiment
        Experiments to filter.
    filter : None or str or list of int
        - ``None``: keep every experiment.
        - ``str``: keep experiments whose ``name`` contains this substring
          (e.g. ``"KO"`` keeps every knock-out experiment).
        - ``list of int``: keep only the experiments at these positions.

    Returns
    -------
    list of DuckSeg.experiment_loader.Experiment
        The filtered experiments, in their original order.

    Raises
    ------
    AssertionError
        If ``filter`` is a list containing non-integer values, or is of an
        unsupported type.
    """
    if filter is None:
        return list(experiments)
    elif type(filter) is str:
        return [exp for exp in experiments if filter in exp.name]
    elif type(filter) is list:
        assert all(type(i) is int for i in filter), "The members of the experiment filter list must be integers!"
        return [exp for i, exp in enumerate(experiments) if i in filter]
    else:
        raise AssertionError("Unknown filter {filter}")

def calculate_frame_indices(frames, experiment_len):
    """
    Resolve a frame selector into a concrete list of frame indices.

    Parameters
    ----------
    frames : int or list of int
        - ``int``: evenly split the ``[0, experiment_len]`` interval into
          this many sample points (``0`` or ``1`` both select just the
          first frame).
        - ``list of int``: use these frame indices directly.
    experiment_len : int
        Number of frames available in the experiment, used to space out
        samples when ``frames`` is an integer.

    Returns
    -------
    list of int
        The selected frame indices.

    Raises
    ------
    AssertionError
        If ``frames`` is a list containing non-integer values, or is of an
        unsupported type.
    """
    if type(frames) is int:
        if frames in (0,1):
            return [0]
        return [i*experiment_len//(frames-1) for i in range(frames)]
    elif type(frames) is list:
        assert all(type(i) is int for i in frames), "The members of the frames list must be integers!"
        return list(frames)
    else:
        raise AssertionError("Unknown frame filter {filter}")

def show_images_in_row(images, titles=None, main_title="Image Row"):
    """
    Display a list of images side by side in a single matplotlib figure.

    Parameters
    ----------
    images : list of numpy.ndarray
        Images to display, left to right.
    titles : list of str, optional
        Per-image subplot titles. Shorter than ``images`` is fine; the
        remaining images are left untitled.
    main_title : str, optional
        Figure-level title (suptitle).
    """
    n = len(images)
    plt.figure(figsize=(4 * n, 4))
    for i, img in enumerate(images):
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.axis('off')
        if titles and i < len(titles):
            plt.title(titles[i])
    plt.suptitle(main_title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust to make room for suptitle
    plt.show()

def show_experiment_images(experiments, all_frames):
    """
    Display one row of images per experiment, in a stack of figures.

    Parameters
    ----------
    experiments : list of DuckSeg.experiment_loader.Experiment
        Experiments providing the row titles (``experiment.name``).
    all_frames : list of list of numpy.ndarray
        Frames to show for each experiment; ``all_frames[i]`` is the row of
        images shown for ``experiments[i]``.
    """
    for experiment, frames in zip(experiments, all_frames):
        show_images_in_row(frames, main_title=experiment.name)

def apply_transform(transform, all_frames_copy):
    """
    Apply a single-frame transform to every frame of every experiment.

    Parameters
    ----------
    transform : Callable[[numpy.ndarray], numpy.ndarray]
        Function applied independently to each frame.
    all_frames_copy : list of list of numpy.ndarray
        Frames grouped by experiment, as produced by
        :func:`generate_masks_using_filters_and_transforms`.

    Returns
    -------
    list of list of numpy.ndarray
        The transformed frames, in the same nested structure.
    """
    return [[transform(frame) for frame in experiment_frames] for experiment_frames in all_frames_copy]

def apply_transforms(experiments, all_frames, transforms, show_after_transform):
    """
    Apply a pipeline of transforms in sequence, optionally previewing steps.

    Parameters
    ----------
    experiments : list of DuckSeg.experiment_loader.Experiment
        Experiments the frames belong to, used only for preview titles.
    all_frames : list of list of numpy.ndarray
        Frames grouped by experiment, before any transform is applied.
    transforms : list of Callable[[numpy.ndarray], numpy.ndarray]
        Transforms to apply in order, e.g. from
        :mod:`DuckSeg.image_transforms`.
    show_after_transform : list of int
        Indices (supporting negative/from-the-end indexing) of transforms
        after which the intermediate result is previewed with
        :func:`show_experiment_images`.

    Returns
    -------
    list of list of numpy.ndarray
        A deep copy of ``all_frames`` with every transform applied in
        order; the input is left untouched.
    """
    all_frames_copy = deepcopy(all_frames)
    for (transform_id, transform) in enumerate(transforms):
        #print(f"Applying transform {transform_id}")
        all_frames_copy = apply_transform(transform, all_frames_copy)
        if transform_id in show_after_transform or (transform_id - len(transforms)) in show_after_transform:
            show_experiment_images(experiments, all_frames_copy)
    return all_frames_copy

def generate_masks_using_filters_and_transforms(batch, experiment_filter, frame_filter, transforms, show_after_transform):
    """
    Preview cell-mask generation on a subset of experiments and frames.

    This is the function the example notebook uses to tune the
    preprocessing pipeline before running it over the full batch: it
    selects experiments and frames with :func:`apply_experiment_filter` and
    :func:`calculate_frame_indices`, runs ``transforms`` over the selected
    frames (optionally showing intermediate results), then applies
    :func:`DuckSeg.experiment_evaluator.generate_mask_transform` and
    displays the resulting masks.

    Parameters
    ----------
    batch : DuckSeg.experiment_loader.ExperimentBatch
        The experiment batch to preview.
    experiment_filter : None or str or list of int
        Which experiments to process; see :func:`apply_experiment_filter`.
    frame_filter : int or list of int
        Which frames to process from each selected experiment; see
        :func:`calculate_frame_indices`.
    transforms : list of Callable[[numpy.ndarray], numpy.ndarray]
        Preprocessing transforms to apply before masking, e.g. from
        :mod:`DuckSeg.image_transforms`.
    show_after_transform : list of int
        Indices of transforms after which to preview the intermediate
        images; see :func:`apply_transforms`.

    Raises
    ------
    AssertionError
        If ``experiment_filter`` selects zero experiments.
    """
    # apply experiment filter and frame indices
    experiments = apply_experiment_filter(batch.experiments(), experiment_filter)
    assert len(experiments) > 0, "No experiments after applying `experiment_filter`"
    frame_indices = calculate_frame_indices(frame_filter, experiments[0].max_len-1)
    all_frames = []
    for experiment in experiments:
        frames = experiment.frames()
        frames = [frames[i] for i in frame_indices]
        all_frames.append(frames)

    all_frames_copy = apply_transforms(experiments, all_frames, transforms, show_after_transform)

    # apply the masking transform
    all_frames_copy = apply_transform(generate_mask_transform, all_frames_copy)
    show_experiment_images(experiments, all_frames_copy)


