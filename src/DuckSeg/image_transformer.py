import matplotlib.pyplot as plt
from copy import deepcopy
from DuckSeg.experiment_evaluator import generate_mask_transform

def apply_experiment_filter(experiments, filter):
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
    for experiment, frames in zip(experiments, all_frames):
        show_images_in_row(frames, main_title=experiment.name)

def apply_transform(transform, all_frames_copy):
    return [[transform(frame) for frame in experiment_frames] for experiment_frames in all_frames_copy]

def apply_transforms(experiments, all_frames, transforms, show_after_transform):
    all_frames_copy = deepcopy(all_frames)
    for (transform_id, transform) in enumerate(transforms):
        #print(f"Applying transform {transform_id}")
        all_frames_copy = apply_transform(transform, all_frames_copy)
        if transform_id in show_after_transform or (transform_id - len(transforms)) in show_after_transform:
            show_experiment_images(experiments, all_frames_copy)
    return all_frames_copy

def generate_masks_using_filters_and_transforms(batch, experiment_filter, frame_filter, transforms, show_after_transform):
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


