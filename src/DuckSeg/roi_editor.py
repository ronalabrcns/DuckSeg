import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import blended_transform_factory
import ipywidgets as widgets
from IPython.display import display
import numpy as np
from PIL import Image
import os
from DuckSeg.experiment_loader import generate_ROI_ranges


def _load_ranges_and_shape(experiment):
    roi = experiment.ROI_frame()
    return generate_ROI_ranges(roi), roi.shape


def _save(experiment, centers, heights, roi_shape):
    new_roi = np.zeros(roi_shape, dtype=np.uint8)
    for center, height in zip(centers, heights):
        s = int(round(center - height / 2))
        e = int(round(center + height / 2))
        s = max(0, s)
        e = min(roi_shape[0], e)
        new_roi[s:e, :] = 255
    Image.fromarray(new_roi).save(experiment.corrected_roi_path())


def _load_original_ranges_and_shape(experiment):
    roi = np.array(Image.open(experiment.ROI_path))
    return generate_ROI_ranges(roi), roi.shape


def show_roi_editor(experiment):
    frames = experiment.frames_u8()
    ranges, roi_shape = _load_ranges_and_shape(experiment)
    original_ranges, _ = _load_original_ranges_and_shape(experiment)

    centers = [float(start + end) / 2 for start, end in ranges]
    heights = [float(end - start) for start, end in ranges]
    original_centers = [float(start + end) / 2 for start, end in original_ranges]
    original_heights = [float(end - start) for start, end in original_ranges]
    plt.ioff()

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.canvas.header_visible = False
    n_frames = len(frames)
    mid = min(5, n_frames - 1)
    im = ax.imshow(frames[mid], cmap='viridis', aspect='auto')
    ax.set_title(experiment.name)

    transform = blended_transform_factory(ax.transAxes, ax.transData)
    roi_rects = []
    for center, height in zip(centers, heights):
        rect = patches.Rectangle(
            (0, center - height / 2), 1, height,
            transform=transform, alpha=0.35,
            facecolor='white', edgecolor='white', linewidth=0, zorder=5
        )
        ax.add_patch(rect)
        roi_rects.append(rect)

    play = widgets.Play(value=mid, min=0, max=n_frames - 1, step=1, interval=100)
    slider = widgets.IntSlider(
        value=mid, min=0, max=n_frames - 1, step=1,
        description=f'{experiment.name}:', style={'description_width': 'initial'}
    )
    widgets.jslink((play, 'value'), (slider, 'value'))

    def update_frame(change):
        im.set_data(frames[change['new']])
        fig.canvas.draw_idle()
    slider.observe(update_frame, names='value')

    def update_rect(i):
        roi_rects[i].set_xy((0, centers[i] - heights[i] / 2))
        roi_rects[i].set_height(heights[i])
        fig.canvas.draw_idle()

    # Compute per-ROI button vertical positions to match the canvas
    fig_height_px = fig.get_figheight() * fig.dpi
    axes_pos = ax.get_position()
    axes_top_px = (1 - axes_pos.y1) * fig_height_px
    axes_height_px = axes_pos.height * fig_height_px
    img_height = frames[0].shape[0]

    def center_to_px(c):
        return axes_top_px + (c / img_height) * axes_height_px

    btn_h = 28
    right_items = []
    prev_bottom = 0
    for i in range(len(centers)):
        up_btn = widgets.Button(description='▲', layout=widgets.Layout(width='36px', height=f'{btn_h}px'))
        down_btn = widgets.Button(description='▼', layout=widgets.Layout(width='36px', height=f'{btn_h}px'))

        def make_shift(idx, delta):
            def shift(_):
                centers[idx] += delta
                update_rect(idx)
                _save(experiment, centers, heights, roi_shape)
            return shift

        up_btn.on_click(make_shift(i, -1))
        down_btn.on_click(make_shift(i, 1))

        top_px = int(center_to_px(centers[i])) - btn_h // 2
        spacer_px = max(0, top_px - prev_bottom)
        if spacer_px > 0:
            right_items.append(widgets.Box(layout=widgets.Layout(height=f'{spacer_px}px')))
        right_items.append(widgets.VBox([up_btn, down_btn], layout=widgets.Layout(height=f'{btn_h * 2}px')))
        prev_bottom = top_px + btn_h * 2

    right_panel = widgets.VBox(right_items, layout=widgets.Layout(
        height=f'{int(fig_height_px)}px', width='44px', overflow='hidden'
    ))

    reset_btn = widgets.Button(description='Reset', button_style='warning')
    up_all = widgets.Button(description='▲ all', layout=widgets.Layout(width='60px'))
    down_all = widgets.Button(description='▼ all', layout=widgets.Layout(width='60px'))

    def on_reset(_):
        for i in range(len(centers)):
            centers[i] = original_centers[i]
            heights[i] = original_heights[i]
            update_rect(i)
        corrected = experiment.corrected_roi_path()
        if os.path.exists(corrected):
            os.remove(corrected)

    def shift_all(delta):
        for i in range(len(centers)):
            centers[i] += delta
            update_rect(i)
        _save(experiment, centers, heights, roi_shape)

    reset_btn.on_click(on_reset)
    up_all.on_click(lambda _: shift_all(-1))
    down_all.on_click(lambda _: shift_all(1))
    plt.ion()

    display(widgets.VBox([
        widgets.HBox([play, slider, up_all, down_all, reset_btn]),
        widgets.HBox([fig.canvas, right_panel], layout=widgets.Layout(width='fit-content', align_items='flex-start'))
    ]))


def show_roi_editors(experiments):
    for exp in experiments:
        show_roi_editor(exp)
