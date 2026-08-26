<img width="1536" height="1024" alt="DuckSeg logo" src="https://github.com/user-attachments/assets/c976cce1-d3c2-4bc3-bc5b-70451415ea06" />

# DuckSeg

**Deep-learning Utility for Cell masKing and Segmentation**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ronalabrcns/DuckSeg/blob/main/notebooks/DuckSeg_Colab.ipynb)
[![Documentation](https://img.shields.io/readthedocs/duckseg)](https://duckseg.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![GitHub issues](https://img.shields.io/github/issues/ronalabrcns/DuckSeg)](https://github.com/ronalabrcns/DuckSeg/issues)

DuckSeg is a Python toolkit for quantifying protein recruitment to sites of
laser-induced DNA damage in live-cell microscopy time-lapse videos. It
combines deep-learning cell segmentation ([CellSAM](https://github.com/vanvalenlab/cellSAM))
with automated per-cell tracking and region-of-interest (ROI) brightness
scoring, replacing what is otherwise a slow, manual, and
reviewer-dependent image-analysis workflow.

📖 **Full documentation, API reference, and a pipeline walkthrough:**
[duckseg.readthedocs.io](https://duckseg.readthedocs.io/)

🚀 **Try it now, no installation required:**
[Open the example notebook in Google Colab](https://colab.research.google.com/github/ronalabrcns/DuckSeg/blob/main/notebooks/DuckSeg_Colab.ipynb)

---

## Background

Laser micro-irradiation assays induce a localized DNA damage stripe across
a nucleus and image, over time, whether a fluorescently tagged protein of
interest (e.g. a DNA repair or mismatch-repair factor) accumulates at that
stripe relative to the rest of the cell. Answering this question at scale —
across many cells, many fields of view, and multiple genotypes (e.g.
wild-type vs. knock-out) — requires:

1. segmenting individual cells in every frame of every video,
2. tracking each cell across the time-lapse despite movement and
   segmentation noise,
3. measuring fluorescence intensity inside the irradiated ROI vs. the rest
   of the cell, per frame, and
4. aggregating and comparing that measurement across cells, experiments,
   and conditions.

DuckSeg automates all four steps, with a human-in-the-loop review stage and
configurable automated outlier filters, so results stay reproducible
without sacrificing quality control.

## Key features

- **Deep-learning cell segmentation** for every frame of every video via
  [CellSAM](https://github.com/vanvalenlab/cellSAM), with a configurable
  preprocessing pipeline to tune masking quality per dataset.
- **Automated per-cell tracking** across a time-lapse using mask-overlap
  matching, robust to modest cell movement between frames.
- **Laser-ROI brightness scoring** — per-frame, per-cell comparison of
  in-ROI vs. out-of-ROI fluorescence, as both an absolute difference and a
  ratio, with batch-level and WT-vs-KO comparison plots built in.
- **Interactive ROI correction** and **interactive outlier review** as
  Jupyter widgets, plus configurable automated outlier filters.
- **CSV export** of per-cell brightness measurements for downstream
  statistics or supplementary data.
- **Runs anywhere Jupyter runs** — a local installation or, with no setup
  at all, the bundled Google Colab notebook.

## Installation

DuckSeg targets Python 3.10+.

Clone the repository:

```bash
git clone https://github.com/ronalabrcns/DuckSeg.git
cd DuckSeg
```

Create and activate a virtual environment:

```bash
python -m venv myenv
source myenv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Install [CellSAM](https://github.com/vanvalenlab/cellSAM/) via pip:

```bash
pip install git+https://github.com/vanvalenlab/cellSAM.git
```

Start Jupyter Lab:

```bash
python -m jupyter lab
```

The `notebooks/` directory contains the example notebooks. Start with
[`DuckSeg_Colab.ipynb`](notebooks/DuckSeg_Colab.ipynb) (also runnable
directly on [Google Colab](https://colab.research.google.com/github/ronalabrcns/DuckSeg/blob/main/notebooks/DuckSeg_Colab.ipynb),
no local install needed) to evaluate an experiment batch end to end.

## Loading experiment data

Point DuckSeg at a folder containing your videos and ROI masks:

- `NAME_00123.nd2` (or `.tif`/`.tiff`) — the raw video
- `NAME_00123_ROI.tif` — the laser-ROI mask, same frame size as the video
- Set the `name` variable to `NAME` (e.g. `MSH2`)

Encode the experimental group in the file name:

- `WT` or `KO`
- e.g. `NAME_WT_00123.tiff`

```python
from DuckSeg.experiment_loader import ExperimentBatch
batch = ExperimentBatch(name, data_dir)
```

## Pipeline parameters

- **`experiment_filter`** — which experiments in the batch are processed:
  - `None` — all experiments
  - list of indices, e.g. `[0, 2, 4]` — only these
  - string, e.g. `'KO'` — only experiments whose name contains this label
- **`frame_filter`** — which frames to process from each experiment:
  - integer `N` — evenly split the video into `N` sample points (`1` = first
    frame only, `2` = first + last, `3` = first + middle + last, …)
  - list of indices, e.g. `[0, 10]` — the specific frames requested
- **`show_after_transform`** — list of transform indices; after applying
  each of these, the intermediate result is displayed (supports negative
  indices to count from the end of the transform list).

## Preprocessing transforms

Transforms are composable `image -> image` functions applied in sequence
before masking (see [`image_transforms.py`](src/DuckSeg/image_transforms.py)):

- `convert_to_u8` — rescale to 8-bit; required before masking
- `set_brightness(value)` — rescale a frame to a target mean brightness
- `adjust_contrast(alpha, beta)` — linear contrast/brightness adjustment
- `threshold_image(value)` — binary threshold, for a strong masking signal

```python
from DuckSeg.image_transforms import convert_to_u8, set_brightness, threshold_image

transforms = [
    convert_to_u8,
    set_brightness(20),
    threshold_image(20),
]
```

You can add new transforms in that same file — anything with the shape
`Callable[[np.ndarray], np.ndarray]` (or a factory returning one) works.

## Usage

1. **Tune preprocessing** — optimize the masking transforms so most cells
   are segmented correctly:

   ```python
   from DuckSeg.image_transformer import generate_masks_using_filters_and_transforms
   generate_masks_using_filters_and_transforms(
       batch, experiment_filter, frame_filter, transforms, show_after_transform
   )
   ```

2. **Segment and track** the full batch (slow; cached to disk):

   ```python
   from DuckSeg.experiment_evaluator import evalute_batch
   evalute_batch(batch, transforms=transforms)
   ```

3. **Filter outliers** — cells where segmentation or tracking failed:
   - Automated filters, added in
     [`outlier_filters.py`](src/DuckSeg/outlier_filters.py):
     - `area_between(10, 5000)` — reject cells whose mask area (in pixels)
       ever leaves this range
     - `max_area_change_between_frames(0.2)` — reject cells with an
       abrupt (>20%) frame-to-frame mask area change
     - `max_total_area_change(0.2)` — reject cells whose mask area drifts
       by more than 20% over the whole video
     - `id_starts_with('REV')` — keep only cells whose ID (derived from
       the file name) starts with this prefix
   - Manual outliers — select outliers by hand in the interactive review
     grid, or by cell ID.
   - Adjust `range_modifier` to widen/narrow the ROI band used when
     checking whether the fluorescence falls where expected.

4. **Visualize the results.**

## Output

`plot_brightness_score_wt_ko` compares brightness intensity between WT and
KO groups, in two modes (`ScoreType`):

1. `Ratio` — in-ROI / out-of-ROI brightness
2. `Diff` — in-ROI − out-of-ROI brightness

Output plot axes:

- x-axis — frame index in the video
- y-axis — difference or ratio

<img width="450" alt="Example brightness score plot" src="https://github.com/user-attachments/assets/a091b8a0-e625-424c-b50e-5909a60b3992" />

Per-cell, per-frame values can also be exported to CSV with
[`export_cell_videos`](src/DuckSeg/video_exporter.py) for downstream
statistics or supplementary data.

## Documentation

The full API reference and a stage-by-stage pipeline walkthrough (mirroring
the example notebook) are on ReadTheDocs:
**[duckseg.readthedocs.io](https://duckseg.readthedocs.io/)**

## Citing DuckSeg

If DuckSeg is useful in your research, please cite the associated
Bioinformatics Advances publication (details to be added upon publication)
and/or link to this repository.

## License

DuckSeg is released under the [MIT License](LICENSE).
