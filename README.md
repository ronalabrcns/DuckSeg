<img width="1536" height="510" alt="DuckSeg logo" src="https://github.com/user-attachments/assets/af3217f1-4586-4ba8-a6a9-5cb529e4dedd" />

# DuckSeg

**Deep-learning Utility for Cell masKing and Segmentation**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ronalabrcns/DuckSeg/blob/main/notebooks/DuckSeg_Example.ipynb)
[![Documentation](https://img.shields.io/readthedocs/duckseg)](https://duckseg.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

DuckSeg measures how a fluorescent protein gathers at sites of
laser-induced DNA damage in live-cell time-lapse videos. It segments every
nucleus with [CellSAM](https://github.com/vanvalenlab/cellSAM), follows each
cell through the video, and compares the brightness inside the irradiated
stripe with the rest of the nucleus, turning a slow, manual analysis into a
reproducible one.

📖 **Documentation:** [duckseg.readthedocs.io](https://duckseg.readthedocs.io/)

## How it works

1. **Load** a folder of videos and their laser-ROI masks.
2. **Correct the stripe** position if needed, in an interactive editor (optional).
3. **Tune the preprocessing** on a few frames so most nuclei are outlined well.
4. **Segment and track** every cell that touches the stripe (results are cached).
5. **Review** the cells in a video grid and drop failures, by hand or with area filters.
6. **Plot and export** the recruitment curves (mean ± SEM) and per-cell CSV data.

## Get started

### Google Colab (no installation)

Open [`DuckSeg_Example.ipynb`](https://colab.research.google.com/github/ronalabrcns/DuckSeg/blob/main/notebooks/DuckSeg_Example.ipynb),
choose a GPU runtime, and add your DeepCell token (from
[users.deepcell.org](https://users.deepcell.org/)) to the Colab *Secrets*
panel as `DEEPCELL_ACCESS_TOKEN`. It runs on an example dataset; switch
`DATA_SOURCE` to `"drive"` to analyse your own videos from Google Drive.

### Local installation

```bash
git clone https://github.com/ronalabrcns/DuckSeg.git
cd DuckSeg
pip install -e .
pip install git+https://github.com/vanvalenlab/cellSAM.git
export DEEPCELL_ACCESS_TOKEN="<your token>"   # Windows: set DEEPCELL_ACCESS_TOKEN=<your token>
python -m jupyter lab
```

A CUDA GPU is strongly recommended. `requirements.txt` is a pinned snapshot
of a working Linux/NVIDIA environment. See the
[installation guide](https://duckseg.readthedocs.io/en/latest/installation.html)
for details.

## Your data

Put one video and one ROI mask per experiment in a folder:

```
MSH2_001.nd2        # the video (.nd2, .tif or .tiff)
MSH2_001_ROI.tif    # the laser stripe: non-zero rows, same size as the video
```

More in [Prepare your data](https://duckseg.readthedocs.io/en/latest/guides/data_preparation.html).

## Quick example

```python
from DuckSeg.experiment_loader import ExperimentBatch
from DuckSeg.image_transforms import convert_to_u8, set_brightness, threshold_image
from DuckSeg.experiment_evaluator import evaluate_batch, plot_brightness_score, ScoreType
from DuckSeg.outlier_filters import area_between, outlier_vids
from DuckSeg.video_exporter import export_cell_videos

batch = ExperimentBatch("MSH2", "/path/to/videos")
batch.max_len = min(e.frame_count() for e in batch.experiments())   # same length for all videos

transforms = [convert_to_u8, set_brightness(20), threshold_image(20)]
vids = evaluate_batch(batch, transforms=transforms)    # segment + track (slow, cached)

outliers = outlier_vids(vids, manual=set(), filters=[area_between(10, 5000)])
cell_filter = lambda vid: vid.id not in outliers

plot_brightness_score(batch, score_type=ScoreType.Ratio, range_modifier=4,
                      cell_filter=cell_filter, baseline_correct=True,
                      save_path="recruitment.png")
export_cell_videos([v for v in vids if cell_filter(v)], "brightness_scores.csv",
                   range_modifier=4)
```

<img width="450" alt="Example recruitment curve (mean ± SEM)" src="https://github.com/user-attachments/assets/a091b8a0-e625-424c-b50e-5909a60b3992" />

## Learn more

| | |
|---|---|
| [Pipeline walkthrough](https://duckseg.readthedocs.io/en/latest/workflow.html) | Every step of the example notebook |
| [How-to guides](https://duckseg.readthedocs.io/en/latest/guides/index.html) | ROI editor, preprocessing, caching, filtering, comparing conditions, export |
| [Concepts](https://duckseg.readthedocs.io/en/latest/concepts.html) | What the scores mean, `range_modifier`, tracking, limitations |
| [Troubleshooting](https://duckseg.readthedocs.io/en/latest/troubleshooting.html) | Common errors and fixes |
| [API reference](https://duckseg.readthedocs.io/en/latest/modules.html) | All modules and functions |

## Citing DuckSeg

If DuckSeg helps your research, please cite the associated Bioinformatics
Advances publication (details to follow) and link to this repository.

## License

[MIT](LICENSE)
