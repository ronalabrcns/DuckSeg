"""
CSV export of per-cell laser-ROI brightness measurements.
"""

import pandas as pd
from DuckSeg.experiment_evaluator import CellVideo


def export_cell_videos(videos: list[CellVideo], output_path: str, range_modifier=1):
    """
    Export per-frame in-ROI/out-of-ROI brightness for a set of cells to CSV.

    For every cell, writes four columns: mean brightness outside the ROI
    (``_normal``), inside the ROI (``_ROI``), their difference
    (``_diff``), and their ratio (``_ratio``) — the raw numbers behind the
    ``ScoreType.Diff``/``ScoreType.Ratio`` plots, provided here for
    external analysis or as supplementary data for publication. This is
    the function the example notebook uses to save quantitative results
    alongside its plots.

    Parameters
    ----------
    videos : list of DuckSeg.experiment_evaluator.CellVideo
        Cell videos to export. Videos with differing frame counts are
        padded with missing values (NaN) up to the longest video.
    output_path : str
        Destination CSV file path.
    range_modifier : float, optional
        ROI width factor forwarded to
        :meth:`~DuckSeg.experiment_evaluator.CellVideo.calculate_brightness`.
    """
    columns = {}
    for video in videos:
        vals = video.calculate_brightness(range_modifier=range_modifier)
        normal = vals[:, 0]
        lasered = vals[:, 1]
        columns[f'{video.id}_normal'] = normal
        columns[f'{video.id}_ROI'] = lasered
        columns[f'{video.id}_diff'] = lasered - normal
        columns[f'{video.id}_ratio'] = lasered / normal

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in columns.items()]))
    df.to_csv(output_path, index=False)
