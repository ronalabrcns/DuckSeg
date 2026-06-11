import pandas as pd
from DuckSeg.experiment_evaluator import CellVideo


def export_cell_videos(videos: list[CellVideo], output_path: str, range_modifier=1):
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
