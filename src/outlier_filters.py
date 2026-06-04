import numpy as np

def outlier_vids(vids, manual, filters):
    outliers = set(manual)
    for vid in vids:
        for filter in filters:
            if not filter(vid):
                outliers.add(vid.id)
    return outliers

area_cache = {}
def area(vid: 'CellVideo') -> int:
    if vid.id not in area_cache:
        area_cache[vid.id] = [np.sum(mask) for mask in vid.masks]
    return area_cache[vid.id]

def area_between(start, end):
    def filter(vid):
        return start <= min(area(vid)) and max(area(vid)) < end
    return filter

def max_area_change_between_frames(max_ratio):
    def filter(vid):
        ratios = [area(vid)[i+1]/area(vid)[i] for i in range(len(vid.frames)-1)]
        return all(abs(1-ratio) < max_ratio for ratio in ratios)
    return filter

def max_total_area_change(ratio):
    def filter(vid):
        return 1 - min(area(vid))/max(area(vid)) < ratio
    return filter

def id_starts_with(s):
    def filter(vid):
        return vid.id.startswith(s)
    return filter
