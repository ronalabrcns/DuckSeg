import nd2
import os
import os.path
import dataclasses
import numpy as np
from typing import Optional
from numpy import ndarray
import matplotlib.animation as animation
from IPython.display import HTML, display
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import enum

def nd2_u8_frames(file_name):
    with nd2.ND2File(file_name) as f:
        num_frames = f.attributes.sequenceCount
        return [get_frame_u8(f, i) for i in range(num_frames)]

def nd2_frames(file_name):
    with nd2.ND2File(file_name) as f:
        num_frames = f.attributes.sequenceCount
        return [np.array(f.read_frame(i)) for i in range(num_frames)]

def get_frame_u8(f: nd2.ND2File, i: int) -> np.ndarray:
    return (f.read_frame(i)/(2.**16-1)*255).astype(np.uint8)

def generate_ROI_ranges(roi) -> list[tuple[float]]:
    in_roi = False
    ranges = []
    last_start = None
    j = roi.shape[1]//2
    for i in range(roi.shape[0]):
        if in_roi:
             if roi[i,j] == 0:
                 in_roi = False
                 ranges.append((last_start, i))
        else:
            if roi[i,j] != 0:
                in_roi = True
                last_start = i
    if in_roi:
        ranges.append((last_start, roi.shape[1]))
    return ranges

@dataclasses.dataclass
class Experiment:
    name: str
    path: str
    ROI_path: str
    max_len: Optional[int]
    loaded_video: Optional[list[ndarray]] = None
    loaded_video_u8: Optional[list[ndarray]] = None
    loaded_ROI: Optional[ndarray] = None
    _ROI_ranges: Optional[list[tuple[int]]] = None
    def first_frame_u8(self):
        return self.frame_u8(0)

    def frames(self):
        if self.loaded_video is None:
            self.loaded_video = nd2_frames(self.path)
            if self.max_len is not None:
                self.loaded_video = self.loaded_video[:self.max_len]
        return self.loaded_video

    def frames_u8(self):
        if self.loaded_video_u8 is None:
            self.loaded_video_u8 = nd2_u8_frames(self.path)
            if self.max_len is not None:
                self.loaded_video_u8 = self.loaded_video_u8[:self.max_len]
        return self.loaded_video_u8
    def frame_u8(self, i: int):
        return self.frames_u8()[i]
    def plot_first_frame(self):
        plt.imshow(self.first_frame_u8())
        plt.show()
    def plot_frame(self, i: int):
        plt.imshow(self.frame_u8(i))
        plt.show()
    def play_video(self):
        frames = experiment.frames_u8()
        fig, ax = plt.subplots()
        im = ax.imshow(frames[0])
        plt.close()
        def update(frame):
            im.set_data(frame)
            return [im]
        ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=50)
        
        # Display the animation in the notebook
        a = HTML(ani.to_jshtml())
        display(a)
    def ROI_frame(self):
        if self.loaded_ROI is None:
            image = Image.open(self.ROI_path)
            self.loaded_ROI = np.array(image)
        return self.loaded_ROI

    def plot_ROI(self):
        plt.imshow(self.ROI_frame())
        plt.show()


    def ROI_ranges(self) -> list[tuple[float]]:
        if self._ROI_ranges is None:
            self._ROI_ranges = generate_ROI_ranges(self.ROI_frame())
        return self._ROI_ranges

    def cache_dir(self) -> str:
        return f'{self.path}_cache'
    def cache_path(self, name: str) -> str:
        return os.path.join(self.cache_dir(), name)
    def ensure_cache_dir_exists(self):
        if not os.path.exists(self.cache_dir()):
            os.makedirs(self.cache_dir())
    def save_data(self, name: str, data, force: bool = False):
        cache_path = self.cache_path(name)
        self.ensure_cache_dir_exists()
        assert force or not os.path.exists(cache_path), f"{cache_path} already exists. Use force=True to overwrite"
        with open(cache_path, 'wb') as file:
            pickle.dump(data, file)
    def load_data(self, name: str, force: bool = False):
        try:
            with open(self.cache_path(name), 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None

    def in_batch_id(self) -> str:
        if 'WT' in self.name:
            return f'WT {self.name[-1]}'
        if 'KO' in self.name:
            return f'KO {self.name[-1]}'

class ExperimentType(enum.Enum):
    WT = 0
    KO = 1

@dataclasses.dataclass
class ExperimentBatch:
    name: str
    path: str
    max_len: Optional[int] = None
    def experiment_files(self):
        return os.listdir(self.path)
    def experiments(self) -> list[Experiment]:
        videos = [file for file in self.experiment_files() if file.endswith(".nd2")]
        ROIs = [file[:-4] + "_ROI.tif" for file in videos]
        experiments = []
        for (v, r) in zip(videos, ROIs):
            experiment = Experiment(
                name=v[:-4],
                path=os.path.join(self.path, v),
                ROI_path=os.path.join(self.path, r),
                max_len=self.max_len
            )
            assert os.path.exists(experiment.path)
            #assert os.path.exists(experiment.ROI_path)
            experiments.append(experiment)
        return sorted(experiments, key=lambda exp: exp.name)
    def ko_experiment(self, i: int):
        end = f"00{i}.nd2"
        return [e for e in self.experiments() if "_KO_" in e.path and e.path.endswith(end)][0]
    def wt_experiment(self, i: int):
        end = f"00{i}.nd2"
        return [e for e in self.experiments() if "_WT_" in e.path and e.path.endswith(end)][0]
    def wt_experiments(self):
        return [e for e in self.experiments() if "_WT_" in e.path]
    def ko_experiments(self):
        return [e for e in self.experiments() if "_KO_" in e.path]


    def experiment(self, i: int, type: ExperimentType) -> Experiment:
        if type == ExperimentType.WT:
            return self.wt_experiment(i)
        elif type == ExperimentType.KO:
            return self.ko_experiment(i)

def experiment_batches(data_dir):
    """
    Load the experiments from data_dir
    """
    experiments = os.listdir(data_dir)
    out = {}
    for d in experiments:
        full_dir = os.path.join(data_dir, d)
        name = d[11:]
        out[name] = ExperimentBatch(name, full_dir)
    return out
