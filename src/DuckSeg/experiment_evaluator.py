import dataclasses
from DuckSeg.experiment_loader import ExperimentBatch, Experiment, ExperimentType
from typing import Optional
import numpy as np
from numpy import ndarray
import enum
from DuckSeg.aligner import transform_2d, BilateralFilter
import cv2
import torch

from cellSAM import segment_cellular_image, get_model
import matplotlib.pyplot as plt

cellsam_model = None


class ScoreType(enum.Enum):
    Diff = 0
    Ratio = 1

@dataclasses.dataclass
class BoundingBox:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def enlarged(self, percent: float):
        """
        returns an enlarged bounding box
        """
        x_diff = (self.x_max - self.x_min)*percent/100/2
        y_diff = (self.y_max - self.y_min)*percent/100/2
        return BoundingBox(self.x_min - x_diff, self.x_max + x_diff, self.y_min - y_diff, self.y_max + y_diff)

    def from_cellsam_bounding_box(box: list[float]):
        return BoundingBox(box[1], box[3], box[0], box[2])

    def cellsam_bounding_box(self) -> list[float]:
        return [self.y_min, self.x_min, self.y_max, self.x_max]

    def cut(self, array):
        x_min = max(0, int(self.x_min))
        x_max = max(0, int(self.x_max))
        y_min = max(0, int(self.y_min))
        y_max = max(0, int(self.y_max))
        return array[x_min:x_max,y_min:y_max]

    def as_indices(self):
        x_min = max(0, int(self.x_min))
        x_max = max(0, int(self.x_max))
        y_min = max(0, int(self.y_min))
        y_max = max(0, int(self.y_max))
        return BoundingBox(x_min, x_max, y_min, y_max)

    def intersection_area(self, other: 'BoundingBox') -> float:
        # Calculate the overlapping area
        x_overlap = max(0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        y_overlap = max(0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))

        # If the boxes don't overlap, either dimension will be zero
        return x_overlap * y_overlap
    def intersection_box(self, other: 'BoundingBox') -> 'BoundingBox':
        x_min = max(self.x_min, other.x_min)
        x_max = min(self.x_max, other.x_max)
        y_min = max(self.y_min, other.y_min)
        y_max = min(self.y_max, other.y_max)
        return BoundingBox(x_min, x_max, y_min, y_max)

    def subtract_from(self, other: 'BoundingBox') -> 'BoundingBox':
        x_min = other.x_min - self.x_min
        x_max = other.x_max - self.x_min
        y_min = other.y_min - self.y_min
        y_max = other.y_max - self.y_min
        return BoundingBox(x_min, x_max, y_min, y_max)


    def intersects_y_range(self, y_min: int, y_max: int) -> bool:
        return self.x_max > y_min and y_max > self.x_min

    def range_to_relative(self, range: tuple[float]) -> tuple[float]:
        return (range[0] - self.x_min, range[1] - self.x_min)
        
def evalute_batch(batch: ExperimentBatch, start = 1, transforms=None, use_cache=True):
    for experiment in batch.experiments():
        print(f"Evaluating {experiment.name}")
        if use_cache:
            try:
                wt_eval = ExperimentEvaluator.load_experiment(experiment)
                print("Loaded from cache")
                continue
            except:
                pass
        wt_eval = evalute_experiment1(experiment, transforms=transforms)
        wt_eval.save(force=True)
        
def print_counts(arr):
    unique_elements, counts = np.unique(arr, return_counts=True)
    
    # Print each element and its count
    for element, count in zip(unique_elements, counts):
        print(f"Element: {element}, Count: {count}")

def get_mask_num(arr):
    unique_elements, counts = np.unique(arr, return_counts=True)
    max_count = 0
    max_value = 0
    for element, count in zip(unique_elements, counts):
        if max_count < count and element != 0:
            max_count = count
            max_value = element
    return max_value

def generate_mask_inner(frame, bounding_boxes=None):
    global cellsam_model
    if cellsam_model is None:
        cellsam_model = get_model()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask, embedding, bounding_boxes = segment_cellular_image(frame, device=device.type, model = cellsam_model, bounding_boxes=bounding_boxes, bbox_threshold=0.7)
    except torch.cuda.OutOfMemoryError:
        """
        torch can run out of memory on the gpu, reset the session and try again
        """
        print("Out of memory. Restting cuda session!")
        #numba.cuda.select_device(0)
        #numba.cuda.close()
        #cellsam_model = get_model()
        torch.cuda.empty_cache()
        mask, embedding, bounding_boxes = segment_cellular_image(frame, device=device.type, model = cellsam_model, bounding_boxes=bounding_boxes)
    return mask, bounding_boxes

def generate_mask_transform(frame):
    mask, _ = generate_mask_inner(frame)
    return mask

def generate_mask(frame, threshold_value = 1, bounding_boxes=None, transforms=None):
    if transforms is not None:
        #print(f"Applying {len(transforms)} transforms")
        frame = deepcopy(frame)
        for transform in transforms:
            frame = transform(frame)
    elif threshold_value is not None:
        ret, thresh_img = cv2.threshold(frame, threshold_value, 255, cv2.THRESH_BINARY)
        frame = thresh_img*128
    return generate_mask_inner(frame, bounding_boxes=bounding_boxes)

from copy import deepcopy

def apply_transform(transform, all_frames_copy):
    return [[transform(frame) for frame in experiment_frames] for experiment_frames in all_frames_copy]

def apply_transforms(experiments, all_frames, transforms, show_after_transform = None):
    all_frames_copy = deepcopy(all_frames)
    for (transform_id, transform) in enumerate(transforms):
        #print(f"Applying transform {transform_id}")
        all_frames_copy = apply_transform(transform, all_frames_copy)
        if show_after_transform and (transform_id in show_after_transform or (transform_id - len(transforms)) in show_after_transform):
            show_experiment_images(experiments, all_frames_copy)
    return all_frames_copy

@dataclasses.dataclass
class ExperimentEvaluator:
    experiment: Experiment
    # boxes for the starting frame
    starting_boxes: Optional[list[BoundingBox]] = None
    starting_mask: Optional[ndarray] = None
    # starting boxes intersecting the lines
    filtered_starting_boxes: Optional[list[BoundingBox]] = None
    filtered_starting_inds: Optional[list[int]] = None
    # Filtered (only ones intersecting the lines) boxes for each frame
    boxes: Optional[list[list[BoundingBox]]] = None
    masks: Optional[list[ndarray]] = None
    # Cell images grouped by frame
    cell_image_lists: Optional[list[list[ndarray]]] = None
    cell_mask_lists: Optional[list[list[ndarray]]] = None
    cell_video_inds: Optional[list[list[int]]] = None

    def create_starting_boxes(self, threshold_value = 1, transforms=None):
        frame = self.experiment.first_frame_u8()
        mask, bounding_boxes = generate_mask(frame, threshold_value=threshold_value, transforms=transforms)
        self.starting_boxes = [BoundingBox.from_cellsam_bounding_box(box) for box in bounding_boxes]
        self.starting_mask = mask

    def filter_starting_boxes(self):
        ranges = self.experiment.ROI_ranges()
        self.filtered_starting_boxes = [box for box in self.starting_boxes if any(box.intersects_y_range(*laser_range) for laser_range in ranges)]
        self.filtered_starting_inds = [i for i, box in enumerate(self.starting_boxes) if any(box.intersects_y_range(*laser_range) for laser_range in ranges)]

    def create_boxes(self, limit = None, transforms=None):
        experiment = self.experiment
        num_frames = len(experiment.frames_u8())
        if self.boxes is not None and limit is not None and len(self.boxes) >= limit:
            return
        boxes = []
        masks = []
        self.boxes = boxes
        self.masks = masks
        #last_boxes = self.filtered_starting_boxes
        for (i, frame) in enumerate(experiment.frames_u8()):
            if limit is not None and i == limit:
                break
            print(f"Running frame {i+1}/{num_frames}")
            # enlarge the last bounding boxes by 10% and pass it to cellsam
            #enlarged = [box.enlarged(10).cellsam_bounding_box() for box in last_boxes]
            mask, bounding_boxes = generate_mask(frame, bounding_boxes=None, transforms=transforms)
            masks.append(mask)
            boxes.append([BoundingBox.from_cellsam_bounding_box(box) for box in bounding_boxes])
            #last_boxes = boxes[-1]
    def create_masked_cells(self):
        self.cell_image_lists = []
        self.cell_mask_lists = []
        for i, (boxes, mask, frame) in enumerate(zip(self.boxes, self.masks, self.experiment.frames())):
            cell_images = []
            cell_masks = []
            for box in boxes:
                boxed_frame = box.cut(frame)
                boxed_mask = box.cut(mask)
                #print_counts(boxed_mask)
                mask_val = get_mask_num(boxed_mask)
                boxed_mask = boxed_mask == mask_val
                masked_boxed_frame = boxed_frame * boxed_mask
                cell_images.append(masked_boxed_frame)
                cell_masks.append(boxed_mask)
            self.cell_image_lists.append(cell_images)
            self.cell_mask_lists.append(cell_masks)
    def generate_ROI_ranges(self):
        self.cell_ROI_image_list = []
        self.cell_ROI_range_list = []
        for i, (boxes, mask, frame) in enumerate(zip(self.boxes, self.masks, self.experiment.frames())):
            image_list = []
            range_list = []
            for box in boxes:
                boxed_roi = box.cut(self.experiment.ROI_frame())
                boxed_roi_ranges = generate_ROI_ranges(boxed_roi)
                image_list.append(boxed_roi)
                range_list.append(boxed_roi_ranges)
            self.cell_ROI_image_list.append(image_list)
            self.cell_ROI_range_list.append(range_list)
    def create_cell_video_inds(self):
        self.cell_video_inds = []
        for box in self.filtered_starting_boxes:
            inds = []
            for boxes in self.boxes:
                ind = np.argmax([float(box.intersection_area(next_box)) for next_box in boxes])
                inds.append(ind)
            self.cell_video_inds.append(inds)
    def create_cell_video_inds2(self):
        self.cell_video_inds = []
        for box in self.filtered_starting_boxes:
            curr_box = box
            inds = []
            for boxes in self.boxes:
                ind = np.argmax([float(curr_box.intersection_area(next_box)) for next_box in boxes])
                inds.append(ind)
                curr_box = boxes[ind]
            self.cell_video_inds.append(inds)
    def create_cell_video_inds3(self):
        """
        uses overlapping mask area instead of overlapping box area
        """
        self.cell_video_inds = []
        ranges = self.experiment.ROI_ranges()
        i = 0
        for box, mask in zip(self.boxes[0], self.cell_mask_lists[0]):
            i += 1
            print(f"calculating {i}/{len(self.boxes[0])}")
            if not any(box.intersects_y_range(*laser_range) for laser_range in ranges):
                continue
            curr_box = box
            curr_mask = mask
            inds = []
            for (boxes, masks) in zip(self.boxes, self.cell_mask_lists):
                diffs = [float(mask_diff(curr_box, curr_mask, next_box, next_mask)) for next_box, next_mask in zip(boxes, masks)]
                ind = np.argmin(diffs)
                # if we didn't create a mask for a cell in this frame, use the prvious one
                if diffs[ind] > np.sum(curr_mask) * 0.6:
                    inds.append(inds[-1])
                else:
                    i_j = (len(inds), ind)
                    inds.append(i_j)
                    curr_box = boxes[ind]
                    curr_mask = masks[ind]
            self.cell_video_inds.append(inds)
    def show_cell_video(self, ind: int):
        for frame in self.cell_frames(ind):
            plt.imshow(frame)
            plt.show()

    def cell_frames(self, ind):
        return [cell_image_list[j] for cell_image_list, j in zip(self.cell_image_lists, self.cell_video_inds[ind])]
    def play_cell_video(self, ind: int):
        frames = self.cell_frames(ind)
        play_nb_video(frames)
    def cell_video(self, ind: int) -> 'CellVideo':
        masks = []
        frames = []
        boxes = []
        exp_frames = self.experiment.frames()
        for real_ind, (i, j) in enumerate(self.cell_video_inds[ind]):
            frame = exp_frames[real_ind]
            box = self.boxes[i][j]
            boxed_mask = self.cell_mask_lists[i][j]
            boxed_frame = box.cut(frame)
            masked_boxed_frame = boxed_frame * boxed_mask
            frames.append(masked_boxed_frame)
            #frames.append(self.cell_image_lists[i][j])
            masks.append(self.cell_mask_lists[i][j])
            boxes.append(self.boxes[i][j])
        return CellVideo(id=f"{self.experiment.name}_{ind}", frames=frames, boxes=boxes, masks=masks, laser_ranges=self.experiment.ROI_ranges())
    def cell_videos(self) -> list['CellVideo']:
        return [self.cell_video(i) for i in range(len(self.cell_video_inds))]
            
    @staticmethod
    def from_other(other):
        import types
        new = ExperimentEvaluator(other.experiment)
        for name, value in other.__dict__.items():
            # Check if the attribute is not a method
            if not isinstance(value, types.FunctionType):
                # Set the attribute in the destination class
                setattr(new, name, value)
        return new

    def save(self, force=False):
        self.experiment.save_data("evaluator", self, force=force)

    @staticmethod
    def load_experiment(experiment: Experiment):
        old = experiment.load_data("evaluator")
        old.experiment = experiment
        #old.cell_videos = ExperimentEvaluator.cell_videos
        return ExperimentEvaluator.from_other(old)

def mask_diff(box1, mask1, box2, mask2):
    box1 = box1.as_indices()
    box2 = box2.as_indices()
    intersection_box = box1.intersection_box(box2)
    intersection_in_mask1 = box1.subtract_from(intersection_box).cut(mask1)
    intersection_in_mask2 = box2.subtract_from(intersection_box).cut(mask2)
    common = np.sum(intersection_in_mask1 ^ intersection_in_mask2)
    m1 = np.sum(mask1) - np.sum(intersection_in_mask1)
    m2 = np.sum(mask2) - np.sum(intersection_in_mask2)
    return common + m1 + m2
    
def evalute_experiment1(experiment: Experiment, limit = None, transforms=None):
    evaluator = ExperimentEvaluator(experiment)
    evaluator.create_starting_boxes(transforms=transforms)
    evaluator.filter_starting_boxes()
    evaluator.create_boxes(limit = limit, transforms=transforms)
    evaluator.create_masked_cells()
    #evaluator.create_cell_video_inds()
    #evaluator.create_cell_video_inds2()
    evaluator.create_cell_video_inds3()
    return evaluator

def zero_pad_bottom_right_2d(img, target_shape):
    H, W = img.shape
    target_H, target_W = target_shape
    pad_bottom = max(0, target_H - H)
    pad_right = max(0, target_W - W)
    return np.pad(img, ((0, pad_bottom), (0, pad_right)), mode='constant')
    
@dataclasses.dataclass
class CellVideo:
    id: str
    frames: list[ndarray]
    boxes: list[BoundingBox]
    masks: list[ndarray]
    laser_ranges: list[tuple[float]]

    def map(self, fn) -> 'CellVideo':
        """
        transforms the frames of a cell_video
        """
        return CellVideo(id=self.id, frames=[fn(frame.copy()) for frame in self.frames], masks = self.masks, laser_ranges = self.laser_ranges)

    def homogenize_size(self) -> 'CellVideo':
        shape0 = max(frame.shape[0] for frame in self.frames)
        shape1 = max(frame.shape[1] for frame in self.frames)
        shape = (shape0, shape1)
        return CellVideo(id=self.id, frames = [zero_pad_bottom_right_2d(frame, shape) for frame in self.frames], masks = [zero_pad_bottom_right_2d(mask, shape) for mask in self.masks], laser_ranges = self.laser_ranges, boxes = self.boxes)

    def transform_2d(self, transforms: list[tuple[float, float, float]]) -> 'CellVideo':
        """
        transforms: (shift0, shift1, rotation)
        """
        new_frames = [transform_2d(frame, *transform) for (frame, transform) in zip(self.frames, transforms)]
        new_masks = [transform_2d(mask.astype(np.uint8), *transform) > 0 for (mask, transform) in zip(self.masks, transforms)]
        return CellVideo(self.id, new_frames, self.boxes, new_masks, self.laser_ranges)

    def calc_max_mask_diff(self) -> int:
        diffs = []
        for i in range(len(self.masks)-1):
            m1, m2 = self.masks[i], self.masks[i+1]
            size0 = max((m1.shape[0], m2.shape[0]))
            size1 = max((m1.shape[1], m2.shape[1]))
            arr1 = np.zeros((size0, size1),np.bool_)
            arr1[:m1.shape[0], :m1.shape[1]] = m1
            arr2 = np.zeros((size0, size1),np.bool_)
            arr2[:m2.shape[0], :m2.shape[1]] = m2
            diffs.append(np.sum(arr1^arr2))
        return max(diffs)
    def calc_min_corr(self, blur=None) -> float:
        corrs = []
        if blur is None:
            blur = BilateralFilter(9, 75, 75)
        for i in range(len(self.masks)-1):
            m1, m2 = blur(self.frames[i]), blur(self.frames[i+1])
            size0 = max((m1.shape[0], m2.shape[0]))
            size1 = max((m1.shape[1], m2.shape[1]))
            arr1 = np.zeros((size0, size1),np.bool_)
            arr1[:m1.shape[0], :m1.shape[1]] = m1
            arr2 = np.zeros((size0, size1),np.bool_)
            arr2[:m2.shape[0], :m2.shape[1]] = m2
            corrs.append(np.corrcoef(arr1.flatten(), arr2.flatten()))
        return min(corrs)

    def max_element_count(self) -> int:
        return max(mask.size for mask in self.masks)
    def frame_u8(self, i: int):
        return (self.frames[i]/(2.**16-1)*255).astype(np.uint8)
    def frames_u8(self):
        return [self.frame_u8(i) for i in range(len(self.frames))]

    def max_relative_mask_diff(self) -> float:
        return self.calc_max_mask_diff() / self.max_element_count()
    def frame_count(self) -> int:
        return len(self.frames)

    def play_video(self):
        frames = self.frames
        play_nb_video(self.frames)

    def __post_init__(self):
        self.laser_ranges = [range for range in self.laser_ranges if self.boxes[0].intersects_y_range(*range)]
        assert len(self.laser_ranges) != 0, "ROI ranges modified after evaluate_batch!"

    def expanded_ranges(self, modifier):
        res = []
        for range in self.laser_ranges:
            diff = (range[1] - range[0])*(modifier-1)/2
            res.append((range[0]-diff, range[1]+diff))
        return res
    def video_with_ranges(self, range_modifier=1):
        ranges = self.expanded_ranges(range_modifier)
        copied_frames = [frame.copy() for frame in self.frames]
        for range in ranges:
            range = self.boxes[0].range_to_relative(range)
            for frame in copied_frames:
                frame[[max([int(range[0]), 0]),min([int(range[1]), frame.shape[0]-1])],:] = 0
        return copied_frames

    def play_video_with_ranges(self, range_modifier=1):
        frames = self.video_with_ranges(range_modifier=range_modifier)
        play_nb_video(frames)

    def calculate_brightness(self, range_modifier=1):
        ranges = self.expanded_ranges(range_modifier)
        #assert len(ranges) == 1
        if len(ranges) != 1:
            pass
            #print(f"Multiple ranges {ranges}")
        range = self.boxes[0].range_to_relative(ranges[0])
        start, end = (int(range[0]), int(range[1]))
        brighnesses = []
        for mask, frame in zip(self.masks, self.frames):
            lasered = np.sum(frame[start:end,:]) / np.sum(mask[start:end,:])
            normal = (np.sum(frame[:start,:]) + np.sum(frame[end:,:])) / (np.sum(mask[:start,:]) + np.sum(mask[end:,:]))
            brighnesses.append((normal, lasered))
        return np.array(brighnesses)
    def plot_brightness(self, range_modifier=1):
        plt.plot(eval3.cell_video(1).calculate_brightness(range_modifier=range_modifier))
        plt.show()
    def plot_brightness_diff(self, range_modifier=1):
        diff = self.calculate_brightness_diff(range_modifier=range_modifier)
        plt.plot(diff)
        plt.show()
    def calculate_brightness_diff(self, range_modifier=1):
        vals = self.calculate_brightness(range_modifier=range_modifier)
        diff = vals[:,1] - vals[:,0]
        return diff
    def calculate_brightness_ratio(self, range_modifier=1):
        vals = self.calculate_brightness(range_modifier=range_modifier)
        diff = vals[:,1]/vals[:,0]
        return diff
    def calculate_brightness_score(self, score: ScoreType, range_modifier=1):
        if score == ScoreType.Diff:
            return self.calculate_brightness_diff(range_modifier=range_modifier)
        elif score == ScoreType.Ratio:
            return self.calculate_brightness_ratio(range_modifier=range_modifier)
        else:
            raise AssertionError
        
    def plot_brightness_ratio(self, range_modifier=1):
        diff = self.calculate_brightness_ratio(range_modifier=range_modifier)
        plt.plot(diff)
        plt.show()
    def max_size(self) -> tuple[int, int]:
        return (max(frame.shape[0] for frame in self.frames), max(frame.shape[1] for frame in self.frames))


def combine_cell_videos(videos: list[CellVideo], output_width = 1024, extra_draw=None, cell_filter=lambda x: True):
    sizes = [vid.max_size() for vid in videos]
    y = 0
    x = 0
    next_y = 0
    frame_count = max(vid.frame_count() for vid in videos)
    positions = []
    y_sizes = [0]
    y_inds = [0]
    for (x_size, y_size) in sizes:
        assert output_width > x_size
        if x + x_size < output_width:
            positions.append((x, y))
            x += x_size
            next_y = max((next_y, y + y_size))
            y_sizes[-1] = max((y_sizes[-1], y_size))
        else:
            y = next_y
            x = x_size
            next_y = next_y + y_size
            positions.append((0, y))
            y_sizes.append(y_size)
        y_inds.append(len(y_sizes)-1)
    output_height = next_y
    frames = np.zeros((frame_count, output_width, output_height), np.uint16)
    for i in range(len(frames)):
        for (vid, pos, y_ind) in zip(videos, positions, y_inds):
            y_size = y_sizes[y_ind]
            if i >= vid.frame_count():
                continue
            vid_frame = vid.frames[i]
            frames[i, pos[0]:pos[0]+vid_frame.shape[0], pos[1]:pos[1]+vid_frame.shape[1]] = vid_frame
            if extra_draw is not None:
                extra_draw(frames[i, pos[0]:pos[0]+vid_frame.shape[0], pos[1]:pos[1]+y_size], vid)
    return frames

def text_to_ndarray(text, array_size, dtype, font_scale=1, thickness=2, font_color=None):
    """
    Generate a grayscale ndarray with specified text.

    Parameters:
        text (str): The text to be added to the ndarray.
        array_size (tuple): Size of the array (height, width).
        font_scale (int, optional): Scale of the font. Defaults to 1.
        thickness (int, optional): Thickness of the font. Defaults to 2.

    Returns:
        ndarray: Grayscale image array with text.
    """
    # Step 1: Create a blank grayscale image with the specified size
    blank_image = np.zeros((array_size[0], array_size[1]), dtype=dtype)
    
    # Step 2: Define font and position for text
    font = cv2.FONT_HERSHEY_SIMPLEX
    if font_color is None:
        font_color = np.iinfo(dtype).max  # White color in grayscale

    # Calculate text size to center the text
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (blank_image.shape[1] - text_size[0]) // 2
    text_y = (blank_image.shape[0] + text_size[1]) // 2
    position = (text_x, text_y)

    # Step 3: Add text to the blank image
    cv2.putText(blank_image, text, position, font, font_scale, font_color, thickness)
    
    return blank_image

def add_title_to_video(array, title: str, font_scale=1, thickness=2, height=30) -> np.ndarray:
    array_size = (height, array.shape[2])
    text = text_to_ndarray(title, array_size, array.dtype, font_scale=font_scale, thickness=thickness, font_color=int(np.max(array)))
    return np.stack([np.concatenate((text, frame), axis=0) for frame in array], axis=0)

def border_draw_mask_diff(frame, vid):
    if vid.max_relative_mask_diff() > 0.3:
        draw_border(frame, 3, intensity=np.max(frame))

def draw_border(array, border_width, intensity=255):
    """
    Draw a border with configurable width around a 2D ndarray.
    
    Parameters:
        array (ndarray): The 2D array where the border will be drawn.
        border_width (int): The width of the border.
        intensity (int): The intensity (brightness) of the border. Default is 255 (white).
        
    Returns:
        ndarray: The array with the border drawn.
    """
    # Ensure the border width is valid
    rows, cols = array.shape
    if border_width * 2 > rows or border_width * 2 > cols:
        raise ValueError("Border width is too large for the given array dimensions.")

    # Set the top and bottom border
    array[:border_width, :] = intensity           # Top border
    array[-border_width:, :] = intensity          # Bottom border

    # Set the left and right border
    array[:, :border_width] = intensity           # Left border
    array[:, -border_width:] = intensity          # Right border

    return array

def combine_batch_videos(batch: ExperimentBatch, extra_draw=None, cell_filter=lambda x: True):
    experiments = batch.experiments()
    exp_videos = []
    for exp in experiments:
        print(exp.name)
        cell_videos = ExperimentEvaluator.load_experiment(exp).cell_videos()
        cell_videos = [video for video in cell_videos if cell_filter(video)]
        vid = combine_cell_videos(ExperimentEvaluator.load_experiment(exp).cell_videos(), extra_draw=extra_draw, cell_filter=cell_filter)
        titled_vid = add_title_to_video(vid, exp.in_batch_id(), height=100)
        exp_videos.append(titled_vid)
    #exp_videos = [combine_cell_videos(ExperimentEvaluator.load_experiment(exp).cell_videos()) for exp in experiments[:2]]
    vid = np.concatenate(exp_videos, axis=2)
    #play_nb_video(vid)
    return vid

def calc_brightness_score(batch: ExperimentBatch, score_type: ScoreType = ScoreType.Ratio, range_modifier=4, cell_filter=lambda x: True, experiment_type = None, print_progress=False):
    if experiment_type == ExperimentType.WT:
        experiments = batch.wt_experiments()
    elif experiment_type == ExperimentType.KO:
        experiments = batch.ko_experiments()
    else:
        experiments = batch.experiments()
    return calculate_brightness_score_for_experiments(
            experiments,
            score_type=score_type,
            range_modifier=range_modifier,
            cell_filter=cell_filter,
            print_progress=print_progress,
        )
def calculate_brightness_score_for_experiments(experiments: list[Experiment], score_type: ScoreType = ScoreType.Ratio, range_modifier=4, cell_filter=lambda x: True, print_progress=False):
    all_scores = []
    all_diffs = []
    for experiment in experiments:
        if print_progress:
            print(experiment.name)
        eval = ExperimentEvaluator.load_experiment(experiment)
        diffs = []
        for cell_video in eval.cell_videos():
            if not cell_filter(cell_video):
                continue
            diffs.append(cell_video.calculate_brightness_score(score_type, range_modifier=range_modifier))
        all_diffs.append(diffs)
    return all_diffs

def calculate_brightness_score_for_videos(videos: list[CellVideo], score_type: ScoreType = ScoreType.Ratio, range_modifier=4, cell_filter=lambda x: True, print_progress=False):
    diffs = []
    vid_ids = []
    for vid in videos:
        if cell_filter(vid):
            vid_ids.append(vid.id)
            diffs.append(vid.calculate_brightness_score(score_type, range_modifier=range_modifier))
    return vid_ids, diffs

def score_avg(diffb):
    for diffa in diffb:
        try:
            np.any(np.isnan(diffa))
        except:
            print(f"Diffa {diffa}")
            print(f"Diffb {diffb}")
            raise
    diffs = np.vstack([diffa for diffa in diffb if not np.any(np.isnan(diffa))])
    mean = np.mean(diffs, axis=0)
    std = np.std(diffs, ddof=1, axis=0)
    se = std/diffs.shape[0]**0.5
    return (mean, se)
def plot_score_avg(diffs, show = True, color='blue', label=None):
    mean, sem = score_avg(diffs)
    x = range(len(mean))
    plt.plot(x, mean, label=label, color=color)
    plt.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.3)
    plt.legend()
    if show:
        plt.show()

def plot_brightness_score(batch: ExperimentBatch, score_type: ScoreType = ScoreType.Ratio, range_modifier=4, cell_filter=lambda x: True, color='blue'):
    diffs = calc_brightness_score(batch, score_type = score_type, range_modifier=range_modifier, cell_filter=cell_filter)
    flattened = [diff for exp_diff in diffs for diff in exp_diff]
    plt.figure()
    plot_score_avg(flattened, color=color)
    
def plot_brightness_score_double(ctrl: ExperimentBatch, cnnd: ExperimentBatch, score_type: ScoreType = ScoreType.Ratio, range_modifier=4, cell_filter=lambda x: True):
    ctrl_diffs = calc_brightness_score(ctrl, score_type = score_type, range_modifier=range_modifier, cell_filter=cell_filter)
    cnnd_diffs = calc_brightness_score(cnnd, score_type = score_type, range_modifier=range_modifier, cell_filter=cell_filter)
    ctrl_flattened = [diff for exp_diff in ctrl_diffs for diff in exp_diff]
    cnnd_flattened = [diff for exp_diff in cnnd_diffs for diff in exp_diff]
    plot_score_double(ctrl_flattened, ctrl.name, cnnd_flattened, cnnd.name)

def plot_brightness_scores(batches: list[ExperimentBatch], score_type: ScoreType = ScoreType.Ratio, range_modifier=4, cell_filter=lambda x: True):
    scores_names = []
    for batch in batches:
        diffs = calc_brightness_score(batch, score_type = score_type, range_modifier=range_modifier, cell_filter=cell_filter)
        flattened = [diff for exp_diff in diffs for diff in exp_diff]
        scores_names.append((flattened, batch.name))
    plot_scores_avg(scores_names)


def plot_brightness_score_wt_ko(batch: ExperimentBatch, score_type: ScoreType = ScoreType.Ratio, range_modifier=4, cell_filter=lambda x: True, print_progress=False):
    wt_diffs = calc_brightness_score(batch, score_type = score_type, range_modifier=range_modifier, cell_filter=cell_filter, experiment_type = ExperimentType.WT, print_progress=print_progress)
    ko_diffs = calc_brightness_score(batch, score_type = score_type, range_modifier=range_modifier, cell_filter=cell_filter, experiment_type = ExperimentType.KO, print_progress=print_progress)
    wt_flattened = [diff for exp_diff in wt_diffs for diff in exp_diff]
    ko_flattened = [diff for exp_diff in ko_diffs for diff in exp_diff]
    plot_score_double(wt_flattened, "wt", ko_flattened, "ko")

def plot_score_double(score1, score1_name, score2, score2_name):
    plot_scores_avg([(score1, score1_name), (score2, score2_name)])
    #plot_score_avg(score1, label = score1_name, show=False)
    #plot_score_avg(score2, label = score2_name, color = "red")

def plot_scores_avg(scores_names):
    colors = ['red','blue','green','purple','yellow','black']
    for i, (score, name) in enumerate(scores_names):
        plot_score_avg(score, label = name, show=i+1 == len(scores_names), color=colors[i])

def plot_brightness_score_series(batch: ExperimentBatch, score_type: ScoreType = ScoreType.Ratio, range_modifier=4, cell_filter=lambda x: True):
    # what do I want to plot?
    # the distribution of the score over time
    diffs = calc_brightness_score(batch, score_type = score_type, range_modifier=range_modifier, cell_filter=cell_filter)
    flattened = [diff for exp_diff in diffs for diff in exp_diff]
    plotly_time_series_distribution_3d(flattened)

import plotly.graph_objs as go

def plotly_time_series_distribution_3d(
    list_of_series,
    num_bins=20,
    title='Distribution of Scores Over Time',
    xaxis_title='Time Step',
    yaxis_title='Score Value',
    zaxis_title='Density',
    colorscale='Viridis',
    density=True
):
    """
    Creates an interactive 3D surface plot using Plotly showing how the distribution
    of values in time series evolves over time.

    Parameters:
    - list_of_series: list of lists or 2D NumPy array
    - num_bins: number of histogram bins
    - title: plot title
    - xaxis_title, yaxis_title, zaxis_title: axis labels
    - colorscale: Plotly color scale name
    - density: whether to normalize histogram (density=True) or show raw counts
    """

    data = np.array(list_of_series)
    data = data[~np.isnan(data).any(axis=1)]  # Remove NaN-containing series

    if data.size == 0:
        raise ValueError("No valid time series to plot after removing NaN values.")

    time_steps = data.shape[1]
    min_val, max_val = np.min(data), np.max(data)
    bins = np.linspace(min_val, max_val, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Build histogram matrix Z
    Z = []
    for t in range(time_steps):
        values_at_t = data[:, t]
        hist, _ = np.histogram(values_at_t, bins=bins, density=density)
        Z.append(hist)
    Z = np.array(Z).T  # shape: (bins, time_steps)

    # Create surface plot
    surface = go.Surface(
        z=Z,
        x=np.arange(time_steps),  # time steps
        y=bin_centers,            # bin centers (score values)
        colorscale=colorscale
    )

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title=xaxis_title),
            yaxis=dict(title=yaxis_title),
            zaxis=dict(title=zaxis_title),
        ),
        autosize=False,
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig = go.Figure(data=[surface], layout=layout)
    fig.show()

def plot_all_scores(diffs, outliers=None, max_per_fig = 40, sort_by=lambda y: -1000000 if np.any(np.isnan(y)) else np.mean(y)):
    flattened = [diff for exp_diff in diffs for diff in exp_diff]
    return plot_sorted_3d_lines(flattened, max_per_fig=max_per_fig)

def plot_all_vid_scores(scores, ids, outliers=None, max_per_fig = 40, sort_by=lambda y: -1000000 if np.any(np.isnan(y)) else np.mean(y)):
    scores = scores[:]
    return plot_sorted_3d_lines(scores, max_per_fig=max_per_fig, sort_by=sort_by, ids=ids)

def plot_sorted_3d_lines(lines, max_per_fig=None, sort_by=lambda y: -1000000 if np.any(np.isnan(y)) else np.mean(y), ids = None):
    from math import ceil
    """
    Plot 1‑D lines (Y only) in 3‑D, sorted by mean.
    
    Parameters
    ----------
    lines : list[array‑like]
        Each element is a 1‑D sequence of Y values.
    max_per_fig : int or None, optional
        Maximum number of lines per figure.  If None, all lines go in one.
    """
    # 1. Sort by mean
    if ids is not None:
        assert len(ids) == len(lines)
        idd_lines = list(zip(lines, ids))
        sorted_lines, sorted_ids = list(zip(*sorted(idd_lines, key= lambda k: sort_by(k[0]))))
    else:
        sorted_lines = sorted(lines, key=sort_by)
        sorted_ids = None
    n_lines = len(sorted_lines)

    # 2. Figure batching
    if max_per_fig is None or max_per_fig <= 0:
        max_per_fig = n_lines
    n_figs = ceil(n_lines / max_per_fig)
    max_per_fig = ceil(n_lines / n_figs)

    for fig_idx in range(n_figs):
        start = fig_idx * max_per_fig
        end   = min(start + max_per_fig, n_lines)
        batch = sorted_lines[start:end]

        fig = go.Figure()
        for local_idx, y in enumerate(batch):
            x = np.arange(len(y))
            z = np.full_like(x, local_idx+start)           # stack within the batch
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                name=f'Line {start+local_idx}',
                line=dict(width=4)
            ))

        if sorted_ids is not None:
            zaxis = dict(
                tickvals=list(range(start, end)),
                ticktext=[sorted_ids[i] for i in range(start, end)]
            )
        else:
            zaxis=None
        fig.update_layout(
            title=f'3‑D Lines {start}–{end-1} (sorted by mean)',
            scene=dict(
                xaxis_title='Index',
                yaxis_title='Y',
                zaxis_title='Line rank within figure',
                zaxis=zaxis
            ),
            width=800, height=600,
            showlegend=False
        )
        fig.show()


#import numpy as np
import imageio.v3 as iio
import base64, io, uuid
from IPython.display import display, HTML
import ipywidgets as widgets
import tempfile
import os

def encode_mp4_temp(frames, fps=24, plugin="FFMPEG"):
    """
    Encode uint8 RGB frames → MP4 bytes using a real temp file
    (needed for plugins such as ffmpeg that can’t handle BytesIO).

    Returns
    -------
    bytes  : the MP4 file’s contents
    """
    # create a temporary file that survives until we manually delete
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_name = tmp.name          # path to pass into imageio
    try:
        # write video to that path
        iio.imwrite(tmp_name, frames, plugin=plugin, fps=fps)

        # read the finished file back into memory
        with open(tmp_name, "rb") as f:
            mp4_bytes = f.read()
    finally:
        os.remove(tmp_name)          # clean up the temp file

    return mp4_bytes

def video_widget(frames, fps=20, max_width='800px'):
    """
    Interactive HTML5 video player with:
      • Scroll‑pan & Ctrl+wheel zoom
      • Playback‑rate slider (0.1×–10×)
    
    Parameters
    ----------
    frames : iterable of uint8 RGB frames, shape (H,W,3)
    fps    : int, encoding frame‑rate
    max_width : CSS width for the video container
    """
    # --- Encode frames to MP4 in memory ---
    b64 = base64.b64encode(encode_mp4_temp(frames, fps=fps)).decode()
    
    # Unique IDs so multiple widgets coexist
    vid_id  = f"v{uuid.uuid4().hex}"
    slid_id = f"s{uuid.uuid4().hex}"
    
    # --- HTML video, wrapped in a scrollable div for panning ---
    video_html = f"""
    <div style="overflow:auto; border:1px solid #ccc; width:{max_width};">
      <video id="{vid_id}" src="data:video/mp4;base64,{b64}" 
             controls style="width:100%; display:block;"></video>
    </div>
    """
    
    # --- Playback‑rate slider (log scale 0.1–10) ---
    slider = widgets.FloatLogSlider(value=1, base=10, min=-1, max=1,
                                    description='Speed ×', readout_format='.2f')
    slider.layout.width = max_width
    slider.add_class(slid_id)
    
    # --- JS: link slider → playbackRate, add Ctrl‑wheel zoom ---
    js = f"""
    <script>
    (function() {{
        const video  = document.getElementById("{vid_id}");
        const slider = document.querySelector(".{slid_id} input");
        
        // Sync slider → video
        const updateRate = () => {{ video.playbackRate = parseFloat(slider.value); }};
        slider.addEventListener('input', updateRate);
        updateRate();  // initial
        
        // Ctrl+wheel zoom
        let scale = 1;
        video.addEventListener('wheel', e => {{
            if(!e.ctrlKey) return;
            e.preventDefault();
            scale *= (e.deltaY < 0 ? 1.1 : 0.9);
            scale = Math.max(0.2, Math.min(scale, 10));
            video.style.transformOrigin = '0 0';
            video.style.transform = `scale(${{scale}})`;
        }}, {{passive:false}});
    }})();
    </script>
    """
    
    display(widgets.VBox([widgets.HTML(video_html + js), slider]))
