# DuckSeg
DuckSeg: Deep-learning Utility for Cell masKing and Segmentation

<img width="1536" height="1024" alt="DuckSeg_logo_goodtext_nobg_4" src="https://github.com/user-attachments/assets/c976cce1-d3c2-4bc3-bc5b-70451415ea06" />


# Installation
Save current repository to your system and unzip folder
```
unzip cell-segmentation-master.zip
```

Create a new virtual environment using
```
python -m venv myenv
```

activate it
```
source ~/myenv/bin/activate
```

then install the required packages
```
python -m pip install -r requirements.txt
```

Install [CellSAM](https://github.com/vanvalenlab/cellSAM/) project via *pip*!
```
pip install git+https://github.com/vanvalenlab/cellSAM.git
```

start jupyter lab using
```
python -m jupyter lab
```

The `notebooks` directory contains the example notebooks. Start with `transforms_POLE1.ipynb` to evaluate an experiment

# Loading experiment data
Specify the path to the folder containing all the videos:

- NAME_00123.nd2 files
- NAME_00123_ROI.tiff files
- *Set the name variable to NAME (e.g. MSH2)*

Add the grouping to the file names:
- WT or KO
- e.g. NAME_WT_00123.tiff

# Parameters
1. ```experiment_filter```
- Which experiments from the bathc are processed
    - ```None``` - all experiments
    - List of number ```[0,2,4]``` - only these
    - String ```'KO'``` - only those with *KO* label
2. ```frame_filter```
- Which frames to process from each experiment
    - integer - split to 1-2-3 equal parts the video
    - list of integers - show the specified frames *e.g.* ```[0,10]``` first and nineth frame 
3. ```show_after_transform```
- List of integers, show the processed frames after applying these tranforms.

# Transforms
?????


# Usage

1. Optimize the masking with the utilized transforms. Goal to optimize the masks for most of the cells correctly
2. Run the training analysis ```evalute_batch(batch, transforms=transforms)```
3. Filter outliers:
- Filter that are not correctly identified (the fluroesncence is not between the lines)
- Change ```range_modifier``` parameter to increase the distance between lines [pixels]
- Manula outliers - Select outliers by hand
- Extra outliers - Select outliers by filters
  - area_between -  10 pixel - 5000 pixel a mask méret
  - max_area_change_between_frames(0.2) - ket frame között max területváltozás (%)
  - max_total_area_change(0.2) - legnagyobb legkisebb között (%)
  - id_starts_with('REV') - ID alapján, modnjuk fájlnév alapján
We can add new filters in the ```outlier_filters.py``` file.
4. Visualize the results

# Output
```plot_brightness_score_wt_ko``` function to compare the brightness intensities between groups, with two modes:
1. ScoreType = Ratio
2. ScoreType = Diff

Output plot:
- x-axis - frames in video
- y-axis - difference or ratio

<img width="450" alt="image" src="https://github.com/user-attachments/assets/a091b8a0-e625-424c-b50e-5909a60b3992" />