import ipywidgets as widgets
from IPython.display import display
import dataclasses
from PIL import Image
from io import BytesIO
import numpy as np


@dataclasses.dataclass
class Video:
    id: str
    frames: list[bytes] # png
    size: tuple[int,int] # video size

def show_video_grid_jupyter(video_list, outliers=None, manual_outliers=None, container_width=1200):
    """
    Show a grid of videos with independent sliders in Jupyter.
    Uses Image widgets instead of matplotlib with clear_output.
    Includes global controls to synchronize all videos and control image sizes.
    
    Args:
        video_list: List of Video dataclass objects
        outliers: Set of video IDs that should be highlighted with a border
        container_width: Total width available for the grid (default 1200px)
        
    Returns:
        update_outliers: Function to update which videos are highlighted
    """
    if manual_outliers is None:
        manual_outliers = set()
    if outliers is None:
        outliers = set()
    outliers.update(manual_outliers)
    video_list = video_list[:]
    video_list.sort(key = lambda vid: (vid.size[0]/vid.size[1], vid.id))
    
    n_videos = len(video_list)
    
    
    # Find the maximum number of frames across all videos
    max_frames = max(len(video.frames) for video in video_list)
    
    # Size control slider
    size_slider = widgets.IntSlider(
        value=100, min=25, max=500, step=25,
        description='Image Size:', style={'description_width': 'initial'}
    )
    
    # Show outliers checkbox
    show_outliers_checkbox = widgets.Checkbox(
        value=True,
        description='Show outliers',
        style={'description_width': 'initial'}
    )
    
    # Calculate grid dimensions based on image size
    def calculate_grid_dimensions(image_size):
        margin = 20  # Space between images
        effective_width = image_size + margin
        n_cols = max(1, container_width // effective_width)
        n_rows = (n_videos + n_cols - 1) // n_cols
        return n_cols, n_rows
    
    # Global controls
    global_play = widgets.Play(value=0, min=0, max=max_frames-1, step=1, interval=100)
    global_slider = widgets.IntSlider(value=0, min=0, max=max_frames-1, step=1, description='Global:')
    
    # Link global play and slider
    widgets.jslink((global_play, 'value'), (global_slider, 'value'))
    
    # Container for the video grid (will be updated when size changes)
    video_grid_container = widgets.VBox()
    
    players = []
    individual_sliders = []
    individual_plays = []
    img_widgets = []
    video_containers = []  # Store containers for border styling
    checkboxes = []
    
    def get_border_style(video_id, is_outlier):
        """Get CSS border style based on outlier status"""
        if is_outlier:
            return "3px solid red"
        else:
            return "1px solid #ddd"
    
    def create_video_players(image_size):
        """Create all video player widgets with specified image size"""
        nonlocal players, individual_sliders, individual_plays, img_widgets, video_containers, checkboxes
        
        players = []
        individual_sliders = []
        individual_plays = []
        img_widgets = []
        video_containers = []
        checkboxes = []
        
        for i, video in enumerate(video_list):
            max_frame = len(video.frames) - 1
            
            # Create individual slider and play control
            play = widgets.Play(value=0, min=0, max=max_frame, step=1, interval=100)
            slider = widgets.IntSlider(value=0, min=0, max=max_frame, step=1, description=f'{video.id}:')
            widgets.jslink((play, 'value'), (slider, 'value'))
            
            individual_sliders.append(slider)
            individual_plays.append(play)
            
            # Create Image widget with dynamic size
            img_widget = widgets.Image(
                value=video.frames[0],
                format='png',
                width=image_size,
                height=int(image_size * 0.75)  # Maintain aspect ratio
            )
            img_widgets.append(img_widget)
            
            # Create a container with border styling and video ID label
            is_outlier = video.id in outliers
            
            # Video ID label
            id_label = widgets.HTML(
                value=f"<div style='text-align: center; font-weight: bold; padding: 2px; background-color: #f0f0f0;'>{video.id}</div>"
            )

            checkbox = widgets.Checkbox(description="", indent=False, value=video.id in manual_outliers);
            checkboxes.append(checkbox)
            
            # Container with border
            video_container = widgets.VBox(
                [widgets.Box([checkbox, id_label]), widgets.HBox([play, slider]), img_widget],
                layout=widgets.Layout(
                    border=get_border_style(video.id, is_outlier),
                    padding='5px',
                    margin='2px'
                )
            )
            video_containers.append(video_container)
            
            def make_update_function(video, img_widget):
                def update_frame(change):
                    frame_idx = change['new']
                    # Handle case where global slider exceeds individual video length
                    if frame_idx < len(video.frames):
                        frame = video.frames[frame_idx]
                        img_widget.value = frame
                return update_frame
            
            slider.observe(make_update_function(video, img_widget), names='value')
            
            # Trigger initial display
            slider.value = 0
            
            players.append(video_container)
    
    def update_grid_layout():
        """Update the grid layout based on current image size and visibility settings"""
        image_size = size_slider.value
        show_outliers = show_outliers_checkbox.value
        
        # Filter visible players based on outlier visibility setting
        visible_players = []
        for i, video in enumerate(video_list):
            is_outlier = video.id in outliers
            if show_outliers or not is_outlier:
                visible_players.append(players[i])
        
        # Calculate grid dimensions based on visible players
        n_visible = len(visible_players)
        if n_visible == 0:
            video_grid_container.children = [widgets.HTML("<p>No videos to display</p>")]
            return
            
        margin = 20  # Space between images
        effective_width = image_size + margin
        n_cols = max(1, container_width // effective_width)
        n_rows = (n_visible + n_cols - 1) // n_cols
        
        # Update image sizes for all widgets (even hidden ones)
        for img_widget in img_widgets:
            img_widget.width = image_size
            img_widget.height = int(image_size * 0.75)
        
        # Arrange visible video grid
        grid = []
        for r in range(n_rows):
            row = []
            for c in range(n_cols):
                idx = r * n_cols + c
                if idx < len(visible_players):
                    row.append(visible_players[idx])
            if row:  # Only add non-empty rows
                grid.append(widgets.HBox(row))

        # Update the container
        video_grid_container.children = grid
    
    def update_outliers_function(new_outliers, manuals):
        """Update which videos are highlighted as outliers"""
        nonlocal outliers, checkboxes, manual_outliers
        outliers = set(new_outliers) if new_outliers else set()
        manual_outliers.update(manuals)
        outliers.update(manual_outliers)
        
        # Update border styles for all video containers
        for i, video in enumerate(video_list):
            if i < len(video_containers):
                is_outlier = video.id in outliers
                video_containers[i].layout.border = get_border_style(video.id, is_outlier)
                is_manual_outlier = video.id in manual_outliers
                checkboxes[i].value = is_manual_outlier
        
        # Update grid layout to handle visibility changes
        update_grid_layout()
    
    # Create initial players
    create_video_players(size_slider.value)
    
    # Size slider callback
    def on_size_change(change):
        update_grid_layout()
    
    size_slider.observe(on_size_change, names='value')
    
    # Global control functions
    def sync_all_to_global(change):
        """Sync all individual sliders to global slider"""
        global_value = change['new']
        for slider in individual_sliders:
            # Only update if the value is within the video's range
            if global_value <= slider.max:
                slider.value = global_value
    
    # Connect global slider to individual sliders
    global_slider.observe(sync_all_to_global, names='value')
    
    def update_checkbox(checkbox, video_id, video):
        nonlocal manual_outliers, outliers
        checkbox = checkbox['owner']
        if checkbox.value:
            manual_outliers.add(video_id)
            outliers.add(video_id)
        else:
            manual_outliers.discard(video_id)
            outliers.discard(video_id)
        video.layout.border = get_border_style(video_id, checkbox.value)

    
    # Connect buttons
    show_outliers_checkbox.observe(lambda c: update_grid_layout(), names='value')
    for checkbox, video, vid_player in zip(checkboxes, video_list, video_containers):
        checkbox.observe(lambda c, video_id=video.id, vid_player=vid_player: update_checkbox(c, video_id, vid_player), names='value')
    
    # Initial grid layout
    update_grid_layout()
    
    # Arrange global controls
    global_controls = widgets.VBox([
        widgets.HTML("<b>Global Controls:</b>"),
        widgets.HBox([global_play, global_slider]),
        show_outliers_checkbox,
        size_slider
    ])
    
    # Combine everything
    full_widget = widgets.VBox([global_controls, video_grid_container])
    display(full_widget)
    
    # Return the update function for external use
    return update_outliers_function

def grayscale_to_green_png_bytes(img: np.ndarray, norm=None) -> list:
    # Normalize image to 0–255
    if norm is None:
        norm = (img_norm.min(), img_norm.max())
    img_norm = img.astype(np.float32)
    img_norm -= norm[0]
    img_norm /= norm[1]
    img_norm = (img_norm * 255).astype(np.uint8)

    # Create green image (height, width, 3), green channel populated
    green_image = np.zeros((img_norm.shape[0], img_norm.shape[1], 3), dtype=np.uint8)
    green_image[..., 1] = img_norm

    # Convert to PIL image
    pil_image = Image.fromarray(green_image, mode='RGB')

    # Save to buffer as PNG
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)

    # Return list of bytes
    return buffer.getvalue()

def min_max(vids):
    min_val = min(frame.min() for vid in vids for frame in vid.frames)
    max_val = max(frame.max() for vid in vids for frame in vid.frames)
    return min_val, max_val

def normalize_videos_for_display(vids: list['CellVideo'], range_modifier=4) -> list[Video]:
    """
    Makes sure frame sizes match and converts images from greyscale to greenscale
    """
    vids = [vid.homogenize_size() for vid in vids]
    norms = [min_max([vid]) for vid in vids]
    vids2 = [Video(id=vid.id,frames=[grayscale_to_green_png_bytes(frame, norm=norm) for frame in vid.video_with_ranges(range_modifier=4)], size=vid.frames[0].shape) for vid, norm in zip(vids, norms)]
    return vids2
