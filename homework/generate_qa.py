import json
from pathlib import Path
import torch

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from math import inf

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def _get_relative_cart_data(info_path: str = None, img_width: int = 150, img_height: int = 100, view_index = None) -> list:
    """

    """
    with open(info_path) as ipf:
        data = json.load(ipf)
        width_factor = img_width / ORIGINAL_WIDTH
        height_factor = img_height / ORIGINAL_HEIGHT
        ctr = [img_width / 2, img_height / 2]
        karts = {}

        #     class_id, track_id, x1, y1, x2, y2 = detection
        # class_id: object type
        # track_id: object in seq of detected objects
        kart_ctrs = {}
        for i, detection in enumerate(data['detections'][view_index]):
            try:
                if OBJECT_TYPES[detection[0]] != OBJECT_TYPES[1]:
                    continue
            except KeyError as e:
                continue

            instance_id = detection[1]
            kart_name = data['karts'][instance_id]
            if kart_name not in karts:
                 kart = {
                     'kart_name': kart_name,
                     'is_center_kart': False,
                     'instance_id': instance_id,
                     'center': [inf, inf],
                 }
                 karts[kart_name] = kart
            else:
                kart = karts[kart_name]

            if instance_id not in kart_ctrs:
                kart_ctrs[kart_name] = []

            detection_ctr = [ (abs(detection[2] - detection[4]) / 2) * width_factor,
                              (abs(detection[3] - detection[5]) / 2) * height_factor]
            kart_ctrs[kart_name].append(detection_ctr)

        # find cart centers and ego cart
        delta_min = inf

        for k in karts.keys():
            kart_ctrs_ = kart_ctrs[k]

            for ctr_ in kart_ctrs_:

                detection_delta_avg = (abs(sum(ctr) / 2) - (sum(ctr_)) / 2)

                if detection_delta_avg <= delta_min:
                    delta_min = detection_delta_avg
                    ctr_kart = k

            karts[k]['center'] = torch.tensor(kart_ctrs[k], dtype=float).mean(dim=0)

        karts[ctr_kart]['is_center_kart'] = True
        return karts.values()

def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """
    with open(info_path) as ipf:
        data = json.load(ipf)
        objs = []
        width_factor = img_width / ORIGINAL_WIDTH
        height_factor = img_height / ORIGINAL_HEIGHT
        ctr = [img_width / 2, img_height / 2]
        karts = data['karts']

        #     class_id, track_id, x1, y1, x2, y2 = detection
        # class_id: object type
        # track_id: object in seq of detected objects
        # for i, detections in enumerate(data['detections'][view_index]):
        #     kart = {
        #         'kart_name': karts[i],
        #         'is_center_kart': False,
        #         'instance_id': None,
        #         'center': [inf, inf],
        #     }
        #     delta_min = inf
        #     ctr = [inf, inf]
        #     kart_idx = -1
        #     # focus on detection[view_index]
        #     for detection in detections:
        #         # assign instance_id -> detection[1]
        #         if detection[0] != OBJECT_TYPES[1]:
        #             continue
        #         detection_ctr = [ (abs(detection[2] - detection[4]) / 2) * width_factor,
        #                           (abs(detection[3] - detection[5]) / 2) * height_factor]

        #         detection_delta_avg = (sum(ctr) / 2) - (sum(detection_ctr) / 2)
        #         if detection_delta_avg <= delta_min:
        #             ctr = detection_ctr
        #             delta_min = detection_delta_avg
        #             kart_idx = data['karts'][detection[1]]

        #     kart['center'] = ctr
        #     kart['is_center_kart'] = kart['kart_name'] == karts[kart_idx]
        #     objs.append(kart)

        #return objs

        return _get_relative_cart_data(info_path, img_width, img_height, view_index)

def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    with open(info_path) as ipf:
        return json.load(ipf)['track']


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    qa_pairs = []
    with open(info_path) as ipf:
        data = json.load(ipf)

    karts = data['karts']
    #distances_down_track = {karts[i]: v, for i, v in enumerate(data['distance_down_track'])}
    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)

    # 1. Ego car question
    q = 'What kart is the ego car?'

    ego_kart = None
    ego_kart_ctr = None

    for kart in kart_objects:
        if kart['is_center_kart']:
            ego_kart = kart['kart_name']
            ego_kart_ctr = kart['center']

    qa_pairs.append({'question': q,
                     'answer': ego_kart})

    q = 'How many karts are there in the scenario?'
    # 2. Total karts question
    qa_pairs.append({'question': q,
                     'answer': len(kart_objects)})

    # 3. Track information questions
    qa_pairs.append({'question': 'What track is this?',
                        'answer': extract_track_info(info_path)})

    for kart in kart_objects:

        # 4. Relative position questions for each kart
        q1 = 'Is {kart_name} to the left or right of the ego car?'
        q2 = 'Is {kart_name} in front of or behind the ego car?'
        q3 = 'Where is {kart_name} relative to the ego car?'

        kart_name = kart['kart_name']

        if kart['kart_name'] == ego_kart:
            continue

        x, y = kart['center']
        relative_pos = ''
        left_cars = 0
        right_cars = 0
        front_cars = 0
        behind_cars = 0
        if x > ego_kart_ctr[0]:
            qa_pairs.append({'question': q1.format(kart_name=kart_name),
                              'answer': 'left'})
            relative_pos += 'left, '
            left_cars += 1
        else:
            qa_pairs.append({'question': q1.format(kart_name=kart_name),
                              'answer': 'right'})
            relative_pos += 'right, '
            right_cars += 1
        if y > ego_kart_ctr[1]:
            qa_pairs.append({'question': q2.format(kart_name=kart_name),
                              'answer': 'behind'})
            relative_pos += 'behind'
            behind_cars += 1
        else:
            qa_pairs.append({'question': q2.format(kart_name=kart_name),
                              'answer': 'front'})
            relative_pos += 'front'
            front_cars += 1

        qa_pairs.append({'question': q3.format(kart_name=kart_name),
                              'answer': relative_pos})

        # 5. Counting questions
        qa_pairs.append({'question': 'How many karts are to the left of the ego car?',
                         'answer': left_cars})
        qa_pairs.append({'question': 'How many karts are to the right of the ego car?',
                          'answer': right_cars})
        qa_pairs.append({'question': 'How many karts are in front of the ego car?',
                          'answer': front_cars})
        qa_pairs.append({'question': 'How many karts are behind the ego car?',
                          'answer': behind_cars})

    return qa_pairs

def generate_bulk(source_dir: str = 'data/valid', dest_dir: str = 'data/train', display_images=False):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    info_files = source_dir.glob("*_info.json")

    for info_file in info_files:
        base_name = info_file.stem.replace("_info", "")
        image_files = list(info_file.parent.glob(f"{base_name}_*_im.jpg"))

        for image_file in image_files:
            frame_id, view_index = extract_frame_info(image_file)
            qa_file = dest_dir.joinpath(Path(f"{base_name}_{view_index:02d}_qa_pairs.json"))

            # Generate QA pairs
            qa_pairs = generate_qa_pairs(info_file, int(view_index))

            print(qa_pairs)
            print(qa_file)

            # Display the image
            if display_images:
                # Visualize detections
                annotated_image = draw_detections(str(image_file), info_file)

                plt.figure(figsize=(12, 8))
                plt.imshow(annotated_image)
                plt.axis("off")
                plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
                plt.show()

            with open(qa_file, 'w') as qaf:
                json.dump(qa_pairs, qaf)

        # # Print QA pairs
        # print("\nQuestion-Answer Pairs:")
        # print("-" * 50)
        # for qa in qa_pairs:
        #     print(f"Q: {qa['question']}")
        #     print(f"A: {qa['answer']}")
        #     print("-" * 50)


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_qa_pairs, "bulk": generate_bulk})


if __name__ == "__main__":
    main()
