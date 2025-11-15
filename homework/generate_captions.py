from pathlib import Path
import json
from PIL import Image, ImageDraw
from math import inf

import fire
from matplotlib import pyplot as plt

from generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_captions(info_path: str, image_file: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """

    captions = []
    with open(info_path) as ipf:
        data = json.load(ipf)

    karts = data['karts']
    #distances_down_track = {karts[i]: v, for i, v in enumerate(data['distance_down_track'])}
    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    image_file = Path(*image_file.parts[1:])

    for kart in kart_objects:
        if kart['is_center_kart']:
            ego_kart = kart['kart_name']
            ego_kart_ctr = kart['center']

    # 1. Ego car
    captions.append({'image_file': str(image_file),
                    'caption': '{kart_name} is the ego car.'.format(kart_name=ego_kart)})

    # 2. Counting
    num_karts = str(len(kart_objects))
    captions.append({'image_file': str(image_file),
                     'caption': f'There are {num_karts} karts in the scenario.'})

    # 3. Track name
    track_name = extract_track_info(info_path)
    captions.append({'image_file': str(image_file),
                        'caption': f'The track is {track_name}.'})

    # 4. Relative position

    caption = '{kart_name} is {position} of the ego car.'

    for kart in kart_objects:
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
            # captions.append({'image_file': str(image_file),
            #                  'caption': caption.format(kart_name=kart_name,
            #                                            position='left')})
            relative_pos += 'left and '
            left_cars += 1
        else:
            # captions.append({'image_file': str(image_file),
            #                  'caption': caption.format(kart_name=kart_name,
            #                                            position='right')})
            relative_pos += 'right and '
            right_cars += 1
        if y > ego_kart_ctr[1]:
            # captions.append({'image_file': str(image_file),
            #                 'caption': caption.format(kart_name=kart_name,
            #                                           position='behind')})
            relative_pos += 'behind'
            behind_cars += 1
        else:
            # captions.append({'image_file': str(image_file),
            #                  'caption': caption.format(kart_name=kart_name,
            #                                            position='front')})

            relative_pos += 'front'
            front_cars += 1

        captions.append({'image_file': str(image_file),
                        'caption': caption.format(kart_name=kart_name,
                                                  position=relative_pos)})

    return captions

def generate_bulk(source_dir: str = 'data/valid', dest_dir: str = 'data/train', display_images=False, total=150):
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
    count = 0
    for info_file in info_files:
        if count > total:
            return
        count += 1
        base_name = info_file.stem.replace("_info", "")
        image_files = list(info_file.parent.glob(f"{base_name}_*_im.jpg"))

        for image_file in image_files:
            frame_id, view_index = extract_frame_info(image_file)
            captions_file = dest_dir.joinpath(Path(f"{base_name}_{view_index:02d}_captions.json"))

            # Generate QA pairs
            captions = generate_captions(info_file, image_file, int(view_index))

            print(captions_file)

            # Display the image
            if display_images:
                print(captions)

                # Visualize detections
                annotated_image = draw_detections(str(image_file), info_file)

                plt.figure(figsize=(12, 8))
                plt.imshow(annotated_image)
                plt.axis("off")
                plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
                plt.show()

            with open(captions_file, 'w') as qaf:
                json.dump(captions, qaf)

        # # Print QA pairs
        # print("\nQuestion-Answer Pairs:")
        # print("-" * 50)
        # for qa in qa_pairs:
        #     print(f"Q: {qa['question']}")
        #     print(f"A: {qa['answer']}")
        #     print("-" * 50)



def check_caption(info_file: str, view_index: int):
    captions = generate_captions(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "bulk": generate_bulk})


if __name__ == "__main__":
    main()
