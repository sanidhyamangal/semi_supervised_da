"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import argparse  # for argument parsing
import os
from pathlib import Path
from typing import Tuple  # for os related ops

from PIL import Image

from logger import logger


def create_data_folders(output_path:str) -> None:
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    for i in [90, 180,270, 360]:
        if not os.path.exists(os.path.join(output_path, str(i))):
            os.mkdir(os.path.join(output_path, str(i)))


def process_rotation(image:Image, image_name:str ,output_path:str) -> None:
    for i in [90,180,270,360]:
        new_image = image.rotate(i)

        new_image.save(f"{output_path}/{i}/{image_name}")


def perform_rotation(input_path:str, output_path:str, resize_shape:Tuple[int]=(244,244)) -> None:
    img_path = Path(input_path)
    all_images_path = [
            str(path) for path in img_path.glob("*/*")
        ]
    
    create_data_folders(output_path)

    for images in all_images_path:
        image = Image.open(images).resize(resize_shape)

        logger.info("Performing Rotation on {}".format(images))
        process_rotation(image, image_name=images.split("/")[-1], output_path=output_path)







# ?perform_rotation("data", "rotation_data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Script to create a rotation dataset for training rotation net")


    parser.add_argument('--input_path',
                              help='Input path for loading the images from input',
                              dest="input_path", required=True)
    parser.add_argument('--output_path',
                              help='Output path for saving the images from input',required=True,
                              dest="output_path")
    parser.add_argument('--image_size',
                              default="244,244",
                              help='resize image shape',
                              dest="resize_shape")

    
    args = parser.parse_args()

    image_size = list(map(lambda x: int(x), args.resize_shape.split(",")))

    if len(image_size) != 2:
        raise argparse.ArgumentError("image_size can only be of shape 2")
    
    perform_rotation(input_path=args.input_path, output_path=args.output_path, resize_shape=image_size)
