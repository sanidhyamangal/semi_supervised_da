#!/usr/bin/python3
"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import argparse  # for parsing args
import os
import random
import shutil
from pathlib import Path
from typing import List

from logger import logger


def copy_datafiles(domain_name:str, _path_list:List[str], output_dir:str,mode:str="train"):
    for img in _path_list:
        _img_class = img.split("/")[-2:]
        _new_img_path = os.path.join(output_dir, domain_name, mode, *_img_class)
        logger.info(f"Moving {img} ======> {_new_img_path}")
        shutil.copy(img, _new_img_path)

def create_train_test_dirs(domain_path:Path, output_dir:str) -> None:
    os.makedirs(os.path.join(output_dir, domain_path.name, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, domain_path.name, "test"), exist_ok=True)

    for _class in domain_path.glob("*"):
        if _class.is_dir():
            os.makedirs(os.path.join(output_dir, domain_path.name, "train", _class.name), exist_ok=True)
            os.makedirs(os.path.join(output_dir, domain_path.name, "test", _class.name), exist_ok=True)


def perform_train_test_split(domain_path:Path, output_dir:str, test_ratio:int) -> None:
    create_train_test_dirs(domain_path, output_dir)

    _train_images = {img_path for img_path in domain_path.glob("*/*")}
    _test_images = set(random.sample(_train_images, len(_train_images) // test_ratio))

    _train_images -= _test_images

    copy_datafiles(domain_path.name, _train_images ,output_dir)
    copy_datafiles(domain_path.name, _test_images ,output_dir, "test")


def split_train_test(input_dir:str,output_dir:str, test_ratio:int=10):

    _all_dirs_path = Path(f"{input_dir}")

    _all_domains = [path for path in _all_dirs_path.glob("*") if path.is_dir()]
    
    for domain in _all_domains:
        perform_train_test_split(domain, output_dir, test_ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Script to create a train test split dataset for the multiple domain based unsupervised testing")

    parser.add_argument('--input_path',
                        help='Input path for loading the images from input',
                        dest="input_path",
                        required=True)
    parser.add_argument('--output_path',
                        help='Output path for saving the images from input',
                        required=True,
                        dest="output_path")
    parser.add_argument('--test_ratio', type=int,
                        default=10,
                        help='Describe test ratio in which data needs to be split',
                        dest="test_ratio")

    args = parser.parse_args()

    split_train_test(args.input_path, args.output_path, args.test_ratio)
