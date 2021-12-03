"""
author: Sanidhya Mangal
github: sanidhyamangal
"""

# import random
import random
# import path for path functions
from pathlib import Path
from typing import Optional, Tuple

# import tensorflow
import tensorflow as tf

from datapipeline.transforms import custom_augment


class PreprocessMixin:
    # function to make process the images
    def process_image(self, image_path):
        # read image into a raw format
        raw_image = tf.io.read_file(image_path)
        # decode the image
        decode_image = tf.image.decode_png(raw_image, channels=self.channel)

        # return the resized images
        return tf.image.resize(decode_image, self.image_shape)

class OnlyPerturbedMixin(PreprocessMixin):

    def process_image(self, image_path):
        un_augmented_image =  super().process_image(image_path)

        return custom_augment(un_augmented_image)

class PerturbedAndBaseMixin(PreprocessMixin):
    def process_pertubed_images(self, image_path):
        un_augmented_image = self.process_image(image_path)

        return un_augmented_image, custom_augment(un_augmented_image)

class BaseCreateDatasetMixin:
    def create_dataset(self,
                       batch_size: int,
                       shuffle: bool = True,
                       autotune: Optional[int] = None,
                       drop_remainder: bool = False,
                       pertubed_images: bool = False,
                       **kwargs):

        cache = kwargs.pop('cache', False)
        prefetch = kwargs.pop('prefetch', False)
        num_transform_ops = kwargs.pop("num_transformation_ops", 4)

        # make a dataset for the labels
        labels_dataset = tf.data.Dataset.from_tensor_slices(
            self.all_images_labels)

        # develop an image dataset
        image_dataset = tf.data.Dataset.from_tensor_slices(
            self.all_images_path)

        # process the image dataset
        if pertubed_images:
            image_dataset = image_dataset.map(self.process_pertubed_images, num_parallel_calls=autotune)
            ds = image_dataset
        else:
            image_dataset = image_dataset.map(self.process_image,
                                            num_parallel_calls=autotune)
            ds = tf.data.Dataset.zip((image_dataset, labels_dataset))

        return self.perform_data_ops(ds, shuffle, cache, batch_size,
                                     drop_remainder, prefetch)

    def perform_data_ops(self, ds, shuffle, cache, batch_size, drop_remainder,
                         prefetch):

        # create a batch of dataset
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)

        # check if cache is enabled or not
        if cache:
            ds = ds.cache()

        # check if prefetch is specified or not
        if prefetch:
            ds = ds.prefetch(prefetch)

        return ds


class LoadData(PreprocessMixin, BaseCreateDatasetMixin):
    """
    A data loader class for loading images from the respective dirs
    """

    # constructor for loading data path
    def __init__(self,
                 path,
                 image_shape: Tuple[int] = (224, 224),
                 channel: int = 3):

        # load root path
        self.path_to_dir = Path(path)
        self.image_shape = image_shape
        self.channel = channel
        self.all_images_labels = self.load_labels()

    def load_labels(self):

        # path to all the images in list of str
        self.all_images_path = [
            str(path) for path in self.path_to_dir.glob("*/*")
        ]

        # shuffle the images to add variance
        random.shuffle(self.all_images_path)

        # get the list of all the dirs
        all_root_labels = sorted([
            str(path.name) for path in self.path_to_dir.glob("*")
            if path.is_dir()
        ])

        # design the dict of the labels
        self.root_labels = dict((c, i) for i, c in enumerate(all_root_labels))

        # add the labels for all the images
        all_images_labels = [
            self.root_labels[Path(image).parent.name]
            for image in self.all_images_path
        ]

        return all_images_labels

class LoadPACDataset(PerturbedAndBaseMixin,LoadData):
    pass

class LoadSuperConData(OnlyPerturbedMixin, LoadData):
    """
    A data loader class for loading images from the respective dirs
    """

    def load_labels(self):

        # path to all the images in list of str
        self.all_images_path = [
            str(path) for path in self.path_to_dir.glob("*/*/*")
        ]

        # shuffle the images to add variance
        random.shuffle(self.all_images_path)

        # get the list of all the dirs
        all_root_labels = list({
            str(path.name) for path in self.path_to_dir.glob("*/*")
            if path.is_dir()
        })

        # design the dict of the labels
        self.root_labels = dict((c, i) for i, c in enumerate(all_root_labels))

        # add the labels for all the images
        all_images_labels = [
            self.root_labels[Path(image).parent.name]
            for image in self.all_images_path
        ]

        return all_images_labels
