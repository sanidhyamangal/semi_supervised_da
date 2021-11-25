"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import tensorflow as tf  # for deep learning
import random  # for random sampling

tf.random.set_seed(0)
random.seed(0)

TRANSFORMS = {
    "contrast": tf.image.random_contrast,
    "brightness": tf.image.random_brightness,
    "random_flip_up_down": tf.image.flip_up_down,
    "random_flip_left_right": tf.image.flip_left_right,
    "hue": tf.image.random_hue,
    "saturation": tf.image.random_saturation,
}

TRANSFORM_ARGS = {
    "contrast": dict(lower=0.2, upper=0.5),
    "brightness": dict(max_delta=0.4),
    "random_flip_up_down": dict(),
    "random_flip_left_right": dict(),
    "hue": dict(max_delta=0.3),
    "saturation": dict(lower=5, upper=10),
}


def ApplyRandomMC(image, num_ops: int = 4) -> tf.Tensor:
    image_ops = random.sample(list(TRANSFORMS), num_ops)

    img = image

    for op in image_ops:
        img = TRANSFORMS[op](img, **TRANSFORM_ARGS[op])

    return img


def GeneratePertuberations(image_dataset, num_ops: int = 4):
    pertubed_images = [
        ApplyRandomMC(image, num_ops) for image in image_dataset
    ]

    return tf.convert_to_tensor(pertubed_images)
