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


@tf.function
def custom_augment(image):
    # Random flips
    image = random_apply(tf.image.flip_left_right, image, p=0.5)
    image = random_apply(tf.image.flip_up_down, image, p=0.8)

    # Randomly apply transformation (color distortions) with probability p.
    image = random_apply(color_jitter, image, p=0.4)
    image = random_apply(color_drop, image, p=0.2)

    return (image)


@tf.function
def color_jitter(x, s=0.5):
    # one can also shuffle the order of following augmentations
    # each time they are applied.
    x = tf.image.random_brightness(x, max_delta=0.8 * s)
    x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    x = tf.image.random_saturation(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    x = tf.image.random_hue(x, max_delta=0.2 * s)
    x = tf.clip_by_value(x, 0, 255.0)
    return x


@tf.function
def color_drop(x):
    x = tf.image.rgb_to_grayscale(x)
    x = tf.tile(x, [1, 1, 3])
    return x


@tf.function
def random_apply(func, x, p):
    return tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                tf.cast(p, tf.float32)), lambda: func(x), lambda: x)
