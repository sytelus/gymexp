import tensorflow as tf
from typing import Tuple

class NoOpFrameProcessor:
    def __call__(self, session, frame):
        pass

class AtariFrameProcessor:
    """Resizes and converts RGB Atari frames to grayscale"""
    def __init__(self, in_frame_size=(210, 160, 3), 
        out_frame_size=(84,84), grayscale=True, crop_to_bounding_box=(34, 0, 160, 160)):

        self.frame = tf.placeholder(shape=in_frame_size, dtype=tf.uint8)
        if grayscale:
            self.processed = tf.image.rgb_to_grayscale(self.frame)
        if crop_to_bounding_box is not None:
            self.processed = tf.image.crop_to_bounding_box(self.processed, *crop_to_bounding_box)
        self.processed = tf.image.resize_images(self.processed, 
                                                out_frame_size, 
                                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    def __call__(self, session, frame):
        """
        Args:
            session: A Tensorflow session object
            frame: A (210, 160, 3) frame of an Atari game in RGB
        Returns:
            A processed (84, 84, 1) frame in grayscale
        """
        return session.run(self.processed, feed_dict={self.frame:frame})
