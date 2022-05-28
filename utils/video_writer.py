from typing import List
import threading

import cv2
import numpy as np

import moviepy.editor as moviepy


class VideoWriter:

    def __init__(self, fps: int = 30, show: bool = True, save_freq: int = 10, name: str = ''):
        """
        Class for saving img, crating gif and displaying process
        :param fps: fps of gif
        :param show:  whether to show process or not
        :param save_freq: save frequency for last parts
        :param name: name to save gif and img
        """
        self.fps = fps
        self.name = name
        self.save_freq = save_freq
        self.show = show
        self.imgs = []
        self.thread = threading.Thread(target=self._start)

    def _start(self):
        """
        Starts showing process
        :return:
        """
        while True and self.show:
            cv2.imshow(f'Frame', self.imgs[-1][:, :, ::-1])
            if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                break

    def append_img(self, img) -> None:
        """
        Append img (adjusted) to list of imgs
        :param img: image from epoch
        :return: None
        """
        self.imgs.append(self._rescale_to_uint(img))
        if len(self.imgs) == 1:
            self.thread.start()

    def _rescale_to_uint(self, img):
        """
        Rescale float img to uint8 for opencv
        :param img: image from epoch
        :return: None
        """
        img += np.array([123.675, 116.28, 103.53]).reshape((1, 1, 3))
        img = np.clip(img, 0, 255).astype('uint8')
        return img

    def _reorganize_images(self, imgs: List[np.ndarray]):
        """
        Reorganize list of images for creating gif (last images are changing very slowly, so ignore them)
        :param imgs:
        :return:
        """
        num_images = len(imgs)
        new_imgs = imgs[:num_images // 10] + [img for i, img in enumerate(imgs[num_images // 10:]) if
                                             i % self.save_freq == 0]
        return new_imgs

    def save(self) -> None:
        """
        Create gif and image and saves it
        :return: None
        """
        h, w, _ = self.imgs[0].shape
        editor = moviepy.ImageSequenceClip(self._reorganize_images(self.imgs), fps=self.fps)
        editor.write_gif(f'outputs/{self.name}.gif', fps=self.fps)
        cv2.imwrite(f'outputs/{self.name}.png', self.imgs[-1][:, :, ::-1])
