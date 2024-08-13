import cv2 as cv
import numpy as np


def save_video():

    fps = 30
    height, width = 480, 640
    video_writer = cv.VideoWriter('video.mp4', cv.VideoWriter.fourcc(*'mp4v'), fps, (width, height), isColor=True)

    duration = 2  # length of the video in seconds
    fps = 30
    for _ in range(fps * duration):
        image = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
        image = image[:,:,::-1]  # RGB (numpy) to BGR (cv)
        video_writer.write(image)
    video_writer.release()


if __name__ == '__main__':

    save_video()
