#!/usr/bin/env/python

import os
import cv2
import numpy as np
from tqdm import tqdm
from facedetector import FaceDetector
from PIL import Image


INPUT_DIR = os.path.join(os.getcwd(), "Celeb-DF-v2")
OUTPUT_DIR = os.path.join(os.getcwd(), "Celeb-DF-v2-faces")
PADDING = 1.0


def main():
    """
    Process all videos in INPUT_DIR's subdirs and extract faces using FaceDetector. Save faces to OUTPUT_DIR.

    Extracts only faces from the first frame with the current implementation. Remove the break statement on
    line 47 to process all frames (may take a long time, consider using OpenCV with CUDA platform, which 
    requires manual compilation of OpenCV with CUDA support).

    Code inspired by https://github.com/freearhey/face-extractor/tree/master
    """
    for subdir in os.listdir(INPUT_DIR):
        if not os.path.isdir(os.path.join(INPUT_DIR, subdir)):
            continue
        input_dir = os.path.join(INPUT_DIR, subdir)
        output_dir = os.path.join(OUTPUT_DIR, subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(f"Extracting faces from {input_dir}")
        for video in tqdm(os.listdir(input_dir)):
            if not video.endswith(".mp4"):
                continue
            print(f"Processing video {video}")
            stream = cv2.VideoCapture(os.path.join(input_dir, video))
            i = 1
            while True:
                success, frame = stream.read()
                if success and isinstance(frame, np.ndarray):
                    faces = FaceDetector.detect(frame)
                    array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(array)
                    for n, face in enumerate(faces):
                        bbox = face["bounding_box"]
                        pivotX, pivotY = face["pivot"]
                        if bbox["width"] < 10 or bbox["height"] < 10:
                            continue
                        left = pivotX - bbox["width"] / 2.0 * PADDING
                        top = pivotY - bbox["height"] / 2.0 * PADDING
                        right = pivotX + bbox["width"] / 2.0 * PADDING
                        bottom = pivotY + bbox["height"] / 2.0 * PADDING
                        cropped = img.crop((left, top, right, bottom))
                        cropped.save(os.path.join(output_dir, f"{video}_{i}_{n}.jpg"))
                    break  # Only process the first frame, remove this line to process all frames
                else:
                    break  # End of video
                i += 1
            stream.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
