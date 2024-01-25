#!/usr/bin/env python

from playsound import playsound

from typing import Sequence

from google.cloud import vision, storage

from time import sleep

from cv2 import VideoCapture, imwrite, CAP_PROP_AUTO_EXPOSURE, CAP_PROP_EXPOSURE, CAP_PROP_BUFFERSIZE, imshow, destroyAllWindows

from datetime import datetime

import os

from threading import Thread

image_uri = "gs://image-detection-bucket/uploaded_image.jpg"
features = [vision.Feature.Type.LABEL_DETECTION]

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True


video_getter = VideoGet(0).start()

def analyze_image_from_uri(
    image_uri: str,
    feature_types: Sequence,
) -> vision.AnnotateImageResponse:
    client = vision.ImageAnnotatorClient()

    image = vision.Image()
    image.source.image_uri = image_uri
    features = [vision.Feature(type_=feature_type) for feature_type in feature_types]
    request = vision.AnnotateImageRequest(image=image, features=features)

    response = client.annotate_image(request=request)

    return response


def print_labels(response: vision.AnnotateImageResponse):
    print("=" * 80)
    for label in response.label_annotations:
        print(
            f"{label.score:4.0%}",
            f"{label.description:5}",
            sep=" | ",
        )




def upload_blob():
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    bucket_name = "image-detection-bucket"
    # The path to your file to upload
    source_file_name = "/home/user/Documents/code/ai_cat_detect/image.jpg"
    # The ID of your GCS object
    destination_blob_name = "uploaded_image.jpg"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    try:
        blob.delete()
    except:
        print("No blob to delete")

    while True:
        try:
            blob.upload_from_filename(source_file_name)
            break
        except:
            print("Blob already exists.")

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")



def main():
    while True:
        global video_getter
        image = video_getter.frame
        result = "result"

        if result:
            try:
                os.remove("/home/user/Documents/code/ai_cat_detect/image.jpg")
            except:
                print("Error when deleting image locally.")
            imwrite("/home/user/Documents/code/ai_cat_detect/image.jpg", image)
            print("Image successfully captured.")
        else:
            print("No image detected. Try plugging in a camera.")

        upload_blob()
        response = analyze_image_from_uri(image_uri, features)
        print_labels(response)

        for label in response.label_annotations:
            if label.description == "Cat" or "cats" in label.description:
                print("There is a cat. Buzzing!")
                playsound('/home/user/Documents/code/ai_cat_detect/buzz.mp3')

                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                imwrite(f"/home/user/Documents/code/ai_cat_detect/images/{now}_cat.jpg", image)
                print("Cat captured.")

                break

        sleep(8)

if __name__ == "__main__":
    main()










