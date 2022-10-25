import json
import os
import time

import numpy as np
import redis
import settings
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

db = redis.Redis(host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID)

model = ResNet50(include_top=True, weights="imagenet")


def predict(image_name):
    class_name = "shelby cobra"    #(reemplace por datos dummye)
    pred_probability = 0.85        #(reemplace por datos dummye)
    return class_name, pred_probability


def classify_process():
    while True:

        #1
        q = db.brpop(settings.REDIS_QUEUE)[1]

        #2
        q = json.loads(q.decode('utf-8'))
        class_name, pred_probability = predict(q['image'])

        #3
        pred = {
            "prediction":  class_name,  
            "score": 0.95 #float(pred_probability)  (reemplace por datos dummye)
            }  

        #4
        job_id = q['id']
        db.set(job_id, json.dumps(pred))

        # Don't forget to sleep for a bit at the end
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()