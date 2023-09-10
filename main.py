
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Depends
import imutils
import pickle
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
import shutil
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
# Load your RCNN model and label encoder
with open("label_encoder_RCNN_MobileNet.pickle", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Load your RCNN model weights (you should have the model architecture defined in this file)
from tensorflow.keras.models import load_model
model = load_model("table_detector_RCNN_MobileNet.h5")

# Constants
MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200
INPUT_DIMS = (224, 224)
MIN_PROBA = 0.99

def non_max_suppression(boxes, probabilities, overlap_thresh):
    if len(boxes) == 0:
        return []

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by their probabilities
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probabilities)

    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the index
        # value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of the
        # bounding box and the smallest (x, y) coordinates for the
        # end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have overlap
        # greater than the specified threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    # Return only the bounding boxes and probabilities that were picked
    return boxes[pick], probabilities[pick]


def selective_search(image):
    # Preprocess the input image
    image = imutils.resize(image, width=500)

    # Run selective search on the image to generate bounding box proposal regions
    print("[INFO] running selective search...")
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    print("[INFO] Number of proposals:", len(rects))

    # Initialize lists for proposals and bounding boxes
    proposals = []
    boxes = []

    # Loop over the region proposal bounding box coordinates
    for (x, y, w, h) in rects[:MAX_PROPOSALS_INFER]:
        # Extract the region from the input image and preprocess it
        roi = image[y:y + h, x:x + w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)

        # Update the proposals and bounding boxes lists
        proposals.append(roi)
        boxes.append((x, y, x + w, y + h))

    # Convert lists to NumPy arrays
    proposals = np.array(proposals, dtype="float32")
    boxes = np.array(boxes, dtype="int32")

    # Classify proposal ROIs using the fine-tuned model
    print("[INFO] classifying proposals...")
    proba = model.predict(proposals)
    print("[INFO] Available classes:", label_encoder.classes_)

    # Apply non-maximum suppression
    print("[INFO] applying NMS...")
    labels = label_encoder.classes_[np.argmax(proba, axis=1)]
    idxs = np.where(labels == "table")[0]

    # Filter bounding boxes and probabilities
    boxes = boxes[idxs]
    proba = proba[idxs][:, 1]

    # Apply non-maximum suppression to the filtered boxes
    boxes, proba = non_max_suppression(boxes, proba, overlap_thresh=0.3)

    return boxes, proba

@app.post("/upload/")
async def upload_image(file: UploadFile):
    # Check if the uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Please upload an image file."
        )

    # Read the uploaded image
    image = await file.read()
    np_image = np.frombuffer(image, np.uint8)
    cv_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Perform selective search
    boxes, proba = selective_search(cv_image)

    # Prepare the response with bounding box coordinates
    response_data = {"bounding_boxes": boxes.tolist()}
    return JSONResponse(content=response_data)

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request":request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
