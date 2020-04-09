import streamlit as st
import numpy as np
import json
import random
import cv2
import torch
from PIL import Image

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Classes of amenities Airbnb mostly cares about
subset = ['Toilet',
         'Swimming pool',
         'Bed',
         'Billiard table',
         'Sink',
         'Fountain',
         'Oven',
         'Ceiling fan',
         'Television',
         'Microwave oven',
         'Gas stove',
         'Refrigerator',
         'Kitchen & dining room table',
         'Washing machine',
         'Bathtub',
         'Stairs',
         'Fireplace',
         'Pillow',
         'Mirror',
         'Shower',
         'Couch',
         'Countertop',
         'Coffeemaker',
         'Dishwasher',
         'Sofa bed',
         'Tree house',
         'Towel',
         'Porch',
         'Wine rack',
         'Jacuzzi']

# Put target classes in alphabetical order (required for the labels being generated)
subset.sort()

# Set up default variables
CONFIG_FILE = "retinanet_model_final/retinanet_model_final_config.yaml"
MODEL_FILE = "retinanet_model_final/retinanet_model_final.pth"

# TODO Way to load model with @st.cache so it doesn't take a long time each time
@st.cache(allow_output_mutation=True)
def create_predictor(model_config, model_weights, threshold):
    """
    Loads a Detectron2 model based on model_config, model_weights and creates a default
    Detectron2 predictor.

    Returns the a Detectron2 default predictor.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_config)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    #st.write(f"Making prediction using: {cfg.MODEL.DEVICE}")
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.SCORE_THRESH_TEST = threshold

    predictor = DefaultPredictor(cfg)

    return predictor

# Inference function - TODO this could probably be abstracted somewhere else... 
from detectron2.engine import DefaultPredictor

def make_inference(image, model_config, model_weights, threshold=0.5, n=5, save=False):
  """
  Makes inference on image (single image) using model_config, model_weights and threshold.

  Returns image with n instance predictions drawn on.

  Params:
  -------
  image (str) : file path to target image
  model_config (str) : file path to model config in .yaml format
  model_weights (str) : file path to model weights 
  threshold (float) : confidence threshold for model prediction, default 0.5
  n (int) : number of prediction instances to draw on, default 5
    Note: some images may not have 5 instances to draw on depending on threshold,
    n=5 means the top 5 instances above the threshold will be drawn on.
  save (bool) : if True will save image with predicted instances to file, default False
  """
  # Create predictor
  predictor = create_predictor(model_config, model_weights, threshold)

  # Convert PIL image to array
  image = np.asarray(image)
  
  # Create visualizer instance
  visualizer = Visualizer(img_rgb=image,
                          # TODO: maybe this metadata variable could be improved?.. yes it can
                          metadata=MetadataCatalog.get(cfg.DATASETS.TEST[0]).set(thing_classes=subset),
                          #metadata=MetadataCatalog.get(cfg.DATASETS.TEST[0]),
                          scale=0.3)
  outputs = predictor(image) # Outputs: https://detectron2.readthedocs.io/modules/structures.html#detectron2.structures.Instances
  
  # Get instance predictions from outputs
  instances = outputs["instances"]

  # Draw on predictions to image
  vis = visualizer.draw_instance_predictions(instances[:n].to("cpu"))

  return vis.get_image(), instances[:n]

def main():
    st.title("Airbnb Amenity Detection 👁")
    st.write("This application replicates [Airbnb's machine learning powered amenity detection](https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e).")
    st.write("## How does it work?")
    st.write("Add an image of a room and a machine learning learning model will look at it and find the amenities like the example below:")
    st.image(Image.open("images/example-amenity-detection.png"), 
             caption="Example of model being run on a bedroom.", 
             use_column_width=True)
    st.write("## Upload your own image")
    st.write("**Note:** The model has been trained on typical household rooms and therefore will only with those kind of images.")
    uploaded_image = st.file_uploader("Choose a png or jpg image", 
                                      type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        
        n_boxes_to_draw = st.slider(label="Number of amenities to detect (boxes to draw)",
                                    min_value=1, 
                                    max_value=10, 
                                    value=5)

        # Make sure image is RGB
        image = image.convert("RGB")

        # TODO: Fix image size if the instance is breaking
        # If image is over certain size
        # Resize image to certain size
        # Don't make the image too small 
        #st.write(image.size)
        
        if st.button("Make a prediction"):
          # TODO: Add progress/spinning wheel here
          "Making a prediction and drawing", n_boxes_to_draw, "amenity bedboxes on your image..."
          with st.spinner("Doing the math..."):
            custom_pred, preds = make_inference(
                image=image,
                model_config=CONFIG_FILE,
                model_weights=MODEL_FILE,
                n=n_boxes_to_draw
            )
            st.image(custom_pred, caption="Amenities detected.", use_column_width=True)
          classes = np.array(preds.pred_classes)
          st.write("Amenities detected:")
          st.write([subset[i] for i in classes])
        
    st.write("## How is this made?")
    st.write("The machine learning happens with a fine-tuned [Detectron2](https://detectron2.readthedocs.io/) model (PyTorch), \
    this front end (what you're reading) is built with [Streamlit](https://www.streamlit.io/) \
    and it's all hosted on [Google's App Engine](https://cloud.google.com/appengine/).")
    st.write("See the [code on GitHub](https://github.com/mrdbourke/airbnb-object-detection) and a [YouTube playlist](https://www.youtube.com/playlist?list=PL6vjgQ2-qJFeMrZ0sBjmnUBZNX9xaqKuM) detailing more below.")
    st.video("https://youtu.be/C_lIenSJb3c")

if __name__ == "__main__":
    main()