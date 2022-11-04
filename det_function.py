from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
import numpy as np
import cv2
import os
import functions
import argparse

# The chosen detector model is "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# because this particular model has a good balance between accuracy and speed.
# You can check the following Colab notebook with examples on how to run
# Detectron2 models
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5.
# Assign the loaded detection model to global variable DET_MODEL
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
)
try:
    # It may fail if no GPU was found
    DET_MODEL = DefaultPredictor(cfg)
except:
    # Load the model for CPU only
    print(
        f"Failed to load Detection model on GPU, "
        "trying with CPU. Message: {exp}."
    )
    cfg.MODEL.DEVICE='cpu'
    DET_MODEL = DefaultPredictor(cfg)

def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. Already "
            "splitted in train/test sets. E.g. "
            "`/home/app/src/data/car_ims_v1/`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "cropped pictures. E.g. `/home/app/src/data/car_ims_v2/`."
        ),
    )
    parser.add_argument(
        "garbage_folder",
        type=str,
    )
    args = parser.parse_args()

    return args

# def vehicle_coordinates(img):
#     outputs = DET_MODEL(img)
#     pred_boxes = outputs["instances"].pred_boxes  
#     # detected_class_indexes = outputs["instances"].pred_classes    
#     box_coordinates = None
#     i=0
#     max_area = 0
#     box_area = pred_boxes.area()[i]
#     #for x in detected_class_indexes:
#     if box_area > max_area:
#         box_coordinates = pred_boxes[i] #The biggest box coordinates. 
#         max_area = box_area
#     i+=1

#     box_coordinates = box_coordinates.tensor.cpu().numpy() #Transforming to numpy.
#     box_coordinates = [(int(x)) for x in box_coordinates[0]] #Float to integer.
         
#     return box_coordinates

def car_coordinates(img, pos):
    outputs = DET_MODEL(img)
    pred_box = outputs["instances"].pred_boxes[pos] 
    
    box_coordinates = pred_box.tensor.cpu().numpy() #Transforming to numpy.
    box_coordinates = [(int(x)) for x in box_coordinates[0]] #Float to integer.
         
    return box_coordinates


# def garbage_coordinates(img):
#     img_height, img_width = img.shape[:2] #To get the image height and width
#     box_coordinates = [0, 0, img_width, img_height] 
#     return box_coordinates

def main(data_folder, output_data_folder, garbage_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to original images folder.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        cropped images.
        
    garbage_folder : str
        Full path to garbage images folder.
    """
        
    for dirpath, filename in functions.walkdir(data_folder): 
        image_path = os.path.join(dirpath, filename)
        image = cv2.imread(image_path)
        try:
            outputs = DET_MODEL(image)
            detected_class_indexes = outputs["instances"].pred_classes
            pred_boxes = outputs["instances"].pred_boxes  
            box_area = pred_boxes.area()
            bx = box_area.tolist()
            if bx == []:
                x = 0
            else:
                pos = bx.index(max(box_area))
                x = detected_class_indexes[pos]  
                
            if x == 2 or x == 7:       
                cropp_coordinates = car_coordinates(image, pos)
                cropp_car = image[cropp_coordinates[1]:cropp_coordinates[3],
                                cropp_coordinates[0]:cropp_coordinates[2]]
                class_label = dirpath.split(os.sep)
                Model_label = class_label[-1]
                #Letter_label = class_label[-1]
                my_path = output_data_folder + '/' + Model_label + '/' # + Letter_label + '/'    
                os.makedirs(my_path, exist_ok=True)
                image_path2 = os.path.join(my_path, filename)
                cv2.imwrite(image_path2, cropp_car)
            else:
                # garbage_coords = garbage_coordinates(image)
                # cropp_img = image[garbage_coords[1]:garbage_coords[3],
                #                 garbage_coords[0]:garbage_coords[2]] 
                # os.makedirs(garbage_folder, exist_ok=True)
                garbage_path = os.path.join(garbage_folder, filename)
                cv2.imwrite(garbage_path, image)
        except AttributeError:
            pass
            
            
if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.output_data_folder, args.garbage_folder)