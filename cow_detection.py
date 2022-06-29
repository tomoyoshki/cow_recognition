'''
This program detects cows of an image
'''
import os
import sys
import time
import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import cv2
import image_slicer
import warnings
import exif

warnings.filterwarnings("ignore")

# load model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# send to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# set to evaluation mode
model.eval()

# load the COCO dataset category names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# we want to capture anything that may look like a cow in the wildlife
DESIRED_CLASS = [
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
]

camera_dictionary = {
    "TW1": {
        "C1N": (3.2004, 85),
    },
    "TW2": {
        "C1S": (3.23088, 84),
        "C1S_B": (3.29184, 78),
        "C2S": (3.59664, 87),
        "C2S_B": (3.3528, 80),
        "C3S": (3.32232, 80),
        "C3S_B": (3.26136, 81),
        "C3N": (3.32232, 80),
        "C3N_B": (3.26136, 81),
        "C4N": (3.29184, 84),
        "C4N_B": (3.24612, 79),
        "C5N": (3.6576, 71),
        "C5N_B": (3.41376, 75)
    },
    "TW3": {
        "C1S": (3.3528, 82)
    },
    "TW4": {
        "C1SW": (3.41376, 88),
        "C1SW_B": (3.38328, 78),
        "C2S": (3.3528, 81),
        "C2S_B": (3.32232, 86),
        "C3NE": (3.6576, 83),
        "C3NE": (3.38328, 76),
        "C4N": (3.62712, 83),
        "C4N_B": (3.3528, 79),
        "C5NW": (3.71856, 82),
        "C5NW_B": (3.41376, 80),
    }
}

camera_coordinates = {
    "TW1": (33.8798, 85.701),
    "TW2": (33.8875, 85.6942),
    "TW3": (34.3825, 85.6427),
    "TW4": (34.3823, 85.6447)
}

def get_prediction(img_path, confidence):
    """
    get_prediction
    parameters:
        - img_path - path of the input image
        - confidence - threshold value for prediction score
    method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
        are chosen.

    """

    # read image and transform to tensor
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img).to(device)

    # put into the model
    pred = model([img])

    # read the predicted class and box
    pred_class = [
        COCO_INSTANCE_CATEGORY_NAMES[i]
        for i in list(pred[0]['labels'].detach().cpu().numpy())
    ]
    pred_boxes = [
        [(i[0], i[1]), (i[2], i[3])]
        for i in list(pred[0]['boxes'].detach().cpu().numpy())
    ]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())

    # filter out those objects that are below the confidence
    pred_t = [pred_score.index(x) for x in pred_score if x>confidence]
    if len(pred_t) == 0:
        return [], []
    pred_t = pred_t[-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class

def detect_object(img_path, confidence=0.5, rect_th=2, text_size=2, text_th=2):
    """
    object_detection_api
    parameters:
        - img_path - path of the input image
        - confidence - threshold value for prediction score
        - rect_th - thickness of bounding box
        - text_size - size of the class label text
        - text_th - thichness of the text
    method:
        - prediction is obtained from get_prediction method
        - for each prediction, bounding box is drawn and text is written
        with opencv
        - the final image is displayed
    """
    # Read the image
    image = cv2.imread(img_path)
    HEIGHT, WIDTH = image.shape[0:2]
    cow_count = 0

    # Slice the images into defined slices
    num_slice = 4
    slices = image_slicer.slice(img_path, num_slice)
    num_slice = len(slices)


    dim =  int(np.sqrt(num_slice))
    sheight = HEIGHT // dim
    swidth = WIDTH // dim


    image_coordinates = []

    slice_paths = []

    # predict for each slice
    for i in range(0, dim):
        for j in range(0, dim):
            # get the slcied image
            sliced = slices[i * dim + j]
            sliced_path = sliced.generate_filename(prefix=img_path[:-4], path=False)

            # predict the selected slice
            boxes, pred_cls = get_prediction(sliced_path, confidence)
            # for each box, draw to the original image
            for k in range(len(boxes)):
                if pred_cls[k] in DESIRED_CLASS:
                    cow_count += 1
                    c1 = (
                        int(boxes[k][0][0] + j * swidth),
                        int(boxes[k][0][1] + i * sheight)
                    )
                    c2 = (
                        int(boxes[k][1][0] + j * swidth),
                        int(boxes[k][1][1] + i * sheight)
                    )
                    mid_x = (c2[0] + c1[0]) // 2
                    mid_y = (c2[1] + c1[1]) // 2
                    image_coordinates.append([mid_x, mid_y])
                    # cv2.rectangle(image, c1, c2, color=(0, 255, 0), thickness=rect_th)
                    # cv2.putText(
                    #     image,
                    #     'cow',
                    #     c1,
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     text_size,
                    #     (0,255,0),
                    #     thickness=text_th
                    # )
            # remove the sliced image from the directory
            slice_paths.append(sliced_path)
    for path in slice_paths:
        try:
            os.remove(path)
        except OSError as e: # name the Exception `e`
            print("Failed with:", e.strerror) # look what it says
            print("Error code:", e.code)
    return image, cow_count, image_coordinates

def DetectCows(chosen_dir):
    data = []
    DATA_DIR = chosen_dir
    CONFIDENCE = 0.42
    OUTPUT_RESULT_DIR = "./cow_count_output.txt"

    # OUTPUT_IMAGE_DIR = "cow_boxed"
    # if not os.path.exists(OUTPUT_IMAGE_DIR):
    #     os.mkdir(OUTPUT_IMAGE_DIR)

    # write to the result text
    f = open(OUTPUT_RESULT_DIR, 'w')
    # loop through all the images
    for folders in tqdm.tqdm(os.listdir(DATA_DIR)):
        for files in os.listdir(f"{DATA_DIR}/{folders}"):
            folder_dir = f"{DATA_DIR}/{folders}/"
            file_dir = f"{DATA_DIR}/{folders}/{files}"
            # check to see if the file exists
            if os.path.exists(file_dir):
                cow_image, cow_count, image_coordinates = detect_object(
                                                                        file_dir,
                                                                        confidence=CONFIDENCE)
                # new_file_dir = file_dir.replace("cow_images", OUTPUT_IMAGE_DIR)
                # # Write the new boxed images
                # if not os.path.exists(f"{OUTPUT_IMAGE_DIR}/{folders}"):
                #     os.mkdir(f"{OUTPUT_IMAGE_DIR}/{folders}")

                # # cv2.imwrite(new_file_dir, cow_image)
                # plt.imshow(cow_image)
                data.append([file_dir, cow_count, image_coordinates])
                # Write the result to the file
                chosen_dir_split = chosen_dir.split("/")
                for path in chosen_dir_split:
                    if len(path.strip()) > 0:
                        f.write("%s, " % path)
                f.write("%s, %s, %d" % (folders, files, cow_count))
                for pt in image_coordinates:
                    f.write(", (%d, %d)" % (pt[0], pt[1]))
                f.write("\n")
    f.close()

# Image pixel coordinate to relative coordinate
def image_to_relative(row, col):
    """
    image_to_relative
        parameters:
        - row: image row
        - col: image col
        method:
        - convert image coordinate to relative coordinate delta L and delta D        
    """
    row_int = int(row)
    col_int = int(col)
    delta_l = 2.14 + 5.32 * (10 ** -3) * np.abs(col_int - 6080 / 2) + -3.39 * (10 ** -4) * row_int + 4.66 * (10 ** -7) * (col_int - 6080 / 2)**2 + -2.14 * (10 ** -6) * np.abs(col_int - 6080/2) * row_int
    delta_d = 412303.05 * row_int ** (-1.33)
    return delta_l, delta_d

def convert_rc_to_xy(lat, long, gamma, r, c, direction):
    """
    Convert pixel coordinates to geographic coordinates
        parameters
        - lat: latitude of the camera
        - long: longitude of the camera
        - gamma: given gamma angle
        - r: image row
        - c: image column
        - direction: direction the camera is pointing at
        method:
        - convert image coordinate into geolocation
    """
    angle = np.deg2rad(45)
    
    # Getting the rotational matrix for differnet angle
    weight_class = {
        "N": np.array([[-1, 0], [0, -1]]),
        "E": np.array([[0, 1], [-1, 0]]),
        "W": np.array([[0, -1], [1, 0]]),
        "S": np.array([[1, 0], [0, 1]]),
        "NW": np.array([
                        [np.cos(angle), -np.sin(angle)], 
                        [np.sin(angle), np.cos(angle)]
                        ])  @ np.array([[-1, 0], [0, -1]]),
        "NE": np.array([
                        [np.cos(angle), -np.sin(angle)], 
                        [np.sin(angle), np.cos(angle)]
                        ])  @ np.array([[0, 1], [-1, 0]]),
        "SW": np.array([
                        [np.cos(angle), -np.sin(angle)], 
                        [np.sin(angle), np.cos(angle)]
                        ])  @ np.array([[0, -1], [1, 0]]),
        "SE": np.array([
                        [np.cos(angle), -np.sin(angle)], 
                        [np.sin(angle), np.cos(angle)]
                        ])  @ np.array([[1, 0], [0, 1]]),
                        
    }

    # Formula given by Chongya
    gamma = np.radians(gamma)
    cosgamma = np.cos(gamma)
    singamma = np.sin(gamma)
    dl, dd = image_to_relative(r,c)
    dx = dl*cosgamma - dd*singamma
    dy = dl*singamma + dd*cosgamma
    R=6378137

    dLat = dy/R
    dLon = dx/(R*np.cos(np.pi*lat/180))
    dy = dLat * 180/np.pi
    dx = dLon * 180/np.pi

    d_mat = weight_class[direction]
    dxp, dyp = d_mat @ np.array([[dx], [dy]])
    y = lat + dyp
    x = long + dxp
    return y, x

def ConvertRelGeo():
    file_name = "./cow_count_output.txt"
    image_info = []
    with open(file_name) as f:
        line = f.readline()
        while line != "":
            image_info.append(line)
            line = f.readline()
    TARGET_FILES = "./relative_final_result.txt"
    image_info = np.array(image_info)
    def convert(file_name):
        with open(file_name, "w") as f:
            cameras = ["TW1", "TW2", "TW3", "TW4"]
            for infos in image_info:
                infos = infos.split(",")
                # strip the line info
                for i in range(0, len(infos)):
                    infos[i] = infos[i].strip()
                # get the filename
                # file_name = f"{infos[0]}/{infos[1]}"
                cow_number = infos[4]

                # if now cow, simply write and continue
                if cow_number == '0':
                    content = "%s, %s, %s, %s, %s\n" % (infos[0], infos[1], infos[2], infos[3], cow_number)
                    f.write(content)
                rest = infos[5:]
                coordinate = ""
                # get each coordinates
                for i in range(0, len(rest), 2):
                    # get the row
                    row = rest[i][1:]
                    # get the column
                    col = rest[i + 1][:-1]

                    camera_direction = None

                    # get the camera type and direction
                    x_0, y_0, gamma = 0, 0, 0
                    for camera in cameras:
                        if camera in infos[1] and camera in camera_coordinates.keys():
                            x_0, y_0 = camera_coordinates[camera]
                            camera_directions = camera_dictionary[camera]
                            camera_directions_keys = list(camera_directions)
                            for direction in camera_directions_keys:
                                if direction in infos[1]:
                                    gamma = camera_dictionary[camera][direction][1]
                                    compass_direction = [
                                        "NE",
                                        "SW",
                                        "SE",
                                        "NW",
                                        "E",
                                        "W",
                                        "S",
                                        "N"
                                    ]
                                    for d in compass_direction:
                                        if d in direction:
                                            camera_direction = d
                                            break
                                    break
                            break
                    # coordinate conversion
                    geo_x, geo_y = convert_rc_to_xy(x_0, y_0, gamma, row, col, camera_direction)
                    coordinate += "(%f; %f), " % (geo_x, geo_y)
                    coordinate = coordinate[:-2]
                    exf = open(f"{infos[0]}/{infos[1]}/{infos[2]}/{infos[3]}", 'rb')
                    img_exf = exif.Image(exf)
                    time_dig = img_exf.get("datetime_digitized")
                    content = "%s, %s, %s, %s, %s, %s, %s\n" % (infos[0], infos[1], infos[2], infos[3], time_dig, cow_number, coordinate)
                f.write(content)

    convert(TARGET_FILES)
    convert(f"{TARGET_FILES[:-3]}csv")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 ./cow_detection.py [file directory]")
        sys.exit()
    target_file_directory = sys.argv[1]
    if not os.path.exists(target_file_directory):
        print("Please enter a valid file directory to the image")
        sys.exit()
    DetectCows(target_file_directory)
    ConvertRelGeo()
