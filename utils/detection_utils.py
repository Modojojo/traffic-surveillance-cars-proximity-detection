import numpy as np
import pandas as pd
import tensorflow as tf
from utils import label_map_util
import cv2

# Defining the paths to models and their label maps
FROZEN_GRAPH_FOLDER = 'frozen_graph_faster_rcnn_inception'
MODEL_PATH = FROZEN_GRAPH_FOLDER + '/frozen_inference_graph.pb'
LABELS_PATH = FROZEN_GRAPH_FOLDER + '/labelmap.pbtxt'

NUM_CLASSES = 91

# Loads label map using utility function
label_map = label_map_util.load_labelmap(LABELS_PATH)
# calls utility function that returns dictionary (id, name) for all the categories in the inputted labelmap
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# same as category but keyed by 'ID'
category_index = label_map_util.create_category_index(categories)
category_map = None
color_map = None


def create_category_id_map(categories):
    global category_map
    category_map = {}
    for instance in categories:
        category_map[instance['id']] = instance['name']
    return


def create_color_dict(color_csv_path):
    global color_map
    df = pd.read_csv(color_csv_path, header=None)
    df.columns = ['ID', 'R', 'G', 'B']
    color_map = df


# Function to load the frozen graph
def load_inference_graph():
    detection_graph = tf.Graph()    # creates a graph and places everything in it
    with detection_graph.as_default():
        graph_def = tf.GraphDef()   # GraphDef is a proto, it is the serialized version of a graph
        with tf.gfile.GFile(MODEL_PATH, 'rb') as f:
            serialized_graph = f.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print('INFERENCE GRAPH LOADED')
    create_category_id_map(categories)
    create_color_dict('utils/colors.csv')
    return detection_graph, sess


# main function to perform object detection
def detect(image, detection_graph, sess):
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image_expanded = np.expand_dims(image, axis=0)
    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                             feed_dict={image_tensor: image_expanded})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)


def draw_boxes(image, image_height, image_width, boxes, scores, classes):
    threshold = 0.65
    proximity_lr = 10
    proximity_ud = 2

    car_regions = []
    car_proximity_lines = []

    # draw region of interest
    top_left, top_right, bottom_left, bottom_right = draw_region_of_interest()
    cv2.line(image, top_left, top_right, (0, 255, 0), 1, 1)     # top line
    cv2.line(image, top_left, bottom_left, (0, 255, 0), 1, 1)       # left line
    cv2.line(image, bottom_left, bottom_right, (0, 255, 0), 1, 1)       # bottom line
    cv2.line(image, top_right, bottom_right, (0, 255, 0), 1, 1)     # right line

    divider_x = 645

    # for all the detected objects find all vehicles detected
    for i in range(len(classes)):
        if scores[i] > threshold and classes[i] in [2, 3, 4, 6, 8]:

            top = boxes[i][0] * image_height
            left = boxes[i][1] * image_width
            bottom = boxes[i][2] * image_height
            right = boxes[i][3] * image_width

            # if the car is inside the region of interest
            if int(top) > top_left[1] and int(bottom) < bottom_left[1]:

                color_df = color_map[color_map['ID'] == classes[i]]
                color = int(color_df['R']), int(color_df['G']), int(color_df['B'])

                p1 = (int(left), int(top))
                p2 = (int(right), int(bottom))

                prox_left_up, prox_left_down, prox_right_up, prox_right_down = create_proximity_line(p1, p2, proximity_lr)

                cv2.line(image, prox_left_up, prox_left_down, color, 2, 2)
                cv2.line(image, prox_right_up, prox_right_down, color, 2, 2)

                cv2.rectangle(image, p1, p2, color, 2, 1)
                cv2.putText(image, str(category_map[classes[i]] + str(' {:.2f}').format(scores[i])), (int(left), int(top)-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

                car_regions.append((int(top), int(left), int(bottom), int(right)))
                car_proximity_lines.append((prox_left_up, prox_left_down, prox_right_up, prox_right_down))

    close_cars = check_too_close(car_regions, car_proximity_lines, divider_x)
    for cars in close_cars:
        car1, car2 = cars
        p1top, p1left, p1bottom, p1right = car1
        p2top, p2left, p2bottom, p2right = car2

        c1_p1 = (p1left, p1top)
        c1_p2 = (p1right, p1bottom)

        c2_p1 = (p2left, p2right)
        c2_p2 = (p2right, p2bottom)

        cv2.rectangle(image, c1_p1, c1_p2, (0, 0, 255), 3, 1)
        cv2.putText(image, "TOO CLOSE", (p1left+5, p1top+15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255), 1)

    return image


def draw_region_of_interest():
    top_left = 500, 300
    top_right = 772, 300
    bottom_left = 7, 600
    bottom_right = 1240, 600
    return top_left, top_right, bottom_left, bottom_right


def create_proximity_line(p1, p2, proximity_distance):
    left, top = p1
    right, bottom = p2
    box_length = bottom - top

    left_up_x = left - proximity_distance
    left_up_y = top + (box_length//4)
    left_down_x = left - proximity_distance
    left_down_y = top + (box_length//4)*3

    right_up_x = right + proximity_distance
    right_up_y = top + (box_length//4)
    right_down_x = right + proximity_distance
    right_down_y = top + (box_length//4)*3

    points = ((left_up_x, left_up_y),
              (left_down_x, left_down_y),
              (right_up_x, right_up_y),
              (right_down_x, right_down_y))
    return points


# logic to check if the cars are getting too close
# checks if proximity points of a car are inside the other car rectangle
def check_too_close(car_regions, car_proximity_lines, divider_coords):

    close_cars = []

    for i in range(len(car_regions)):
        top, left, bottom, right = car_regions[i]
        prox_left_up, prox_left_down, prox_right_up, prox_right_down = car_proximity_lines[i]

        for j in range(len(car_regions)):
            if i != j:
                c_top, c_left, c_bottom, c_right = car_regions[j]

                # checking if the cars are in the same lane, if yes then only compare
                if (right < divider_coords) and (c_right < divider_coords):

                    if (prox_left_up[1] > c_top) and (prox_left_down[1] < c_bottom):
                        if (prox_left_up[0] < c_right) and (prox_left_down[0] < c_right):
                            close_cars.append((car_regions[i], car_regions[j]))
                            break
                    elif (prox_right_up[1] > c_top) and (prox_right_down[1] < c_bottom):
                        if (prox_right_up[0] > c_left) and (prox_right_down[0] > c_left):
                            close_cars.append((car_regions[i], car_regions[j]))
                            break
    return close_cars
