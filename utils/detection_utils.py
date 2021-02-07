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
# placeholder for saving category-Id mapping (dictionary)
category_map = None
# Color Map Dictionary
color_map = None
# counter to save the image number
save_counter = 0


# function to populate caretogy map dictionary
def create_category_id_map(categories):
    global category_map
    category_map = {}
    for instance in categories:
        category_map[instance['id']] = instance['name']
    return


# Function to populate color map dictionary
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
    #print('INFERENCE GRAPH LOADED')
    # Load color maps from csv file after loading the graph
    create_category_id_map(categories)
    create_color_dict('utils/colors.csv')
    return detection_graph, sess


# main function to perform object detection
def detect(image, detection_graph, sess):
    # obtain the tensors
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image_expanded = np.expand_dims(image, axis=0)
    # Run the session and obtain bbox, confidence scores and classes for a single image
    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                             feed_dict={image_tensor: image_expanded})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)


# main function that draws the bounding boxes on the cars
def draw_boxes(image, image_height, image_width, boxes, scores, classes):
    threshold = 0.80    # Threshold value : minimum required confidence score
    proximity_lr = 4    # pixel distance on which the proximity line is to be drawn for left right
    proximity_ud = 2    # pixel distance on which the proximity points will be drawn for up and down proximity detection

    car_regions = []            # Stores the bounding boxes for all the cars detected in one image
    car_proximity_lines = []    # Stores Proximity pline points for all the cars detected in one image ((left-up), (right down))

    # draw region of interest
    top_left, top_right, bottom_left, bottom_right = draw_region_of_interest()
    cv2.line(image, top_left, top_right, (102, 255, 102), 2, 1)         # top line
    cv2.line(image, top_left, bottom_left, (102, 255, 102), 1, 1)       # left line
    cv2.line(image, bottom_left, bottom_right, (102, 255, 102), 2, 1)   # bottom line
    cv2.line(image, top_right, bottom_right, (102, 255, 102), 1, 1)     # right line

    divider_x = 645     # defining the x coordinate of the Lane Divider

    # for all the detected objects find all vehicles detected
    for i in range(len(classes)):
        if scores[i] > threshold and classes[i] in [2, 3, 4, 6, 8]:

            # get absolute value of pixel based on image size - To obtain exact bounding box for the object
            top = boxes[i][0] * image_height
            left = boxes[i][1] * image_width
            bottom = boxes[i][2] * image_height
            right = boxes[i][3] * image_width

            # if the car is inside the region of interest draw the bounding box and check if it is close to another car
            if int(top) > top_left[1] and int(bottom) < bottom_left[1]:

                # Load the color map for the class detected - for drawing the bbox and confidence score in class specific color
                color_df = color_map[color_map['ID'] == classes[i]]
                color = int(color_df['R']), int(color_df['G']), int(color_df['B'])  # Obtain RGB values

                p1 = (int(left), int(top))      # top-left point (co-ordinate) of the detected object
                p2 = (int(right), int(bottom))  # bottom-right point (co-ordinate) of the detected object

                # obtain proximity line coordinates to draw on the Image
                prox_left_up, prox_left_down, prox_right_up, prox_right_down = create_proximity_line(p1, p2, proximity_lr)

                # Drawing Proximity Points
                cv2.line(image, prox_left_up, prox_left_down, color, 2, 2)
                cv2.line(image, prox_right_up, prox_right_down, color, 2, 2)

                # Drawing the bounding box and class label
                cv2.rectangle(image, p1, p2, color, 2, 1)
                cv2.putText(image, str(category_map[classes[i]] + str(' {:.2f}').format(scores[i])), (int(left), int(top)-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

                # saving the bounding box into the list for checking cars that are getting too close
                car_regions.append((int(top), int(left), int(bottom), int(right)))
                # Saving Proximity line which will be used to check this
                car_proximity_lines.append((prox_left_up, prox_left_down, prox_right_up, prox_right_down))

    # Calls the function that performs the check -> if two cars are getting too close
    # and returns the list containing tuples of cars that are getting close
    # After obtaining the above, Draws the Red Bounding Box and TOO CLOSE label on the image
    close_cars = check_too_close(car_regions, car_proximity_lines, divider_x)
    for cars in close_cars:
        # obtain the two cars that are getting close
        car1, car2 = cars
        p1top, p1left, p1bottom, p1right = car1
        p2top, p2left, p2bottom, p2right = car2

        # define bounding box points for first car
        c1_p1 = (p1left, p1top)
        c1_p2 = (p1right, p1bottom)

        # Defining bounding box points for second car
        c2_p1 = (p2left, p2top)
        c2_p2 = (p2right, p2bottom)

        # Drawing the Red BBox and TOO CLOSE label
        cv2.rectangle(image, c1_p1, c1_p2, (0, 0, 255), 3, 1)
        cv2.putText(image, "TOO CLOSE", (p1left+5, p1top+15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255), 1)
        cv2.rectangle(image, c2_p1, c2_p2, (0, 0, 255), 3, 1)

        # Calls the utility function to save the car images for the cars that are getting too close
        util_to_save_cars_that_are_close(c1_p1, c1_p2, c2_p1, c2_p2, image)
    return image


# mark the region inside which cars are needed to be detected
def draw_region_of_interest():
    top_left = 434, 340
    top_right = 835, 340
    bottom_left = 5, 600
    bottom_right = 1242, 600
    return top_left, top_right, bottom_left, bottom_right


# function to create the proximity line points for one bounding box of a car
def create_proximity_line(p1, p2, proximity_distance):
    proximity_distance = proximity_distance*1
    angle_distance = 8
    left, top = p1
    right, bottom = p2
    box_length = bottom - top

    # Create left side proximity line coordinates for the car
    left_up_x = left - proximity_distance
    left_up_y = top + (box_length//3)
    left_down_x = left - proximity_distance
    left_down_y = top + (box_length//3)*2

    # create right side proximity line coordinates for the car
    right_up_x = right + proximity_distance
    right_up_y = top + (box_length//3)
    right_down_x = right + proximity_distance
    right_down_y = top + (box_length//3)*2

    # create a tuple of tuples for the points and return it
    points = ((left_up_x, left_up_y),
              (left_down_x, left_down_y),
              (right_up_x, right_up_y),
              (right_down_x, right_down_y))
    return points


# Utility function to save images of all the cars that are getting too close
def util_to_save_cars_that_are_close(c1_p1, c1_p2, c2_p1, c2_p2, image):
    global save_counter
    pth = 'detected_close_folder/'  # folder where image is saved
    save_counter += 1   # Counter to append to image name
    x = 0   # defines x coordinate stored at index 0 of the tuple
    y = 1   # Defines y coordinate stored at index 1 of the tuple
    # Crop the cars based on their bounding boxes
    car1 = image[c1_p1[y]:c1_p2[y], c1_p1[x]:c1_p2[x]]
    car2 = image[c2_p1[y]:c2_p2[y], c2_p1[x]:c2_p2[x]]
    # Save the image to the directory
    cv2.imwrite(pth + str(save_counter) + str("car1") + '.png', car1)
    cv2.imwrite(pth + str(save_counter) + str("car2") + '.png', car2)
    #print("saving close cars :" + str(save_counter)+'.png')


# logic to check if the cars are getting too close
# checks if proximity points of a car are inside the other car rectangle
def check_too_close(car_regions, car_proximity_lines, divider_coords):

    """

    Args:
        car_regions:
        car_proximity_lines:
        divider_coords:

    Check if the cars are on the same side of the divider, if yes then checks if the proxy line of one car is inside the bbox of the other car,
    if yes, then the cars are close and appends to the close cars list for further plotting

    """

    close_cars = []

    # loop for first car
    for i in range(len(car_regions)):
        c1_top, c1_left, c1_bottom, c1_right = car_regions[i]
        (c1_proxy_left_up_x, c1_proxy_left_up_y), (c1_proxy_left_down_x, c1_proxy_left_down_y), (c1_proxy_right_up_x, c1_proxy_right_up_y), (c1_proxy_right_down_x, c1_proxy_right_down_y)= car_proximity_lines[i]

        # loop for second car
        for j in range(len(car_regions)):
            if i != j:
                c2_top, c2_left, c2_bottom, c2_right = car_regions[j]

                # get proxy points of 2nd car
                (c2_proxy_left_up_x, c2_proxy_left_up_y), (c2_proxy_left_down_x, c2_proxy_left_down_y), (c2_proxy_right_up_x, c2_proxy_right_up_y), (c2_proxy_right_down_x, c2_proxy_right_down_y) = car_proximity_lines[j]

                # check if both the cars are on the same side of the divider
                # both cars are on the left side

                # -- left lane
                if (c1_right < divider_coords) and (c2_right < divider_coords):

                    # check which is the leftmost car
                    if c1_left < c2_left:

                        # checking for left car
                        # if the proxy lines of one car is inside the other cars bbox y coords
                        if (c1_proxy_right_up_y > c2_top) and (c1_proxy_right_down_y < c2_bottom):

                            # now proxy line of left car is inside bbox as per y co-ordinates (vertically) of right car
                            # now we check if left car's proxy line is inside bbox of right car's horizontally
                            if (c1_proxy_right_up_x > c2_left) and (c1_proxy_right_down_x > c2_left):

                                # Now proxy line is perfectly inside the bbox of other car
                                # this means thy both are close
                                close_cars.append((car_regions[i], car_regions[j]))
                                break

                        # Check for right car
                        # if proxy points of right car is inside of left car's bounding box
                        if(c2_proxy_left_up_y > c1_top) and (c2_proxy_left_down_y < c1_bottom):
                            if (c2_proxy_left_up_x < c1_right) and (c2_proxy_left_down_y < c1_right):
                                close_cars.append((car_regions[i], car_regions[j]))
                                break

                    # if car c2 is on the left
                    else:

                        # checking for car 2's proxy is inside car a's box
                        if (c2_proxy_right_up_y > c1_top) and (c2_proxy_right_down_y < c1_bottom):
                            if (c2_proxy_right_up_x > c1_left) and (c2_proxy_right_down_x > c1_left):
                                close_cars.append((car_regions[i], car_regions[j]))
                                break

                        # checking for car 1's proxy is inside car 2's box
                        if (c1_proxy_left_up_y > c2_top) and (c1_proxy_left_down_y < c2_bottom):
                            if (c1_proxy_left_up_x < c2_right) and (c1_proxy_left_down_x < c2_right):
                                close_cars.append((car_regions[i], car_regions[j]))
                                break

                # -- right lane

                # both cars are on the right side
                elif (c1_left > divider_coords) and (c2_left > divider_coords):

                    # check the leftmost car
                    # if car1 is on the left:
                    if c1_left < c2_left:

                        if (c1_proxy_right_up_y > c2_top) and (c1_proxy_right_down_y < c2_bottom):
                            if (c1_proxy_right_up_x > c2_left) and (c1_proxy_right_down_x > c2_left):
                                close_cars.append((car_regions[i], car_regions[j]))
                                break

                        if (c2_proxy_left_up_y > c1_top) and (c2_proxy_left_up_y < c1_bottom):
                            if (c2_proxy_left_up_x < c1_right) and (c2_proxy_left_down_x < c1_right):
                                close_cars.append((car_regions[i], car_regions[j]))
                                break

                    # if car2 is on the left
                    else:

                        # checking firstly for car 2
                        if (c2_proxy_right_up_y > c1_top) and (c2_proxy_right_down_y < c1_bottom):
                            if (c2_proxy_right_up_x > c1_left) and (c2_proxy_right_down_x > c1_left):
                                close_cars.append((car_regions[i], car_regions[j]))
                                break

                        # checking for car 1
                        if (c1_proxy_left_up_y > c2_top) and (c1_proxy_left_down_y < c2_bottom):
                            if (c1_proxy_left_up_x < c2_right) and (c1_proxy_left_down_x < c2_right):
                                close_cars.append((car_regions[i], car_regions[j]))
                                break

    return close_cars
