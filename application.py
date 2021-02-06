import cv2
from utils import detection_utils


detection_graph, sess = detection_utils.load_inference_graph()


def run_webcam(detection_graph, sess):
    capture = cv2.VideoCapture(0)
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = capture.read()
        image = frame
        image = cv2.flip(image, 1)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]
        boxes, scores, classes = detection_utils.detect(image, detection_graph, sess)
        image = detection_utils.draw_boxes(image, image_height, image_width, boxes, scores, classes)
        cv2.imshow('Detection', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
    return


def run_video(detection_graph, sess, video_path):
    capture = cv2.VideoCapture(video_path)
    cv2.namedWindow('Detection on Video', cv2.WINDOW_NORMAL)

    while capture.isOpened():
        ret, frame = capture.read()
        image = frame
        image_height, image_width = image.shape[:2]
        boxes, scores, classes = detection_utils.detect(image, detection_graph, sess)
        image = detection_utils.draw_boxes(image, image_height, image_width, boxes, scores, classes)
        cv2.imshow('Detection on Video', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
    return


def test_on_image(detection_graph, sess, image_path):
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]
    boxes, scores, classes = detection_utils.detect(image, detection_graph, sess)
    image = detection_utils.draw_boxes(image, image_height, image_width, boxes, scores, classes)
    cv2.imshow('Detection on image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#run_webcam(detection_graph, sess)
run_video(detection_graph, sess, 'data/video2.mp4')
#test_on_image(detection_graph, sess, 'test_images/Screenshot3.png')
