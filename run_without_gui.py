import cv2
from utils import detection_utils


detection_graph, sess = detection_utils.load_inference_graph()


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


run_video(detection_graph, sess, 'data/video2.mp4')
