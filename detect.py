"""Yolo v3 detection script.

Saves the detections in the `detection` folder.

Usage:
    python detect.py <images/video> <iou threshold> <confidence threshold> <filenames>

Example:
    python detect.py images 0.5 0.5 data/images/dog.jpg data/images/office.jpg
    python detect.py video 0.5 0.5 data/video/shinjuku.mp4

Note that only one video can be processed at one run.
"""
import pika as pika
import tensorflow as tf
import sys
import cv2
from datetime import datetime

from yolo_v3 import Yolo_v3
from utils import load_images, load_class_names, draw_boxes, draw_frame

tf.compat.v1.disable_eager_execution()

_MODEL_SIZE = (416, 416)
_CLASS_NAMES_FILE = './data/labels/coco.names'
_MAX_OUTPUT_SIZE = 20


def main(type, iou_threshold, confidence_threshold, venue, input_names ):
    class_names = load_class_names(_CLASS_NAMES_FILE)
    n_classes = len(class_names)

    model = Yolo_v3(n_classes=n_classes, model_size=_MODEL_SIZE,
                    max_output_size=_MAX_OUTPUT_SIZE,
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold)

    if type == 'images':
        batch_size = len(input_names)
        batch = load_images(input_names, model_size=_MODEL_SIZE)
        inputs = tf.compat.v1.placeholder(tf.float32, [batch_size, *_MODEL_SIZE, 3])
        detections = model(inputs, training=False)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(scope='yolo_v3_model'))

        with tf.compat.v1.Session() as sess:
            saver.restore(sess, './weights/model.ckpt')
            detection_result = sess.run(detections, feed_dict={inputs: batch})
            pass
        res = draw_boxes(input_names, detection_result, class_names, _MODEL_SIZE)
        filtered_cars = [x for x in res if x.get("car") and x.get("car") > 60.0]
        filtered_persons = [x for x in res if x.get("person") and x.get("person") > 60.0]
        print('Detections have been saved successfully.')

        print(f'Number of persons: {len(filtered_persons)}')
        print(f'Number of cars: {len(filtered_cars)}')
        put_res_on_queue({"nmr_of_cars": len(filtered_cars), "nrm_of_persons": len(filtered_persons),
                          "time_generated": int(datetime.now().timestamp()),
                          "venue": venue})

    elif type == 'video':
        inputs = tf.compat.v1.placeholder(tf.float32, [1, *_MODEL_SIZE, 3])
        detections = model(inputs, training=False)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(scope='yolo_v3_model'))

        with tf.compat.v1.Session() as sess:
            saver.restore(sess, './weights/model.ckpt')

            win_name = 'Video detection'
            cv2.namedWindow(win_name)
            cap = cv2.VideoCapture(input_names[0])
            frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                          cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            fps = cap.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter('./detections/detections.mp4', fourcc, fps,
                                  (int(frame_size[0]), int(frame_size[1])))

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    resized_frame = cv2.resize(frame, dsize=_MODEL_SIZE[::-1],
                                               interpolation=cv2.INTER_NEAREST)
                    detection_result = sess.run(detections,
                                                feed_dict={inputs: [resized_frame]})

                    draw_frame(frame, frame_size, detection_result,
                               class_names, _MODEL_SIZE)

                    cv2.imshow(win_name, frame)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        break

                    out.write(frame)
            finally:
                cv2.destroyAllWindows()
                cap.release()
                print('Detections have been saved successfully.')

    elif type == 'webcam':
        inputs = tf.compat.v1.placeholder(tf.float32, [1, *_MODEL_SIZE, 3])
        detections = model(inputs, training=False)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(scope='yolo_v3_model'))

        with tf.compat.v1.Session() as sess:
            saver.restore(sess, './weights/model.ckpt')

            win_name = 'Webcam detection'
            cv2.namedWindow(win_name)
            cap = cv2.VideoCapture(0)
            frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                          cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            fps = cap.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter('./detections/detections.mp4', fourcc, fps,
                                  (int(frame_size[0]), int(frame_size[1])))

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    resized_frame = cv2.resize(frame, dsize=_MODEL_SIZE[::-1],
                                               interpolation=cv2.INTER_NEAREST)
                    detection_result = sess.run(detections,
                                                feed_dict={inputs: [resized_frame]})

                    draw_frame(frame, frame_size, detection_result,
                               class_names, _MODEL_SIZE)

                    cv2.imshow(win_name, frame)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        break

                    out.write(frame)
            finally:
                cv2.destroyAllWindows()
                cap.release()
                print('Detections have been saved successfully.')

    else:
        raise ValueError("Inappropriate data type. Please choose either 'video' or 'images'.")


def put_res_on_queue(res):
    credentials = pika.PlainCredentials('rabbitmq', 'rabbitmq')
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost', credentials=credentials))
    channel = connection.channel()
    channel.basic_publish(exchange='crowd_detector',
                          routing_key='groen',
                          body=str(res))


if __name__ == '__main__':

    ff = sys.argv

    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5:])
