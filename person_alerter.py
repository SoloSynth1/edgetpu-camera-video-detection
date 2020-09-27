import os
import argparse
import collections
import json
import datetime
import time

from dotenv import load_dotenv
import urllib3
import cv2
import numpy as np
from PIL import Image

import common

load_dotenv()
WEBHOOK_URL = os.environ['WEBHOOK_URL']
WEBHOOK_HEADERS = {'Content-Type': 'application/json'}
http = urllib3.PoolManager()

PERSON_COCO_INDEX = 0

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()


def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    class_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)
    count = int(common.output_tensor(interpreter, 3))

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))

    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]


def send_alert(alert_str):
    data = {'text': alert_str}
    http.request('POST', WEBHOOK_URL, body=json.dumps(data).encode('utf-8'), headers=WEBHOOK_HEADERS)


def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--top_k', type=int, default=10,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='classifier score threshold')
    parser.add_argument('--')
    args = parser.parse_args()

    print('Loading {}.'.format(args.model))
    interpreter = common.make_interpreter(args.model)
    interpreter.allocate_tensors()

    cap = cv2.VideoCapture(args.camera_idx)


    first_hit = 0
    stop_until = 0
    double_confirm_min = 3
    double_confirm_max = 10
    refractory_period = 60

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)

        common.set_input(interpreter, pil_im)
        interpreter.invoke()
        objs = get_output(interpreter, score_threshold=args.threshold, top_k=args.top_k)
        persons = list(filter(lambda x: x.id == PERSON_COCO_INDEX, objs))
        if persons:
            current_time = time.time()
            if current_time > stop_until:
                if (current_time - first_hit) > double_confirm_max:
                    first_hit = current_time
                elif (current_time - first_hit < double_confirm_max) and (current_time - first_hit >= double_confirm_min):
                    send_alert("person detected at {}".format(datetime.datetime.now().isoformat()))
                    print("alerted.")
                    stop_until = current_time + refractory_period
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
