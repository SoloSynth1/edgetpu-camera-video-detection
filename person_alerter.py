import os
import argparse
import collections
import json
import datetime
import time
import base64
from io import BytesIO
import array

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


def upload_image(image):
    with BytesIO() as buffer:
        image.save(buffer, 'png')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    url = "https://api.imgur.com/3/image"
    headers = {
        "Authorization": "Client-ID {}".format(os.environ['IMGUR_CLIENT_ID']),
    }
    r = http.request('POST', url, fields={'image': image_base64}, headers=headers)
    if r.status == 200:
        response = json.loads(r.data)
        return response['data']['link']
    else:
        return None


def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--top_k', type=int, default=5,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default=0)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='classifier score threshold')
    parser.add_argument('--video_width', type=int, help='Width resolution of the Video Capture', default=960)
    parser.add_argument('--video_height', type=int, help='Width resolution of the Video Capture', default=720)
    parser.add_argument('--confirmations', type=int,
                        help='Frames detected with one or more person(s) needed before sending out an alert', default=30)
    parser.add_argument('--time_period', type=int, help='Maximum time for confirmation check (in seconds)', default=10)
    parser.add_argument('--alert_cooldown', type=int, help='Cooldown time between alerts (in seconds)', default=120)
    args = parser.parse_args()

    print('Loading {}.'.format(args.model))
    interpreter = common.make_interpreter(args.model)
    interpreter.allocate_tensors()

    cap = cv2.VideoCapture(args.camera_idx)

    # set VideoCapture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.video_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.video_height)

    cooldown_until = 0
    confirmations = array.array('d', [0]*args.confirmations)

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
            confirmations.append(current_time)
            if confirmations[-1] - confirmations.pop(0) <= args.time_period and confirmations[-1] >= cooldown_until:
                print("alerted at", current_time)
                image_link = upload_image(pil_im)
                send_alert("person detected at {}. \n{}".format(datetime.datetime.now().isoformat(), image_link))
                cooldown_until = current_time + args.alert_cooldown
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
