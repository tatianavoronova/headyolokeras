#! /usr/bin/env python

class DetectorAPI:
    def __init__(self, weights_path, config_path):
        import json
        import os
        from frontend import YOLO
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        with open(config_path) as config_buffer:
            config = json.load(config_buffer)

        ###############################
        #   Make the model
        ###############################

        self.yolo = YOLO(backend=config['model']['backend'],
                    input_size=config['model']['input_size'],
                    labels=config['model']['labels'],
                    max_box_per_image=config['model']['max_box_per_image'],
                    anchors=config['model']['anchors'])

        ###############################
        #   Load trained weights
        ###############################

        self.yolo.load_weights(weights_path)
        self.labels = config['model']['labels']

    def processFrame(self, image):
        import numpy as np
        import time
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        boxes = self.yolo.predict(image)
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)
        return boxes

def _main_():
    import cv2
    from utils import draw_boxes
    config_path = 'config.json'
    weights_path = 'model.h5'

    odapi = DetectorAPI(weights_path, config_path)
    image_path = 'Hajj3.jpg'
    image = cv2.imread(image_path)
    boxes = odapi.processFrame(image)

    image_path2 = 'img2.jpg'
    image2 = cv2.imread(image_path2)
    boxes2 = odapi.processFrame(image2)

    image = draw_boxes(image, boxes, odapi.labels)
    print(len(boxes), 'boxes are found')
    cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)

    image2 = draw_boxes(image2, boxes2, odapi.labels)
    print(len(boxes2), 'boxes are found')
    cv2.imwrite(image_path2[:-4] + '_detected' + image_path2[-4:], image2)

if __name__ == '__main__':
    _main_()