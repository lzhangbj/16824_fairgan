import numpy as np
import dlib
from glob import glob
import cv2
from tqdm import tqdm


# detect faces using dlib
def _filter_bbox(bboxes, h, w):
    max_area = 0
    for i, bbox in enumerate(bboxes):
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area > max_area:
            max_area = area

    ch = h // 2
    cw = w // 2
    min_center_dist = None
    ret_idx = -1
    for i, bbox in enumerate(bboxes):
        cx = (bbox[2] + bbox[0]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        center_dist = np.sqrt((cx - cw) ** 2 + (cy - ch) ** 2)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area >= max_area * 0.7 * 0.7:
            if min_center_dist is None:
                min_center_dist = center_dist
                ret_idx = i
            else:
                if center_dist < min_center_dist:
                    min_center_dist = center_dist
                    ret_idx = i

    return ret_idx


def convert_and_trim_bb(image, rect):
    # extract the starting and ending (x, y)-coordinates of the
    # bounding box
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    # ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    # compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
    # return our bounding box coordinates
    return (startX, startY, w, h)


detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

for mode in ['train', 'val']:
    invalid_cnt = 0
    images = glob(f"images/{mode}/*.jpg")
    face_bbox_dict = {}

    for image in tqdm(images):
        img = cv2.imread(image)
        h, w = img.shape[:2]

        # The 1 in the second argument indicates that we should upsample the image
        # 1 time.  This will make everything bigger and allow us to detect more
        # faces.
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # perform face detection using dlib's face detector
        results = detector(rgb, 1)

        if len(results) == 0:
            boxes = [[w//4, h//4, w//2, h//2],]
            invalid_cnt += 1
        else:
            boxes = [convert_and_trim_bb(img, r.rect) for r in results]
        bboxes = [(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]) for bbox in boxes]

        face_bbox = bboxes[_filter_bbox(bboxes, h, w)]

        image = image.replace('images/', '')

        face_bbox_dict[image] = face_bbox

    with open(f'fairface_{mode}_bbox.txt', 'w') as f:
        for image, bbox in face_bbox_dict.items():
            f.writelines(f"{image},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")




