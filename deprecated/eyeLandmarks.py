import numpy as np
import cv2 as cv
import pandas as pd
import mediapipe as mp

# NOTE:
# upon further look, MP may only support eye segmentation without full face for C++, iOS, and Android implementations
# https://github.com/google/mediapipe
# https://google.github.io/mediapipe/solutions/iris.html
# https://arxiv.org/pdf/2006.11341.pdf

face_mesh = mp.solutions.face_mesh
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

image = cv.imread('data/1.jpg')

with face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1
) as fmesh:
    while True:
        img_h, img_w = image.shape[:2]
        results = fmesh.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            meshpts = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks])
            cv.polylines(image, [meshpts[LEFT_EYE]], True, (0, 255, 0), 1, cv.LINE_AA)
            cv.polylines(image, [meshpts[RIGHT_EYE]], True, (0, 255, 0), 1, cv.LINE_AA)
        cv.imshow('img', image)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
cv.destroyAllWindows()