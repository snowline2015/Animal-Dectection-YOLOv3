import streamlit as st
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utilities import get_information, draw_bounding



def detect_animal(image, conf=0.5, nms=0.4):
    with open('yolo/yolo.names', 'r') as f:
        classes = f.read().splitlines()

    net = cv2.dnn.readNetFromDarknet('yolo/yolov3_custom_train.cfg', 'yolo/yolov3_custom_train_2000.weights')

    layer_names = net.getLayerNames()
    layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    boxes, confidences, classIDs = get_information(outputs, conf, width, height)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, nms)
    img, txt = draw_bounding(image, boxes, confidences, classIDs, idxs, classes)

    return img, txt


def app_main():
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Animal Detection")
    st.write("Made with love by: Ngo Huy Hoang - 19127406 & Trieu Nguyen Minh Huy - 19127424 & Truong The Phu - 19127509")
    st.write("Animal accepted: Buffalo, Cat, Elephant, Rhino, Zebra. Any animal out of this list will cause unpredictable result")

    img = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

    if img is not None:
        image = Image.open(img)
        col1, col2 = st.columns(2)

        col1.subheader("Original Image")
        st.text("")
        plt.imshow(image)
        col1.pyplot()

        pil_image = image.convert('RGB')
        np_image = np.array(pil_image)
        np_image = np_image[:, :, ::-1].copy()
        detect_image, detect_obj = detect_animal(np_image)

        col2.subheader("Animal-Detected Image")
        st.text("")
        plt.imshow(detect_image)
        col2.pyplot()

        st.success("Found Animal: {}".format(detect_obj))


if __name__ == "__main__":
    app_main()