from __future__ import division
from utils.models import *
from utils.utils import *
from utils.datasets import *
from utils.generate_ticket import ticket
import os
import sys
import argparse
import cv2
from PIL import Image
import torch
from torch.autograd import Variable

shopping_list = []

def Convert_RGB(img):
    # Convert Blue, green, red to Red, green, blue
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def Convert_BGR(img):
    # Convert red, blue, green to Blue, green, red
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--webcam", type=int, default=1,  help="Is the video processed video? 1 = Yes, 0 == no" )
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--directorio_video", type=str, help="Directorio al video")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)


    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    if opt.webcam==1:
        cap = cv2.VideoCapture(0)
        out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (1280,960))
    else:
        cap = cv2.VideoCapture(opt.directorio_video)

        if (cap.isOpened() == False):
            print("Error opening the video file")
        else: # Get frame rate information
            fps = int(cap.get(5))
            print("Frame Rate : ",fps,"frames per second") 
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
        out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (frame_width,frame_height))
    
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    
    a=[]
    while cap:
        ret, frame = cap.read()
        if ret is False:
            break

        #The image comes in Blue, Green, Red and we convert it to RGB which is the input required by the model.
        RGBimg=Convert_RGB(frame)
        imgTensor = transforms.ToTensor()(RGBimg)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416)
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))


        with torch.no_grad():
            detections = model(imgTensor)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        for detection in detections:
            if detection is not None:
                detection = rescale_boxes(detection, opt.img_size, RGBimg.shape[:2])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    x1 = int(x1); x2 = int(x2)
                    y1 = int(y1); y2 = int(y2)
                    box_h = y2 - y1
                    color = [int(c) for c in colors[int(cls_pred-1)]]

                    shopping_list += [str(classes[int(cls_pred) - 1])]

                    #print("{} detected in X1: {}, Y1: {}, X2: {}, Y2: {}".format(classes[int(cls_pred)], x1, y1, x2, y2))
                    frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 5)
                    cv2.putText(frame, classes[int(cls_pred-1)], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)# Nombre de la clase detectada
                    cv2.putText(frame, str("%.2f" % float(conf)), (x2-50, y2 - box_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3) # Certeza de prediccion de la clase
        #
        #We converted back to BGR so that cv2 can display it in the correct colours.

        if opt.webcam==1:
            cv2.imshow('frame', Convert_BGR(RGBimg))
            out.write(RGBimg)
        else:
            out.write(Convert_BGR(RGBimg))
            cv2.imshow('frame', RGBimg)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
    ticket(shopping_list, video = True)

    out.release()
    cap.release()
    cv2.destroyAllWindows()
