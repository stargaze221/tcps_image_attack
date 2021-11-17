import torch
import numpy as np
import cv2

from models.experimental import attempt_load
from utils.general import non_max_suppression, xyxy2xywh
from utils.plots import plot_one_box, color_list

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class YoloWrapper:
    def __init__(self, model_param_path, device = DEVICE):
        self.model = attempt_load(model_param_path)
        self.model.to(device)
        print('model loadded using the file at ', model_param_path)
        self.color = color_list()

    def get_predictions(self, torch_images):
        result = self.model(torch_images)
        pred = non_max_suppression(result[0])   # Apply NMS
        return pred

    def draw_image_w_predictions(self, torch_images, show=False, wait=1000):
        np_images = torch_images.permute(0,2,3,1).cpu().numpy()
        np_images_uint8 = np.clip(np_images*255, 0, 255).astype(np.uint8)
        torch_images = torch_images.detach()
        pred = self.get_predictions(torch_images)
        
        for i in range(len(pred)):
            detection = pred[i]
            #cv2_images_uint8 = cv2.cvtColor(np_images_uint8[i], cv2.COLOR_RGB2BGR)
            cv2_images_uint8 = np_images_uint8[i]
            np_pred = []
            if len(detection)>0:
                for box in detection:
                    box = box.cpu().numpy()
                    xyxy = box[:4].astype(int)
                    class_int =  int(box[-1])
                    plot_one_box(xyxy, cv2_images_uint8, color=self.color[class_int%10], label=self.model.names[class_int])
                    np_pred.append(box)
            if show:    
                cv2.imshow('prediction'+str(i), cv2_images_uint8)
                cv2.waitKey(wait)
        return cv2_images_uint8, np.array(np_pred)

    def draw_image_w_prediction_and_target(self, torch_image, action):
        cv2_images_uint8, np_pred = self.draw_image_w_predictions(torch_image, show=False, wait=1000)
        
        ## Draw target box ###
        tgt = np.array(action)
        tgt = (tgt + 1)/2  # [-1,1] -> [0,1] with some pertubation.
        tgt = np.clip(tgt, 0, 1)
        x_ctr = int(tgt[0]*448)
        y_ctr = int(tgt[1]*448)
        w = int(200*tgt[2] + 50)
        h = int(200*tgt[3] + 50)

        xyxy = np.array([x_ctr - w/2, y_ctr - h/2, x_ctr + w/2, y_ctr + h/2])
        xyxy = np.clip(xyxy, 5, 448-5)
        plot_one_box(xyxy, cv2_images_uint8, color=(150, 10, 10), label='target')

        return cv2_images_uint8, np.array(np_pred)

    def draw_image_w_prediction_and_target_and_kf_box(self, torch_image, action, kf_box):
        cv2_images_uint8, np_pred = self.draw_image_w_prediction_and_target(torch_image, action)

        kf_box = np.clip(kf_box, 5, 448-5)

        plot_one_box(kf_box, cv2_images_uint8, color=(10, 10, 10), label='kf box')
        return cv2_images_uint8, np_pred

    def draw_image_w_prediction_and_kf_box(self, torch_image, kf_box):
        cv2_images_uint8, np_pred = self.draw_image_w_predictions(torch_image, show=False, wait=1000)

        kf_box = np.clip(kf_box, 5, 448-5)
        plot_one_box(kf_box, cv2_images_uint8, color=(10, 10, 10), label='kf box')
        return cv2_images_uint8, np_pred



if __name__ == "__main__":
    from PIL import Image


    yolo_model = YoloWrapper('yolov5m.pt')

    ### Load the single image ###
    data = np.asarray(Image.open('image_sample.png').convert('RGB'))
    x_image = torch.FloatTensor(data).to(DEVICE).permute(2, 0, 1).unsqueeze(0)/255
    print('x_image', x_image.size(), 'min:', x_image.min(), 'max:', x_image.max())

    ### Use the model ### 
    pred = yolo_model.get_predictions(x_image)

    ### Use the model to draw prediction ### 
    cv2_images, np_pred = yolo_model.draw_image_w_predictions(x_image, True)

    