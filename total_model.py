from skimage import io 
import numpy as np
import cv2
import threading
import time

import funiture_detection 
import image_prepeocess 
import room_prediction 
class Captioning_image_model:
    def __init__(self):
        self.Fun_detection_model=funiture_detection.Model()
        self.Room_prediction_model=room_prediction.Model("Room_prediction_model")
        self.Im_Prepro=image_prepeocess.Im_preprocess()
    def predict_room(self):
        self.rooms_list= self.Room_prediction_model.predict(self.Im_Prepro.X_room_predict)
    def detect_object(self):
        self.Fun_detection_model.delete_images(funiture_detection.image_path)
        self.Fun_detection_model.delete_images('runs/detect/exp/')

        self.Im_Prepro.write_img(funiture_detection.image_path) # save image

        self.Fun_detection_model.predict()
        
        self.object_list= self.Fun_detection_model.transform_vec()
    def naming_features(self,object_list):
        if len(object_list)==0:
            return ""
        elif len(object_list)==1:
            return " having "+object_list[0]
        else:
            feature=" having "
            for i in range(len(object_list)-1):
                feature+=object_list[i]+", "
            feature+=object_list[len(object_list)-1]
            return feature
    def caption_image(self, urls):
        
        a=time.time()
        self.Im_Prepro.close()
        self.Im_Prepro.from_urls_to_array(urls)
        download_time=time.time()-a

        t_1=threading.Thread(target=self.predict_room)
        t_2=threading.Thread(target=self.detect_object)
        t_1.start()
        t_2.start()
        t_1.join()
        t_2.join()
        captioning_time=time.time()-download_time-a

        result=[]
        result.append({
            "download_time":download_time,
            "captioning_time":captioning_time
        })
        for i,x in enumerate(self.Im_Prepro.correct_idx):
            result.append({
                "url":urls[x],
                "caption":self.rooms_list[i] + self.naming_features(self.object_list[i])
            }
            )

        return result

total_model=Captioning_image_model()

urls=[
    "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/ba-0111778-lr-1592404733.jpg",
    "https://cdn.mos.cms.futurecdn.net/sbj3Y757EZpEFw4adsVVs8-768-80.jpg",
    "https://th.bing.com/th/id/OIP.ESu8Gs7ZxOcEkaZcnNXk_wHaLH?pid=ImgDet&rs=1",
    "https://th.bing.com/th/id/R15a589377bf4d0a006dd94e1ecc57d77?rik=HGhMpQZHIGtp9A&pid=ImgRaw",
]

print(total_model.caption_image(urls))