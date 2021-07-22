from skimage import io 
import numpy as np
import cv2
import threading
mutex = threading.Lock()
class Im_preprocess:
    correct_idx=[] #index in urls that have been downloaded normally
    X_room_predict=[]
    img_list=[]
    def from_urls_to_array(self,urls):
        threads=[]
        for i in range(len(urls)):
            t=threading.Thread(target=self.load_one_image,args=[urls,i])
            t.start()
            threads.append(t)
            
        for thread in threads:
            thread.join()
        self.X_room_predict=np.array(self.X_room_predict)
    
    def load_one_image(self,urls,i):
        try:
            img=io.imread(urls[i])
            img=img[:,:,0:3]
            
            mutex.acquire() #mutex lock
            self.X_room_predict.append(cv2.resize(img,(240,240))/255.0)
            self.img_list.append(cv2.resize(img,(416,416)))
            self.correct_idx.append(i)
            mutex.release() #mutex release
        except:
            pass
    
    def write_img(self, path):
        for i in range(len(self.img_list)):
            io.imsave( path+format(i,'04d')+".png",self.img_list[i])
    def close(self):
        self.correct_idx=[]
        self.X_room_predict=[]
        self.img_list=[]
