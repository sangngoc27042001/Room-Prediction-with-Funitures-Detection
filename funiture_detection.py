import glob
import os
import re

import numpy as np

import detect

image_shape = 416
image_path = 'Resources/images/'
save_dir = 'runs/detect/exp/'
num_classes = 9
classes = [
    "tv",
    "conditioner",
    "sofa",
    "chair",
    "table",
    "view",
    "fan",
    "tv_shelf",
    "fridge"
]


class Model:
    def __init__(self):
        # Navigate to yolov5 folder
        # %cd ~/../content/drive/MyDrive/Colab Notebooks/Furniture_detection/yolov5
        pass

    def predict(self):
        # os.system('python detect.py --weights runs/train/yolov5s_results4/weights/best.pt --img 416 --conf 0.4 --source {}'.format(image_path))
        detect.run()

    def delete_images(self, file_path):
        files = glob.glob(file_path + '*')
        count = len(files)
        for file in files:
            os.remove(file)
        print('{} files deleted'.format(count))

    def transform_vec(self) -> list:
        obj_list = []
        directory=glob.glob(save_dir + "*.txt")
        directory.sort()
        for file in directory:
            # print(file)
            output = np.zeros((1, num_classes))
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
                #   print(content)
                content = content[1:-1]
                z = re.split(r",", content)
                for ele in z:
                    # print(int(id))
                    try:
                        id = re.sub('[\.\s]', '', ele)
                        output[0][int(id)] = 1
                    except:
                        pass
                # print(output[0])
            obj_list.append(output[0].tolist())
            # with open(file, "w") as f:
            #   f.write(str(output[0]))
            #   print(file + ' written successfully')
        #return obj_list
        return [self.convert_obj_list(i) for i in obj_list]

    def convert_obj_list(self,a):
        b=[]
        for i,x in enumerate(a):
            if x == 1:
                b.append(classes[i])
        return b
