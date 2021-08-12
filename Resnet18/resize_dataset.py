import os
import threading
import cv2
import time
from tqdm import tqdm

DATA_DIR = "/home/rvp/Face-Lane-Detection/drive_download/apolloscape/ColorImage_road02/ColorImage/"
ANNOT_DIR = "/home/rvp/Face-Lane-Detection/drive_download/apolloscape/Labels_road02/Label/"
NEW_DATA_FOLDER = "/home/rvp/Face-Lane-Detection/drive_download/apolloscape/ColorImage_road02/Resized_Data/"
NEW_ANNOT_FOLDER = "/home/rvp/Face-Lane-Detection/drive_download/apolloscape/Labels_road02/Resized_Data/"

os.makedirs(NEW_DATA_FOLDER,exist_ok=True)
os.makedirs(NEW_ANNOT_FOLDER,exist_ok=True)

def resize_images(OLD_PATH,NEW_PATH):
    for root,dirs,paths in tqdm(os.walk(OLD_PATH)):
        if len(paths) == 0:
            continue 
        
        for path in paths:
            try:
                image = cv2.imread(os.path.join(root,path))
                image = cv2.resize(image,(352,288))
                new_path = os.path.join(NEW_PATH,root.split(OLD_PATH)[1],path)
                os.makedirs(os.path.dirname(new_path),exist_ok=True)
                cv2.imwrite(new_path,image)
            except:
                pass


# resize_images(ANNOT_DIR,NEW_ANNOT_FOLDER)

t = threading.Thread(target=resize_images,args= (DATA_DIR,NEW_DATA_FOLDER,))
t.setDaemon(True)
t.start()

t1 = threading.Thread(target=resize_images,args= (ANNOT_DIR,NEW_ANNOT_FOLDER,))
t1.setDaemon(True)
t1.start()

t.join()
t1.join()

print("Completed Successfully")