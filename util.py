import logging
import os
from datetime import datetime
import cv2
import glob
import numpy as np


def write_image(out, frame):
    """
    writes frame from the webcam as png file to disk. datetime is used as filename.
    """
    if not os.path.exists(out):
        os.makedirs(out)
    now = datetime.now() 
    dt_string = now.strftime("%H-%M-%S-%f") 
    filename = f'{out}/{dt_string}.png'
    logging.info(f'write image {filename}')
    cv2.imwrite(filename, frame)



def init_cam(width, height):
    """
    setups and creates a connection to the webcam
    """

    logging.info('start web cam')
    cap = cv2.VideoCapture(0)

    # Check success
    if not cap.isOpened():
        raise ConnectionError("Could not open video device")
    
    # Set properties. Each returns === True on success (i.e. correct resolution)
    assert cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    assert cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def get_labels(index):
    """
    As input it receives an array with prediction of the CNN model, the function will
    find the corresponding company for that array and return the company name of the predict logo
    """
    labels = []
    folders = glob.glob ("./images/*")
       
    for folder in folders:
        labels.append(folder.split('/')[2])
    labels.sort()
    return(labels[index])

def find_folder(label):
    """
    When the letter head is recongnized by the CNN and the label (company name)
    is known, this function finds the right folder where the final document
    will be saved
    """
    path = ("/home/fabian/Dokumente")
    dirs = os.listdir(path)
    i = 0
    final_dir = None
    print(label)
    print(dirs)
    for dir in dirs:
        dir2 = os.listdir(f"/home/fabian/Dokumente/{dir}")
        i +=1
        for dir_sub in dir2:
            if dir_sub == label:
                final_dir = dirs[i-1]
                final_path=f'/home/fabian/Dokumente/{final_dir}/{label}/'
    return(final_path)



