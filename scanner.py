"""
This is a pdf scanner which transforms a given picture 
in several steps to a pdf. Finally saves it to a given path.
As input it receives a list of images, the path where to save the file
and label (company)
Most of the code is taken from the following github repository: 
https://github.com/PooryaKhajoui/document-scanner-1/blob/master/README.md
"""

import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from PyPDF2 import PdfReader, PdfFileWriter
import rect


# add image here.
# We can also use laptop's webcam if the resolution is good enough to capture
# readable document content
def image_to_pdf(image_list, path, label):
    """
    Transforms images or a list of images into a pdf and saves it to a given path.
    """
    # today = date.today()
    # date = today.strftime("%b-%d-%Y")
    print("länge liste= ", len(image_list))
    final_image_list = []
    for image in image_list:
        # resize image so it can be processed
        # choose optimal dimensions such that important content is not lost
        image = cv2.resize(image, (1500, 880))


        # creating copy of original image
        orig = image.copy()

        # convert to grayscale and blur to smooth
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #blurred = cv2.medianBlur(gray, 5)

        # apply Canny Edge Detection
        edged = cv2.Canny(blurred, 0, 50)
        orig_edged = edged.copy()

        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        (contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        #x,y,w,h = cv2.boundingRect(contours[0])
        #cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),0)

        # get approximate contour
        for c in contours:
            p = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * p, True)

            if len(approx) == 4:
                target = approx
                break


        # mapping target points to 800x800 quadrilateral
        #approx = rectify(target)
        approx = rect.rectify(target)
        pts2 = np.float32([[0,0],[1131,0],[1131,800],[0,800]])

        M = cv2.getPerspectiveTransform(approx,pts2)
        dst = cv2.warpPerspective(orig,M,(1131,800))

        cv2.drawContours(image, [target], -1, (0, 255, 0), 2)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)


        # using thresholding on warped image to get scanned effect (If Required)
        ret,th1 = cv2.threshold(dst,127,255,cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                    cv2.THRESH_BINARY,11,2)
        th3 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,2)
        ret2,th4 = cv2.threshold(dst,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        #rotate clockwiese
        dst = cv2.rotate(dst, cv2.ROTATE_90_COUNTERCLOCKWISE)
        dst = cv2.flip(dst,0) 
        #create an PIL image from array
        image_1 = Image.fromarray(dst)
        #convert to RBF
        im_1 = image_1.convert('RGB')
        # Append the image to the final_list
        final_image_list.append(im_1)
    #assing first list entry to variabl
    im_1 = final_image_list[0]
    #remove first list entry
    final_image_list.pop(0)

    #get current datetime for file name
    date = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')

    #save all pages as one pdf file
    im_1.save(f'{path}{date}_{label}.pdf', save_all=True, append_images=final_image_list)



