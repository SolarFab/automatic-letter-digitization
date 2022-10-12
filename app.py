"""
This code is creating a website using flask framework in order to automize
the sorting of hardcopy letters.
- On the landing page of the website, you have to take a picture of the 
company's logo 
- the trained CNN model will then predict the company
- in the next step, you have to scan/take pictures of the document and these
will be transformed to a pdf
- finally, the pdf will be saved automatically into the correct document folder
based on the forecast of the CNN model

"""

from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from util import  get_labels, find_folder #write_image, key_action, init_cam,
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras import preprocessing 
from tensorflow.keras.models import load_model
import numpy as np
from scanner import image_to_pdf

#define variables
label = None
image_list = []

# Load pretrained CNN model  
model = load_model("./model/trained_model_logos.h5")  

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')

#assign the right camera 
camera = cv2.VideoCapture(0)

def gen_frames_logo():  # generate frame by frame from camera

    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  
        # fliping the image 
        frame = cv2.flip(frame, 1)

        # draw a rectangle into the frame
        offset = 2
        width = 1000
        x = 1420
        y =580
        cv2.rectangle(img=frame, 
                        pt1=(x-offset,y-offset), 
                        pt2=(x+width+offset, y+width+offset), 
                        color=(0, 255, 0), 
                        thickness=10
                                )   

        if success:
    
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                    
            else:
                pass    

def gen_frames_scan():  # generate frame by frame from camera

    while True:
        # Capture frame-by-frame
        success, frame = camera.read() 
        # fliping the image 
        frame = cv2.flip(frame, 1)

        # draw a rectangle into the frame with the same aspect ratio like a DINA4 page
        offset = 2
        width = 2800
        heigth = 1980
        x = 520
        y =90
        cv2.rectangle(img=frame, 
                        pt1=(x-offset,y-offset), 
                        pt2=(x+width+offset, y+heigth+offset), 
                        color=(0, 255, 0), 
                        thickness=10
        )   

        if success:
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass

@app.route('/')

def index(): #generate the landing page 
    return render_template('landing_page.html')

@app.route('/video_feed')
def video_feed_logo(): #generate the videostream for the logo detection
    return Response(gen_frames_logo(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_scan')
def video_feed_scan(): #generate the videostream for the document scanner
    return Response(gen_frames_scan(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests',methods=['POST','GET'])
def tasks():
    """
    When picture of the logo is taken on website, the logo will be processed through
    the pretrained CNN model and the related company of the logo will be forecasted.
    """
    global camera
    global label
    # Capture frame-by-frame
    success, frame = camera.read() 
    # fliping the image 
    frame = cv2.flip(frame, 1)
    if request.method == 'POST':
        if request.form.get('click') == 'SHOOT':

            #set the dimension of the rectangle in the frame
            offset = 2
            width = 1000
            x = 1420
            y =580
            
            cv2.rectangle(img=frame, # generate rectanle in the frage
                          pt1=(x-offset,y-offset), 
                          pt2=(x+width+offset, y+width+offset), 
                          color=(0, 255, 0), 
                          thickness=10
            ) 
            image = frame[y:y+width, x:x+width, :] # take picture
            image = cv2.resize(image,(224,224)) # resize the image to be processed in the CNN model

            # Use the mobilnet CNN model to predict the logo
            a = preprocessing.image.img_to_array(image, dtype = 'uint8') #transform image to array

            a = np.expand_dims(a, axis = 0) #expand by one dimension (Mobilenet needs 4 dimensions)

            a = mobilenet_v2.preprocess_input(a) # necessary function to preprocesses image-array
  
            logo_nr = np.argmax(model.predict(a))   # will predict the company of the logo as an array
          
            label = get_labels(logo_nr) # returns the company name corresponding to array of the prediction
 
    return render_template('logo_found.html', label = label)



@app.route('/doc-scan',methods=['POST','GET'])
def scan():
    """
        When picture of the logo is taken on website, the logo will be processed through
        the pretrained CNN model and the related company of the logo will be forecasted.
    """
    global camera
    global label    

    # Capture frame-by-frame
    success, frame = camera.read() 
    # fliping the image 
    frame = cv2.flip(frame, 1)
    print(label)
    
    if request.method == 'POST':

        if request.form.get('scan') == 'Scan': # if "scan" button on website is clicked
            print(request.args)
            global image_list

            image = frame #assign the frame to a variable
            image_list.append(image) # create a list of images

        elif request.form.get('click') == 'Save': # if "save" button on website is clicked
            path = find_folder(label) # find the right path where the document should be saved
            image_to_pdf(image_list, path, label) #convert images to pdf and saved it to given path
        elif request.form.get('redo') == 'Again': # if "Again" button on website is clicked
            image_list = [] # list will be emptied/ reset
        elif request.form.get('return') == 'Return to home': # return to landing page
            image_list = [] # list will be emptied/ reset 
               
    elif request.method=='GET':
        return render_template('doc_scan.html')
    return render_template('doc_scan.html')


if __name__ == '__main__':
    app.run(port=5001)

camera.release()
cv2.destroyAllWindows()  