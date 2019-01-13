from flask import Flask, render_template, request
import json
from flask_cors import CORS
import numpy as np
import cv2                              # Library for image processing
from math import floor
import threading
from http.server import BaseHTTPRequestHandler,HTTPServer
from socketserver import ThreadingMixIn
from io import StringIO,BytesIO
from PIL import Image
import time
import logging
import face_recognition
from random import randint


app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/shirt.html')
def plot():
    return render_template('shirt.html')
@app.route('/pant.html')
def ploty():
    return render_template('pant.html')
capture=None


import requests

url = "http://localhost:3000//home/update_customer"


def call_api(data,query):
    querystring = {"data":data,"q":query}
    headers = {
        'cache-control': "no-cache",
        'postman-token': "667e2fa8-97a6-4f9f-4280-b447c1f4177d"
        }
    response = requests.request("POST", url, headers=headers, params=querystring)
    print(response.text)


class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        face_det = True
        self.send_response(200)
        self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()
        shirtno = 1
        pantno = 1        
        ih=1
        i=pantno
        hand_position_x = 0
        hand_position_y = 0
        hand_movement_count = 0
        q = 1
        a = q
       
        obama_image = face_recognition.load_image_file("vaibhav.jpg")
        obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
        s_image = face_recognition.load_image_file("shubham.png")
        s_face_encoding = face_recognition.face_encodings(s_image)[0]
        #n_image = face_recognition.load_image_file("nalin.jpg")
        #n_face_encoding = face_recognition.face_encodings(n_image)[0]
       # sumit_image = face_recognition.load_image_file("sumit.jpg")
       # sumit_face_encoding = face_recognition.face_encodings(sumit_image)[0]
        # Create arrays of known face encodings and their names
        known_face_encodings = [
            obama_face_encoding,
            s_face_encoding        ]
        known_face_names = [
            "Vaibhav Maheshwari",
            "Shubham Chintalwar"
        ]

        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []

        while True:
            try:
                q = q + 1 
                rc,img = capture.read()
                if not rc:
                    continue
                imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                imgarr=["1.png" , "2.png",'s1.png','s2.png' , "s3.png"]
                

               

                #ih=input("Enter the shirt number you want to try")
                imgshirt = cv2.imread(imgarr[ih],1) #original img in bgr
                # if ih==0:
                #     shirtgray = cv2.cvtColor(imgshirt,cv2.COLOR_BGR2GRAY) #grayscale conversion
                #     ret, orig_masks_inv = cv2.threshold(shirtgray,200 , 255, cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
                #     orig_masks = cv2.bitwise_not(orig_masks_inv)

                # else:
                shirtgray = cv2.cvtColor(imgshirt,cv2.COLOR_BGR2GRAY) #grayscale conversion
                ret, orig_masks = cv2.threshold(shirtgray,0 , 255, cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
                orig_masks_inv = cv2.bitwise_not(orig_masks)
        
                origshirtHeight, origshirtWidth = imgshirt.shape[:2]
               
                imgarr=["pant7.jpg",'pant21.png']
        #i=input("Enter the pant number you want to try")
                imgpant = cv2.imread(imgarr[i-1],1)
                imgpant=imgpant[:,:,0:3]#original img in bgr
                pantgray = cv2.cvtColor(imgpant,cv2.COLOR_BGR2GRAY) #grayscale conversion
                if i==0:
                    ret, orig_mask = cv2.threshold(pantgray,100 , 255, cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
                    orig_mask_inv = cv2.bitwise_not(orig_mask)
                else:
                    ret, orig_mask = cv2.threshold(pantgray,50 , 255, cv2.THRESH_BINARY)
                    orig_mask_inv = cv2.bitwise_not(orig_mask)
                origpantHeight, origpantWidth = imgpant.shape[:2]

                face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                #hand_cascade = cv2.CascadeClassifier('fist.xml')

                
                img = cv2.flip(img,180)

                height = img.shape[0]
                width = img.shape[1]
                if face_det:
                    small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

                    rgb_small_frame = small_frame[:, :, ::-1]
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)

                    face_names = []

                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        name = "Buddy"

                        if True in matches:
                            first_match_index = matches.index(True)
                            name = known_face_names[first_match_index]
                        #api call to update name
                        call_api(name, "name")
                        print(name)
                        face_det = False
                        break
        #img = cv2.resize(img[:,:,0:3],(1000,1000), interpolation = cv2.INTER_AREA)
              #  cv2.namedWindow("img",cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty('img',cv2.WND_PROP_FULLSCREEN,cv2.cv.CV_WINDOW_FULLSCREEN)
               # cv2.resizeWindow("img", (int(width*3/2), int(height*3/2)))
                

                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces=face_cascade.detectMultiScale(gray,1.3,5)
                if len(faces) == 0:
                   # call_api("Buddy", "name")
                    face_det = True

                #gray_hands_left = gray[0:100, 900:1000]
                #gray_hands_right = gray[0:100, 0:100]
                ## change to a box with certain height and width
                cv2.rectangle(img, (110,110) , (210,210) , (255,255,255), 2)
                cv2.rectangle(img, (900,110) , (1000,210) , (255,255,255) ,2 )
                #print(q)
                if (q % 2) == 0:
                    left1 = img[110:210, 110:210]
                    left1 = np.int32(left1)
                    right1 = img[110:210 , 900:1000]
                    right1 = np.int32(right1)
                    #cv2.imwrite("1.bmp", img[0:100, 0:100])     # save frame as JPEG file
                elif (q % 2) != 0:
                    #cv2.imwrite("2.bmp" , img[0:100, 0:100])
                    left2 = img[110:210, 110:210]
                    left2 = np.int32(left2)
                    left = left2 - left1

                    right2 = img[110:210 , 900:1000]
                    right2 = np.int32(right2)
                    right = right2 - right1

                    lval = np.mean(left)
                    rval = np.mean(right)
                    print(lval)
                    print(rval)
                    if abs(lval) > 4: 
                        if(q - a > 20):
                            ih = int((ih-1)%5)
                            print(ih)
                            call_api( (ih+1)*(10) + 5 , "mvleft")
                            a = q
                    if abs(rval) > 4:
                        if(q - a > 20):
                            ih = int((ih + 1) % 5)
                            print(ih)
                            call_api((ih+1)*(10) + 5 , "mvright")
                            a = q 



                #hands = hand_cascade.detectMultiScale(gray_hands_right,1.3,5)
                #for (x,y,w,h) in hands:
                #    print(x,y)
                #    if x> 500:
                #        print("api call to left carousel") 
                #    else:
                #        print("api call to right carousel") 
                #    print(x,y)        
                #    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 00), 2)
            #print("hello world")
            # if x > hand_position_x:
            #     hand_position_x = x
            #     hand_movement_count += 1
            # if hand_movement_count == 3:
            #     hand_position_x = 0
            #     hand_movement_count =0
                    #print("Swipe right")
                    #ih = int((ih+1)%4)

                for (x,y,w,h) in faces:
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    #cv2.rectangle(img,(100,200),(312,559),(255,255,255),2)
                    pantWidth =  3 * w  #approx wrt face width
                    pantHeight = pantWidth * origpantHeight / origpantWidth #preserving aspect ratio of original image..
     
            # Center the pant..just random calculations..
                    if i==1:
                        x1 = x-w
                        x2 =x1+3*w
                        y1 = y+5*h
                        y2 = y+h*10
                    elif i==2:
                        x1 = x-w/2
                        x2 =x1+2*w
                        y1 = y+4*h
                        y2 = y+h*9
                    else :
                        x1 = x-w/2
                        x2 =x1+5*w/2
                        y1 = y+5*h
                        y2 = y+h*14
                    # Check for clipping(whetehr x1 is coming out to be negative or not..)
                    if x1 < 0:
                        x1 = 0
                    if x2 > img.shape[1]:
                        x2 =img.shape[1]
                    if y2 > img.shape[0] : 
                        y2 =img.shape[0]
                    if y1 > img.shape[0] : 
                        y1 =img.shape[0]
                    if y1==y2:
                        y1=0  
                    temp=0
                    if y1>y2:
                        temp=y1
                        y1=y2
                        y2=temp
                    # Re-calculate the width and height of the pant image(to resize the image when it wud be pasted)
                    pantWidth = abs(x2 - x1)
                    pantHeight = abs(y2 - y1)
                    #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
                    # Re-size the original image and the masks to the pant sizes
                    pant = cv2.resize(imgpant, (pantWidth,pantHeight), interpolation = cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
                    mask = cv2.resize(orig_mask, (pantWidth,pantHeight), interpolation = cv2.INTER_AREA)
                    mask_inv = cv2.resize(orig_mask_inv, (pantWidth,pantHeight), interpolation = cv2.INTER_AREA)
                   # take ROI for pant from background equal to size of pant image
                    roi = img[y1:y2, x1:x2]
                        # roi_bg contains the original image only where the pant is not
                        # in the region that is the size of the pant.
                    num=roi
                    roi_bg = cv2.bitwise_and(roi,num,mask = mask_inv)
                        # roi_fg contains the image of the pant only where the pant is
                    roi_fg = cv2.bitwise_and(pant,pant,mask = mask)
                    # join the roi_bg and roi_fg
                    dst = cv2.add(roi_bg,roi_fg)
                        # place the joined image, saved to dst back over the original image
                    #img[y1:y2, x1:x2] = dst
                    
            #|||||||||||||||||||||||||||||||SHIRT||||||||||||||||||||||||||||||||||||||||
                    
                    shirtWidth =  3 * w  #approx wrt face width
                    shirtHeight = shirtWidth * origshirtHeight / origshirtWidth #preserving aspect ratio of original image..
                    # Center the shirt..just random calculations..
                    x1s = x-w
                    x2s =x1s+3*w
                    y1s = y+h
                    y2s = y1s+h*4
                    # Check for clipping(whetehr x1 is coming out to be negative or not..)
                    
                    if x1s < 0:
                        x1s = 0
                    if x2s > img.shape[1]:
                        x2s =img.shape[1]  
                    if y2s > img.shape[0] : 
                        y2s =img.shape[0]
                    temp=0
                    if y1s>y2s:
                        temp=y1s
                        y1s=y2s
                        y2s=temp
                    # Re-calculate the width and height of the shirt image(to resize the image when it wud be pasted)
                    shirtWidth = abs(x2s - x1s)
                    shirtHeight = abs(y2s - y1s)
                    # Re-size the original image and the masks to the shirt sizes
                    shirt = cv2.resize(imgshirt, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
                    mask = cv2.resize(orig_masks, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
                    masks_inv = cv2.resize(orig_masks_inv, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
                    # take ROI for shirt from background equal to size of shirt image
                    rois = img[y1s:y2s, x1s:x2s]
                        # roi_bg contains the original image only where the shirt is not
                        # in the region that is the size of the shirt.
                    num=rois
                    roi_bgs = cv2.bitwise_and(rois,num,mask = masks_inv)
                    # roi_fg contains the image of the shirt only where the shirt is
                    roi_fgs = cv2.bitwise_and(shirt,shirt,mask = mask)
                    # join the roi_bg and roi_fg
                    dsts = cv2.add(roi_bgs,roi_fgs)
                    img[y1s:y2s, x1s:x2s] = dsts 
                    imgcart = cv2.imread("cart.png",1)

                    rows,cols,channels = imgcart.shape
                    roio = img[0:rows, 0:cols]
                    

                    img2grayo = cv2.cvtColor(imgcart,cv2.COLOR_BGR2GRAY)
                    reti, masko = cv2.threshold(img2grayo, 10, 255, cv2.THRESH_BINARY)
                    mask_invo = cv2.bitwise_not(masko)

                    img1_bg = cv2.bitwise_and(roio,roio,mask = mask_invo)
                    img2_fg = cv2.bitwise_and(imgcart,imgcart,mask = masko)

                    dstq = cv2.add(img1_bg,img2_fg)
                    img[0:rows, 0:cols ] = dstq
                    break
                #cv2.imshow("input",imgRGB)
                
                imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                jpg = Image.fromarray(imgRGB)
                tmpFile = BytesIO()
                jpg.save(tmpFile,'JPEG')
                self.wfile.write("--jpgboundary".encode())
                self.send_header('Content-type','image/jpeg')
                self.send_header('Content-length',str(tmpFile.getbuffer().nbytes))
                self.end_headers()
                jpg.save(self.wfile,'JPEG')
                time.sleep(0.01)
            except KeyboardInterrupt:
                break
        return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
def main():
    global capture
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1000); 
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000);
    global img
    try:
        server = ThreadedHTTPServer(('localhost', 8087), CamHandler)
        print("server started")
        server.serve_forever()
    except KeyboardInterrupt:
        capture.release()
        server.socket.close()

@app.route("/trycam" , methods=['GET'])
def trycam():
    main()
    

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=5000)
