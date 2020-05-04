from flask import Flask, render_template, url_for, request, redirect,send_file, Response
from flask_sqlalchemy import SQLAlchemy
from flask import json,jsonify
from datetime import datetime
from io import BytesIO
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
import pandas as pd
from datetime import datetime
import io
import cv2
import numpy as np
import dlib
from math import hypot
from scipy.stats import mode
import pickle
from tensorflow.keras.models import load_model


app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True

db = SQLAlchemy(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

df = pd.DataFrame(columns=['Name','Date_Time'])

complete_df = pd.DataFrame()

known_face_encodings = []
known_face_names = []
Path_List = []
temp_list= []

IMG_SIZE=160

#values_list = list()


# knn = pickle.load(open('knn.pickle','rb'))
# svc_model = pickle.load(open('svc_model.pickle','rb'))  
# CNN_model = load_model('CNN_model.h5')
# Vgg_model = load_model('Vgg_model.h5')
# RF = pickle.load(open('RF.pickle','rb'))
    
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

def VGG_frame_shape(img):
    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img_new = img.reshape((-1,img.shape[0],img.shape[1],3))
    return img_new

def CNN_frame_shape(img):
    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img_new = img.reshape((-1,img.shape[0],img.shape[1],1))
    return img_new

def func_CNN(pred):
    chk=pred[0][0]
    if chk < 0.01:
        result = "fake"
    else:
        result = "real"
    return result

def func(pred):
    if pred==0:
        return "fake"
    else :
        return "real"
    
def frame_shape(img):
    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img_new = img.reshape((1,img.shape[0]*img.shape[1]))
    return img_new

def Convert_Image(my_img):
    images = list()
    my_img = cv2.cvtColor(my_img,cv2.COLOR_BGR2GRAY)
    
    images.append(my_img)
    
    vgg_img = cv2.cvtColor(my_img,cv2.COLOR_BGR2RGB)
    
    images.append(vgg_img)
    
    return images

def Process_Roi(my_img):
    roi = cv2.resize(my_img, (160, 160))
    roi=roi/255
    
    return roi    

def KNN_MODEL(roi):
    
    f_img=frame_shape(roi)
    
    pred = knn.predict(f_img)
    
    var=func(pred)

    return var

def SVM_MODEL(roi):
    
    f_img=frame_shape(roi)
    
    svc_pred = svc_model.predict(f_img)
    
    var2=func(svc_pred)
    
    return var2

def RF_MODEL(roi):
    
    f_img=frame_shape(roi)
    
    pred = RF.predict(f_img)
    
    var=func(pred)

    return var

def CNN_MODEL(roi):
    
    f_img=CNN_frame_shape(roi)
    
    pred = CNN_model.predict(f_img)

    var3=func(pred)
    
    return var3

def VGG_MODEL(roi):
    f_img=VGG_frame_shape(roi)
    
    pred = Vgg_model.predict(f_img)

    var4=func_CNN(pred)
    
    return var4
 

def Prediction_Models(my_img,vgg_img):
    predictions=dict()
    
    roi = Process_Roi(my_img)
    roi2 = Process_Roi(vgg_img)
    
    var = KNN_MODEL(roi)
    var2 = SVM_MODEL(roi)
    var3 = CNN_MODEL(roi)
    var4 = VGG_MODEL(roi2)
    var5 = RF_MODEL(roi)
    
    
    predictions['KNN'] = var
    predictions['SVM'] = var2
    predictions['CNN'] = var3
    predictions['VGG'] = var4
    predictions['RF'] = var5
    
    return predictions

def Integrated_Output(img):
    images_list = Convert_Image(img)
    
    my_img = images_list[0]
    vgg_img = images_list[1]
    
    final_results = Prediction_Models(my_img,vgg_img)
    
    return final_results

# def Final_Output_Result(results_after_prediction):
#     for values in results_after_prediction.values():
#         values_list.append(values)
#         final_prediction_mode = mode(values_list)[0]
#     return final_prediction_mode[0]

def Final_Output_Of_Mode(results_after_prediction):
    counter_real=0
    counter_fake=0
    temp_list= list()
    for values in results_after_prediction.values():
        temp_list.append(values)
    for x in range(len(temp_list)):
        
        if temp_list[x]=='real':
            counter_real+=1
        elif temp_list[x]=='fake':
            counter_fake+=1
    # print("counter_fake",counter_fake)
    # print("counter_real",counter_real)
    if counter_real == 2 and counter_fake == 2:
        result = 'real'  
    else:
        final_prediction_mode = mode(temp_list)[0]
        result = final_prediction_mode[0] 
    return result


class FileContent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))
    date_time = db.Column(db.String(300))
    data = db.Column(db.LargeBinary)

    # data = db.Column(db.LargeBinary)

    

    def __repr__(self):
        return '<Task %r>' % self.id

def Load_Img(our_path):
    
    DATA_DIR=our_path
    CATEGORIES_Folder=["images"]

    for category in CATEGORIES_Folder:
        path=os.path.join(DATA_DIR,category) 
        print(path)

        for img in os.listdir(path):
            
            path_of_img = path+"/"+img
            
            name_to_save = img.split('.')[0]

            temp_list.append([name_to_save,path_of_img])

    Path_List.append(temp_list)
    #print(Path_List)

    return Path_List


@app.route('/')
def index():

    DATA_DIR=APP_ROOT
    CATEGORIES_Folder=["images"]

    for category in CATEGORIES_Folder:
        path=os.path.join(DATA_DIR,category) 
        print(path)

        for img in os.listdir(path):
            
            path_of_img = path+"/"+img
            name_to_save = img.split('.')[0]
            temp_list.append([name_to_save,path_of_img])
            known_face_names.append(name_to_save)
            picture_of_me = face_recognition.load_image_file(path_of_img)
            my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

            known_face_encodings.append(my_face_encoding)

    Path_List.append(temp_list)
    print(known_face_names)
    print(known_face_encodings)


    return render_template('index.html')


@app.route("/upload", methods=['POST'])
def upload():

    req_name = request.form.get('PersonName')

    print(req_name)

    target = os.path.join(APP_ROOT, 'images')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
      
        filename = file.filename
        ext = filename.split('.')[-1]
        print(ext)

        fn = req_name + '.' + ext 

        print(fn)

        destination = "/".join([target, fn])

        print(destination)
        file.save(destination)

        picture_of_me = face_recognition.load_image_file(destination)

        try:        
            my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

            known_face_encodings.append(my_face_encoding)

            known_face_names.append(req_name)
	

        except Exception as e:
            print("Exception in Encoding Uploading Files.")
            print(e)

    return render_template("complete.html")

@app.route("/upload_Image", methods=['POST'])
def upload_Image():
    retJson = {}

    CATEGORIES_Folder="images"
    location= os.path.join(APP_ROOT,CATEGORIES_Folder)

    try:
        # postedDate=request.get_json()
        # print(postedDate)
        #name = postedDate['user_name']

        name = request.form['user_name']
       

        imagefile = request.files.get('Image', '')

        print(name)
        
        print(imagefile)

        filename = imagefile.filename
        ext = filename.split('.')[-1]
        print(ext)
        fn = name + '.' + ext 


        try:
            location = os.path.join(location, fn)
            imagefile.save(location)
                
            picture_of_me = face_recognition.load_image_file(location)

            my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

            known_face_encodings.append(my_face_encoding)

            known_face_names.append(name)

            retJson = {"status":200,"msg":"You've successfully Added the person"}
                
        except Exception as e:
            print(e)
            retJson = {"status" : 400,"msg": "Image cannot be saved"}
            return retJson

            

    except Exception as e:
        print(e)
        retJson = {"status" : 302,"msg": "User is already present in the Data Base"}

    return jsonify(retJson)



def Prediction_Of_Image(unknown_picture):
    
    try:
        print("KNOWN FACE ENCODING 01 ----- ",known_face_encodings[0])
        # Find all the faces and face enqcodings in the frame of video
        unknown_face_locations = face_recognition.face_locations(unknown_picture)
        unknown_face_encoding = face_recognition.face_encodings(unknown_picture, unknown_face_locations)

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(unknown_face_locations, unknown_face_encoding):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

    except Exception as e:
        print(e)
        print("It's not a picture of me!")
        name = "Unknown!"
            
    return name


@app.route('/predict', methods=['POST'])

def predict():
    if request.method == "POST":

        if request.files:

            complete_df = pd.DataFrame()

            filestr = request.files['predict_file'].read()

            
            #convert string data to numpy array
            npimg = np.fromstring(filestr, np.uint8)
         
            img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
      
            img_to_be_pred = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 

            print("new image q nh arhiiiiiiiiiiiiiiiiii")
            print(img_to_be_pred)
            
            results_after_prediction = Integrated_Output(img_to_be_pred)

            last_prediction = Final_Output_Of_Mode(results_after_prediction)
            print(last_prediction)

            if last_prediction == "real":

                output = Prediction_Of_Image(img_to_be_pred)
                
                print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO",output)
                file = request.files['predict_file']
                if output != "Unknown!":
                    

                    now = datetime.now()
                
                    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
                    print("date and time:",date_time)
                    
                    present_date_time = date_time
                    addFile = FileContent(name=output, date_time=present_date_time,data=file.read())
                    print("------------------------------------")
                    
                    db.session.add(addFile)
                    db.session.commit()

                    df.loc[0,'Name'] = output
                    df.loc[0,'Date_Time'] = present_date_time
                    print(df)
                    

                    frame = [complete_df,df]
                    complete_df = pd.concat(frame,ignore_index=True)

                    print(complete_df)
                    print(output)    
		     
            else:
                output = "Don't mess with system!!Error 404"

    return render_template('result.html',text = output)

@app.route('/predict_Image', methods=['POST'])
def predict_Image():
    complete_df = pd.DataFrame()

    filestr = request.files['Image'].read()

    #filestr = request.form.get('Image')
    
    npimg = np.fromstring(filestr, np.uint8)
         
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
      
    img_to_be_pred = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    output = Prediction_Of_Image(img_to_be_pred)
    
    file = request.files.get('Image', '')
    if output != "Unknown!" and output != "Unknown" :
        
        now = datetime.now()
    
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        print("date and time:",date_time)
        
        present_date_time = date_time
        addFile = FileContent(name=output, date_time=present_date_time,data=file.read())
        print("------------------------------------")
        
        db.session.add(addFile)
        db.session.commit()

        df.loc[0,'Name'] = output
        df.loc[0,'Date_Time'] = present_date_time
        print(df)
        

        frame = [complete_df,df]
        complete_df = pd.concat(frame,ignore_index=True)

        retJson = {"status":200,"msg":"Successfully Predicted","output":output}
    else:
        
        retJson = {"status":404,"msg":"Don't mess with system!!Error 404","output":output}

    return jsonify(retJson)
            
    


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, frame = self.video.read()
        rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        print("KNOWN FACE ENCODINGS YOOOOOOUUUUU!!",known_face_encodings)

        print("KNOWN FACE NAMES HEHEHEHEHEHEHE",known_face_names)
        
        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()



def gen(camera):

    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/download')
def download():
    file_data_d = FileContent.query.filter_by(id=1).first()
    return send_file(BytesIO(file_data_d.data),attachment_filename='Image.jpg',as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
