from flask import Flask, render_template, request, redirect, url_for,jsonify
import os
import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
import numpy as np
app=Flask(__name__)

UPLOAD_FOLDER='D:/assignment/PlantVillage-Dataset-master/PlantVillage-Dataset-master/flask_app/static/images'

app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER



# def detect(frame):





# print(f)
# print(f[-1])




@app.route('/',methods=['GET', 'POST'])
def index():
  
    if (request.method == 'POST'):
        f = request.files['inputfiles']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))





    return render_template('index.html')

    
@app.route('/predict',methods=['GET', 'POST'])
def predict():
    model=load_model("improvement_vgg16-00000010.hdf5")
    f=glob.glob(UPLOAD_FOLDER+'/*')
  
    if (request.method == 'GET'):
        label=['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
        'Blueberry___healthy',
        
        'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
       
        'Tomato___Late_blight',
        
        'Tomato___Leaf_Mold',
        
        'Tomato___Septoria_leaf_spot',
       
        'Tomato___Spider_mites Two-spotted_spider_mite',
        
        'Tomato___Target_Spot',
        
         'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
         'Tomato___Tomato_mosaic_virus',
        
        'Tomato___healthy']

        try:
            f=glob.glob(UPLOAD_FOLDER+'/*')
            f=f[-1]
        
            img=cv2.imread(f)
            img=cv2.resize(img,(128,128))
        
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            print(img.shape)
            img=preprocess_input(img)
            
            #scaling to 0 to 1 range 
            if(np.max(img)>1):
                    img = img/255.0
            img=np.array([img])
            # print(img)
            
            prediction=model.predict(img)
            print(prediction>0.7)

            prediction=np.argmax(prediction)
            print(label[prediction])
            print("Prediction:",prediction)
            for f in os.listdir(UPLOAD_FOLDER):
                os.remove(os.path.join(UPLOAD_FOLDER, f))
            return render_template('index.html',prediction=label[prediction],result=prediction)
        except:
            return render_template('index.html',prediction="No Image Found",result="no data found")
    
    return render_template('index.html')


  

        
            


    
    


if __name__ == '__main__':
    app.run(debug=True)


