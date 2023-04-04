import os
from app.fakefaceimages_detection import testing
from flask import render_template, request


UPLOAD_FOLDER = 'static/upload'

def index():
    return render_template('index.html')


def app():
    return render_template('app.html')


def detectapp():
    if request.method == 'POST':
        f = request.files['image_name']
        filename = f.filename
        # save our image in upload folder
        path = os.path.join(UPLOAD_FOLDER,filename)
        f.save(path) # save image into upload folder
        # get predictions
        detection = testing(path)
        return render_template('fakeface.html',fileupload=True,detection=detection,filename=filename) # POST REQUEST  
    return render_template('fakeface.html',fileupload=False) # GET REQUEST