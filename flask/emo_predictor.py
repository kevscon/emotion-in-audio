import flask
from flask import request
from flask import Flask, request, redirect, url_for
from werkzeug import secure_filename
import pickle
import numpy as np
import os
import librosa
from sklearn import preprocessing

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['wav'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('index'))
    return """
    <!doctype html>
    <title>Upload new File</title>
    <style>
        h1{
            font-family: helvetica;
        }
        a:link, a:visited {
          background-color: #800000;
          color: white;
          padding: 14px 25px;
          text-align: center;
          text-decoration: none;
          display: inline-block;
          font-family: cursive;
          font-size: large;
        }
    </style>
    <body class ="display" style ="background-image: linear-gradient(to right, #ADA996, #F2F2F2, #DBDBDB, #EAEAEA); background-position: initial initial; background-repeat: initial initial;">
    <center><h1>Upload Audio File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    <p>%s</p>
    <br>
    <a href="http://127.0.0.1:5000/predict">Predict Emotion</a>
    </center>
    </body>
    """ % "<br>".join(os.listdir(app.config['UPLOAD_FOLDER'],))



# load emotion labels
with open("../labels.pkl","rb") as f:
    labels = pickle.load(f)

# load model
with open("../emo_pred.pkl","rb") as f:
    model = pickle.load(f)


@app.route("/predict")
def hello():

    files = sorted(os.listdir('uploads/'))
    file = files[-1]

    # process audio file
    # set librosa parametrs
    offset = 0.5 # time (s) to offset audio file start
    duration = 2.5 # selected duration for each file (s)
    sr = 22050 # sample rate (Hz)
    n_mfcc = 13 # number of cepstral coefs to return
    frame_size = 0.025 # window size (s)
    n_fft = int(frame_size*sr) # number of frames
    frame_stride = 0.01 # stride size (s)
    hop_length = int(frame_stride*sr) # number of samples to hop per frame

    # extract features from input file
    y, sr = librosa.load('uploads/' + file, duration=duration, sr=sr, offset=offset)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length) # coefficients
    mfccs = preprocessing.scale(mfccs[1:], axis=1) # scale coefs
    features = mfccs.reshape(mfccs.shape[0] * mfccs.shape[1]) # reshape mfcc matrix into 1-D array


    # calculate predictions
    predictions = model.predict_proba([features])[0]

    # predictions as percentage
    preds = (predictions*100).round()
    # pair emotional class labels with predictions
    probas = list(zip(model.classes_, preds))
    # sort probabilities
    probas.sort(key=lambda x: x[1], reverse=True)

    pred_list = []
    emo_list = []
    pred_lab_list = []
    # create list of top three predictions
    for proba in probas[0:3]:
        pred_list.append(int(proba[1]))
        emo_list.append(proba[0])

    # load emotion image to display with top result
    image = '/static/emotions-' + emo_list[0] + '.jpg'

    # render app in html with variables
    return flask.render_template('emo_predictor.html',
    predictions=probas[0:3],
    emotions=emo_list,
    preds=pred_list,
    emo=emo_list[0],
    image=image
    )


# Start the server, continuously listen to requests.
# We'll have a running web app!

# For local development:
app.run(debug=True)

# For public web serving:
# app.run(host='0.0.0.0')
