import numpy as np 
import flask
from flask import jsonify, request
import praw 
import pickle 
import re 
from nltk.corpus import stopwords

app = flask.Flask(__name__)
app.config["DEBUG"] = True

model = pickle.load(open("final_model.sav", 'rb'))
reddit = praw.Reddit(client_id = '4olKQ-Up2BCKWQ', client_secret = 'sZsEDlw6NzhL3TpD95p1QMeH81E', user_agent = 'Sahil Gupta', username = 'the_stranded_alien', password = 'strandedalien')

def fetch_features(pred_url):
    submission = reddit.submission(url=str(pred_url))
    submission.comments.replace_more(limit=None)
    comment = ''
    for top_comments in submission.comments:
        comment = comment + " " + top_comments.body
    return submission.title, comment, pred_url

def cleanURL(url):
    url = url.replace("/"," ").replace("."," ").replace("_", " ").replace("-", " ")
    url = re.sub(r"[0-9]+", "", url)
    new_url = []
    for u in (url.split()):
        # print(u)
        if u != "reddit" and u != "comments" and u != "india" and u != "https:" and u != "http:" and len(u) > 4 and u != "www" and u != "com":
            new_url.append(u)
    return " ".join(new_url)

def cleaner(sentence):
    stop_words = set(stopwords.words('english'))
    sentence = sentence.lower()
    sentence = re.sub("[^0-9a-z #+_]","",sentence)
    sentence = re.sub("[/(){}\[\]\|@,;]", " ", sentence)
    sentence = ' '.join(word for word in sentence.split() if word not in stop_words)
    return sentence

def cleaned_feature(title, comments, url):
    u = cleanURL(url)
    t = cleaner(title)
    c = cleaner(comments)
    return str(u) + str(t) + str(c)

def make_prediction(pred_url):
    t,c,u = fetch_features(pred_url)
    feature = cleaned_feature(t,c,u)
    feature = np.array(feature).reshape((1,))
    prediction = model.predict(feature)
    return prediction


@app.route('/',methods=['GET'])
def home():
    return jsonify("Flair Prediction Model As Microservice (API)")

@app.route('/predict', methods=['GET'])
def predict():
    if 'post' in request.args:
        Url = request.args['post']
    else:
        return jsonify("Error : No Post URL provided !")
    
    result = make_prediction(str(Url)) 
    
    return jsonify(list(result))


app.run()