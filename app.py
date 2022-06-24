from __future__ import print_function
from flask import Flask
from flask import render_template,request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings('ignore')

# creates a Flask application, named app
app = Flask(__name__)

# a route where we will display a welcome message via an HTML template
@app.route("/")
def hello():
    return render_template('homepage.html')

@app.route("/predata")
def predata():
    return render_template('rec.html')

@app.route("/predict",methods =["GET", "POST"])
def predict():
    if request.method == "POST":
        nitrogen = request.form["nitrogen"] 
        phosporous = request.form["phosporous"]
        humidity = request.form["humidity"]
        temp = request.form["temp"]
        ph = request.form["ph"]
        rainfall = request.form["rainfall"]
        potassium = request.form["potassium"]
    df = pd.read_csv(r'C:\Users\91936\Downloads\Crop-Recommender-main\Crop-Recommender-main/Crop_recommendation.csv')

    features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
    target = df['label']
#features = df[['temperature', 'humidity', 'ph', 'rainfall']]
# labels = df['label']

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

    acc = []
    model = []


    

    RF = RandomForestClassifier(n_estimators=20, random_state=0)
    RF.fit(Xtrain,Ytrain)

    predicted_values = RF.predict(Xtest)

    x = metrics.accuracy_score(Ytest, predicted_values)
    acc.append(x)
    model.append('RF')

# print(classification_report(Ytest,predicted_values))

# Cross validation score (Random Forest)
    score = cross_val_score(RF,features,target,cv=5)
    score

    data = np.array([[nitrogen,phosporous,humidity,temp,ph,rainfall,potassium]])
    recommend = RF.predict(data)
    print(recommend[0])
    return render_template('rec.html',output=recommend[0])

# run the application
if __name__ == "__main__":
    app.run(debug=True)
