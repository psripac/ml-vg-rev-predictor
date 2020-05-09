from flask import Flask, jsonify
from flask import request
import joblib
from sklearn.linear_model import LinearRegression
import pandas as pd
from datetime import date

app = Flask(__name__)

@app.route('/', methods=['GET'])
def get_samples():
    str1 = "Good Games"
    str2 = "Revenue prediction based on Genre ML Server"
    today = date.today()
    str3 = str(today)
    result = "<HTML><BODY <p> %s </p> <p> %s </p> <p> %s </p> </BODY> </HTML>" % (str1, str2, str3)
    print(result)
    return result

@app.route('/', methods=['POST'])
def predict():

    action = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    action_adventure = [[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    adventure = [[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    board_game = [[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    education = [[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    fighting = [[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    mmo = [[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    misc = [[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]]
    music = [[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]]
    party = [[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]]
    platform = [[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]]
    puzzle = [[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]]
    racing = [[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]]
    rpg = [[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]]
    sandbox = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]]
    fps = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]]
    sim = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]]
    sports = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]]
    strategy = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]]
    visual = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]]

    genre_dict = {
        "action" : action,
        "action adventure" : action_adventure,
        "adventure" : adventure,
        "board game" : board_game,
        "education" : education,
        "fighting" : fighting,
        "mmo" : mmo,
        "misc" : misc,
        "music" : music,
        "party" : party,
        "platform" : platform,
        "puzzle" : puzzle,
        "racing" : racing,
        "rpg" : rpg,
        "sandbox" : sandbox,
        "fps" : fps,
        "sim" : sim,
        "sports" : sports,
        "strategy" : strategy,
        "visual novel" : visual
    }

    sample = {
        "genre" : request.json['genre']
    }

    model1 = joblib.load('D:/Deep Learning Projects/recommender/mldata/finalized_model.pkl')

    genre = sample['genre']

    game = model1.predict(genre_dict[genre])

    predicted_sales = '${:,.2f}'.format((game[0]*1000000)*60)

    statement = "Predicted lifetime revenue for", genre,"game:", predicted_sales
    result = jsonify({'result':str(statement)})

    print(result)
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)