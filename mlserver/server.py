from flask import Flask, jsonify
from flask import request
import joblib
from sklearn.linear_model import LinearRegression
import pandas as pd
from datetime import date

app = Flask(__name__)

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

@app.route('/prediction', methods=['POST','GET'])
def prediction_pg():
    if request.method == 'POST':
        genre = request.form.get('genres')
        
        input_msg = "<h1>The genre you selected was: {}</h1>".format(genre)

        model1 = joblib.load('D:/Deep Learning Projects/recommender/mldata/finalized_model.pkl')

        game = model1.predict(genre_dict[genre])

        predicted_sales = '${:,.2f}'.format((game[0]*1000000)*60)

        statement = '''<p>{} game predicted revenue: {}</p><br>
                    <form method="GET" action="/prediction">
                        <input type="submit" value="Go Back">
                    </form>'''.format(genre.upper(), predicted_sales)

        return input_msg + statement


    get_message = '''
    <head> 
        <title>Prediction Server</title>
    </head>
    <body>
        <h1>Genre Based Video Game Revenue Prediction</h1>
        <h2>by Good Games</h2>
        <p>Please enter a video game genre:</p>
        <form method="POST">
            <select name="genres">
                <option value="action">Action</option>
                <option value="action adventure">Action Adventure</option>
                <option value="adventure">Adventure</option>
                <option value="board game">Board Game</option>
                <option value="education">Education</option>
                <option value="fighting">Fighting</option>
                <option value="mmo">Massively Multiplayer</option>
                <option value="music">Music</option>
                <option value="party">Party</option>
                <option value="platform">Platform</option>
                <option value="puzzle">Puzzle</option>
                <option value="racing">Racing</option>
                <option value="rpg">Role Playing</option>
                <option value="sandbox">Sandbox</option>
                <option value="fps">First Person Shooter</option>
                <option value="sim">Simulation</option>
                <option value="sports">Sports</option>
                <option value="strategy">Strategy</option>
                <option value="visual novel">Visual Novel</option>
                <option value="misc">Miscellaneous</option>
            </select>
            <input type="submit" value="Submit"><br>
        </form>

    </body>'''

    return get_message

@app.route('/', methods=['GET'])
def get_samples():
    str1 = "Good Games"
    str2 = "Revenue prediction based on Genre ML Server"
    today = date.today()
    str3 = str(today)
    result = "<HTML><BODY> <p><h1> %s </p></h1> <p> %s </p> <p> %s </p> </BODY> </HTML>" % (str1, str2, str3)
    print(result)
    return result

@app.route('/', methods=['POST'])
def predict():

    sample = {
        "genre" : request.json['genre']
    }
    
    model1 = joblib.load('D:/Deep Learning Projects/recommender/mldata/finalized_model.pkl')

    genre = sample['genre']

    print(genre)

    game = model1.predict(genre_dict[genre])

    predicted_sales = '${:,.2f}'.format((game[0]*1000000)*60)

    statement = "Predicted lifetime revenue for", genre,"game:", predicted_sales
    result = jsonify({'result':str(statement)})

    print(result)
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)