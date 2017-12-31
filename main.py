from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import tensorflow as tf



model = load_model('./bin/model.h5')

# workound for the keras error using tensorflow as the backend with flask
model._make_predict_function()
graph = tf.get_default_graph()


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    input = np.zeros(46) # will be use to construct input fed into the model

    json = request.get_json(silent=True) # 'silent = True' ignores and returns None instead which is caught with the condition below
    if not json:
        response = jsonify({ 'message' : 'body of the request must be in json format with header application/json. \
                           no json received or syntax error' })
        response.status_code = 400
        return response

    # mandatory keys in the post request
    if not all([key in json and isinstance(json[key], int) for key in ['team1_score', 'team2_score']]):
        response = jsonify({ 'message': 'the scores of each team must be provided with the keys: team1_score, team2_score' })
        response.status_code = 400
        return response

    if json['team1_score'] > json['team2_score']:
        input[0] = 1
    elif json['team2_score'] > json['team1_score']:
        input[1] = 1

    margin = min(15, abs(json['team1_score'] - json['team2_score']))
    if margin != 0:
        # TODO: fix this indexing
        input[2 + margin - 1] = 1

    # period/overtime
    if 'period' in json and 'overtime' in json:
        response = jsonify({ 'message': 'both period and overtime cannot be specified together' })
        response.status_code = 400
        return response
    elif 'period' in json:
        if not isinstance(json['period'], int) or json['period'] not in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]:
            response = jsonify({ 'message': 'period must be of value: 1/1.5/2/2.5/3/3.5/4/4.5. \
                                use "overtime: 1" to indicate overtime' })
            response.status_code = 400
            return response
        input[int(19+2*json['period']-2)] = 1
    elif 'overtime' in json:
        if not isinstance(json['overtime'], int) or json['overtime'] != 1:
            response = jsonify({ 'message': 'use "overtime: 1" to indicate overtime' })
            response.status_code = 400
            return response
        input[27] = 1
    else: # default to halftime
        input[21] = 1

    if 'streak' in json:
        if 'team_streak' not in json or not isinstance(json['team_streak'], int) or json['team_streak'] not in [1, 2]:
            response = jsonify({ 'message': 'team on the streak must be identified by "team_streak: 1/2"' })
            response.status_code = 400
            return response
        input[28 + min(15, int(json['streak']))] = 1
        input[44 + json['team_streak'] - 1] = 1
        
    
    # calculate the prediction
    with graph.as_default():
        input = input.reshape(1, 46) # column vector
        result = model.predict(input)[0].tolist()
        response = jsonify({ 'prediction': result })
        response.status_code = 200
        return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
