from flask import Flask, request, jsonify
import tensorflow as tf

# layers
input_num_units = 5
hidden_num_units = 3
output_num_units = 2

x = tf.placeholder(tf.float32, [None, input_num_units])

seed = 128
weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed), name='weights-hidden'),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed), name='weights-output')
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed), name='biases-hidden'),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed), name='biases-output')
}

hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)

output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
result = tf.nn.softmax(output_layer)

# load in the graph
sess = tf.Session()
saver = tf.train.Saver({ 'weights-hidden': weights['hidden'], 'weights-output': weights['output'],
                        'biases-hidden': biases['hidden'], 'biases-output': biases['output'] })
saver.restore(sess, './bin2/model.ckpt')

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # default to midway through the game is period is not provided
        period = None
        if 'period' not in request.args:
            period = 2
        elif float(request.args.get('period')) not in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]:
            raise Exception('invalid period')

        if len(request.args) == 0:
            raise Exception('no parameters provided')
        elif any(x not in request.args for x in ['home', 'away', 'score1', 'score2']):
            raise Exception('Bad parameters')

        input = [[hash(request.args.get('home')), hash(request.args.get('away')),
                  request.args.get('score1'), request.args.get('score2'), period or request.args.get('period')]]
        response = jsonify({
            'result': sess.run(result, feed_dict={ x: input }).tolist()[0] })
        response.status_code = 200
        return response
    except Exception as e:
        response = jsonify({
            'message': 'Invalid query parameters. Use: ?home=<team>&away=<team>&score1<int>&score2<int>&period=<1/1.5/2/2.5/3/3.5/4/4.5>',
            'error': str(e),
        })
        response.status_code = 404
        return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
