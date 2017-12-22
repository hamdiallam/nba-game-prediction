from flask import Flask, request, jsonify
import tensorflow as tf

# variables used by the model
W = tf.Variable(tf.zeros([2, 2]))
b = tf.Variable(tf.zeros([1, 2]))
x = tf.Variable(tf.zeros([1, 2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# load in the graph
sess = tf.Session()
saver = tf.train.Saver({ 'W': W, 'b': b })
saver.restore(sess, './bin/model.ckpt')

app = Flask(__name__)

@app.route('/api', methods=['GET'])
def data():
    try:
        input = [[request.args.get('team1'), request.args.get('team2')]]
        response = jsonify(sess.run(y, { x: input }).tolist()[0])
        response.status_code = 200
        return response
    except Exception as e:
        response = jsonify({
            'Message': 'Invalid query parameters. Use: ?team1=<score>&team2=<score>',
        })
        response.status_code = 404
        return response


if __name__ == '__main__':
    app.run(debug=True)
