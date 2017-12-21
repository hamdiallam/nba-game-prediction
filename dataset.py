import json
import tensorflow as tf

# loading in training data
sb_nation = json.load(open('./sbnation/train.json', 'r'))
rotowire = json.load(open('./rotowire/train.json', 'r'))

# collecting the training data
y_train = []
x_train = []

for game in sb_nation:
    team1_pts = game['home_line']['TEAM-PTS']
    team2_pts = game['vis_line']['TEAM-PTS']
    if team1_pts > team2_pts:
        result = [1, 0] # home team won
    else:
        result = [0, 1] # visiting team won

    # use every quarter's score with the result of the game
    for i in range(1, 5):
        score1 = game['home_line']['TEAM-PTS_QTR' + str(i)]
        score2 = game['vis_line']['TEAM-PTS_QTR' + str(i)]
        x_train.append([score1, score2])
        y_train.append(result)


# Model based on the MNSIT tensorflow example

x = tf.placeholder(tf.float32, [None, 2])
y_ = tf.placeholder(tf.float32, [None, 2])

W = tf.Variable(tf.zeros([2, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# train the model with x_train and y_train
sess.run(train_step, feed_dict={x: x_train, y_: y_train})


# Evaluating the model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy: ', sess.run(accuracy, feed_dict={x: x_train, y_: y_train}))


# Save the model
saver = tf.train.Saver()
save_path = saver.save(sess, './bin/model.ckpt')
print('Model saved in file: %s' % save_path)
