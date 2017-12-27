import numpy as np
import tensorflow as tf
import csv, os

pathname = './2009-2010.regular_season'

def load_season_data(pathname):
    train_x = []
    train_y = [] # binary 1/0 for win/loss

    filenames = os.listdir(pathname)
    for file in filenames:
        teams = file.split('.')[1]
        reader = csv.reader(open(pathname+'/'+file))
        # some csv are corrupted
        try:
            x, y = create_input_from_game(reader, teams[:3], teams[3:])
            train_x.extend(x); train_y.extend(y)
        except Exception as e:
            print('FILE ERROR: ', file)
        
    return np.stack(train_x), np.stack(train_y)
    

def create_input_from_game(file, team1, team2):
    """ creates an individual row in the training set
    Entries in a row:
        1: team1 winning
        2: team2 winning
        3-18: encoding of the score difference. max 15 point differential
        19-27: encoding of the period. Every 6 minutes. 8 period
        28: overtime
    """
    headers = next(file) # read in the titles for each column

    score1, score2 = 0, 0
    x = []
    for play in file:
        row = np.zeros(28)
        # players involed in the current play. Used to determine which points belong to which team
        team1_players, team2_players = play[:5], play[5:10]

        # Calculation of the points
        if play[headers.index('points')]:
            points = int(play[headers.index('points')])
            if play[headers.index('player')] in team1_players:
                score1 = score1 + points
            else:
                score2 = score2 + points
        # free throws considered seperately from made shots
        elif play[headers.index('etype')] == 'free throw' and play[headers.index('result')] == 'made':
            if play[headers.index('player')] in team1_players:
                score1 = score1 + 1
            else:
                score2 = score2 + 1
        else:
            continue # no points were scored in this play

        if score1 > score2:
            row[0] = 1
        elif score2 > score1:
            row[1] = 1

        diff = min(15, abs(score2 - score1))
        if diff != 0: # make sure that a difference exists
            row[2 + diff - 1] = 1 # encoding of the score differential

        period, minute = int(play[headers.index('period')]), int(play[headers.index('time')].split(':')[0])
        if period > 4:
            row[27] = 1
        else:
            period *= 2
            if minute >= 6: period += 1
            row[19 + period - 2] = 1 # encoding of the 8 possible periods
        x.append(row)

    x = np.stack(x)
    # result of the game. Duplicate the same outcome for every play in the input matrix
    y = np.zeros((x.shape[0], 2))
    if score1 > score2:
        y[:,0] = 1
    else:
        y[:,1] = 1
    
    return x, y



# train a NN model based on this data
train_x, train_y = load_season_data(pathname)
split_size = int(train_x.shape[0]*0.7) # 70/30 split between train/test

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train_y[:split_size], train_y[split_size:]

for i, x in enumerate(train_x[:5]):
    print(x.tolist())
    print(train_y[i].tolist())
assert False

# 1: true if team 1 winning
# 2: true if team 2 winning
# 3-18: encoding of the score difference
# 19-27: encoding of the period

# result: is 1,0 if team 1, 0,1 if team 2

# -> flip all the teams and data -> 2x more data
# -> +-1 to the score, -> 2x

# Model based on the MNSIT tensorflow example

# Period 1, difference 2, team 1 is winning
# 0.52, 0.48 -> argmax -> 1,0

# Period 4, difference 5, team 1 is winning
# 0.65, 0.35 -> argmax -> 1,0

def rotate_x(prev_x):
    new_x = []

    for x in prev_x:
        row = np.zeros(27)

        if x[2] > x[3]:
            row[0] = 1
        elif x[3] > x[2]:
            row[1] = 1

        score_diff = int(min(15, abs(x[2] - x[3])))
        row[2 + score_diff - 1] = 1

        period = int(19 + (x[4] * 2) - 2)
        row[period] = 1

        new_x.append(row)

    return np.array(new_x)    

train_x = rotate_x(train_x)
val_x = rotate_x(val_x)

for x in train_x[:20]:
    print(x.tolist())
print(train_y[:20])

print(len(train_x))
assert False 

import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
model.add(Dense(64, input_dim=len(train_x[0])))
model.add(Dense(
    64, 
    activation='relu', 
    kernel_initializer='random_uniform', 
    bias_initializer='random_uniform'
))
model.add(Dense(
    64, 
    activation='relu', 
    kernel_initializer='random_uniform', 
    bias_initializer='random_uniform'
))
model.add(Dense(
    64, 
    activation='relu', 
    kernel_initializer='random_uniform', 
    bias_initializer='random_uniform'
))
model.add(Dense(2, activation='softmax'))

optimizer = Adam(lr=0.01)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_x, train_y, epochs=1000, batch_size=512)

print("")
results = model.predict(val_x[:500])

count = 0
for (actual,guess) in zip(results, val_y[:500]):
    if np.argmax(actual) == np.argmax(guess):
        count += 1

print("{} of {}".format(count, 500))

assert False

# helper function
def batch_creator(train_x, train_y, batch_size):
    for i in range(0, train_x.shape[0], batch_size):
        yield (train_x[i: i + batch_size], train_y[i: i + batch_size])


# layers
input_num_units = 5
hidden_num_units = 3
output_num_units = 2

x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

epochs = 6
batch_size = 100
learning_rate = 0.01

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

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(epochs):
        avg_cost = 0
        for batch_x, batch_y in batch_creator(train_x, train_y, batch_size):
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / batch_size
        print("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))

    # find predictions on val set
    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, tf.float32))
    print("Validation Accuracy:", sess.run(accuracy, feed_dict={x: val_x, y: val_y}))

    pred_temp2 = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy2 = tf.reduce_mean(tf.cast(pred_temp, tf.float32))
    print("Training Accuracy:", sess.run(accuracy, feed_dict={x: train_x, y: train_y}))

    # Save the model
    saver = tf.train.Saver()
    save_path = saver.save(sess, './bin2/model.ckpt')
    print('Model saved in file: %s' % save_path)
