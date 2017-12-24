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
        except:
            print('FILE ERROR: ', file)
        
    return (np.stack(train_x), np.stack(train_y))
    

def create_input_from_game(file, team1, team2):
    headers = next(file) # read in the titles for each column

    # using scores for every 6 minutes of a game
    quarter_marker = 1
    score1, score2 = 0, 0
    input = []
    for play in file:
        # players involed in the current play. Used to determine which points belong to which team
        team1_players, team2_players = play[:5], play[5:10]

        # check if points were involved in the play
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
        
        if play[headers.index('period')] != str(quarter_marker):
            quarter_marker = int(play[headers.index('period')])

        # only allow one play per unique quarter marker
        current_time = play[headers.index('time')].split(':')
        minute, second = int(current_time[0]), int(current_time[1])
        # delta of 15 seconds
        if minute == 6 and second > 15 and second <  45 and not any(x[4] == quarter_marker + 0.5 for x in input):
            input.append([hash(team1), hash(team2), score1, score2, quarter_marker + 0.5])
        # delta of 30 seconds for a play to occur within the beginning of a quarter
        #   - larger delta because the start of a quarter generally takes longer for a play to occur
        elif minute == 11 and second > 30 and not any(x[4] == quarter_marker for x in input): 
            input.append([hash(team1), hash(team2), score1, score2, quarter_marker])
    
    # result of the game. Duplicate the same outcome for every play in the input matrix
    result = None
    result = [[1, 0]] if score1 > score2 else [[0, 1]]
    for _ in range(len(input) - 1):
        result.append(result[0][:])
    
    return (input, result)


# debuggin
sample_file = os.listdir(pathname)[0]
sample_teams = sample_file.split('.')[1]
sample_team1, sample_team2 = sample_teams[:3], sample_teams[3:]

sample_reader = csv.reader(open(pathname+'/'+sample_file))
sample_result = create_input_from_game(sample_reader, sample_team1, sample_team2)


# train a NN model based on this data
train_x, train_y = load_season_data(pathname)
split_size = int(train_x.shape[0]*0.7) # 70/30 split between train/test

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train_y[:split_size], train_y[split_size:]

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

    # Save the model
    saver = tf.train.Saver()
    save_path = saver.save(sess, './bin2/model.ckpt')
    print('Model saved in file: %s' % save_path)
