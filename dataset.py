import numpy as np
import csv, os

# keras stuff
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

pathname = './2009-2010.regular_season'

def load_season_data(pathname):
    train_x = []
    train_y = [] # binary 1/0 for win/loss

    filenames = os.listdir(pathname)
    for file in filenames:
        teams = file.split('.')[1]
        reader = csv.reader(open(pathname+'/'+file))
        try:
            x, y = create_input_from_game(reader, teams[:3], teams[3:])
            train_x.extend(x); train_y.extend(y)
        except Exception as e:
            print('FILE ERROR: ', file)
        
    return np.stack(train_x), np.stack(train_y)
    

def create_input_from_game(file, team1, team2):
    """ creates an individual row in the training set.
        data is doubled by swapping team1 and team2.
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
            if minute <= 6: period += 0.5
            row[int(19 + 2*period - 2)] = 1 # encoding of the 8 possible periods

        x.append(row)

    x = np.stack(x)
    y = np.zeros((x.shape[0], 2))
    if score1 > score2:
        y[:,0] = 1
    else:
        y[:,1] = 1

    # 2x the data by swapping team1/team2
    x2, y2 = x.copy(), y.copy()
    for row in x2:
        row0, row1 = row[0], row[1]
        row[0] = row1
        row[1] = row0
    y2[:,0] = 1 if y2[0][0] == 0 else 0
    y2[:,1] = 1 if y2[0][1] == 0 else 0

    return np.vstack((x,x2)), np.vstack((y, y2))


train_x, train_y = load_season_data(pathname)
split_size = int(train_x.shape[0]*0.7) # 70/30 split between train/validation

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train_y[:split_size], train_y[split_size:]

# model training
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


model.fit(train_x, train_y, epochs=30, batch_size=512)

print("")
results = model.predict(val_x[:500])

count = 0
for (actual,guess) in zip(results, val_y[:500]):
    if np.argmax(actual) == np.argmax(guess):
        count += 1

print("{} of {}".format(count, 500))
