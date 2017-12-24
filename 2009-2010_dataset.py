import numpy as np
import tensorflow as tf
import csv
import os

pathname = './2009-2010.regular_season'

def load_season_data(pathname):
    train_x = []
    train_y = [] # binary 1/0 for win/loss

    filenames = os.listdir(pathname)
    for file in filenames:
        teams = file.split('.')[1]
        reader = csv.reader(open(file))
        # TODO: double check the append vs extend
        train_x.extend(create_input_from_game(reader, teams[:3], teams[3:]))
    return np.stack(train_x)
    

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
            input.append([team1, team2, score1, score2, quarter_marker + 0.5])
        # delta of 30 seconds for a play to occur within the beginning of a quarter
        #   - larger delta because the start of a quarter generally takes longer for a play to occur
        elif minute == 11 and second > 30 and not any(x[4] == quarter_marker for x in input): 
            input.append([team1, team2, score1, score2, quarter_marker])
    
    # result of the game. Duplicate the same outcome for every play in the input matrix
    result = None
    result = [[1, 0]] if score1 > score2 else [[0, 1]]
    for _ in range(len(input) - 1):
        result.append(result[0][:])
    
    return (np.stack(input), np.stack(result))


# debuggin
sample_file = os.listdir(pathname)[0]
teams = sample_file.split('.')[1]
team1, team2 = teams[:3], teams[3:]

reader = csv.reader(open(pathname+'/'+sample_file))
result = create_input_from_game(reader, team1, team2)

