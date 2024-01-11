import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Parse command line arguments
parser = argparse.ArgumentParser(description='RIID train datata preprocessing')
parser.add_argument('nrows', help='Number of entries to read', nargs='+')
parser.add_argument('save', help='File to save processed data', nargs='+')

args = parser.parse_args()
nrows = int(args.nrows[0])
save_folder = args.save[0]

train = pd.read_csv('./train.csv', nrows=nrows)
train['prior_question_had_explanation'] = train['prior_question_had_explanation'].astype('bool')
questions = pd.read_csv('./communities.csv')

# remove question 10033 from both dataframes since no tag related to it
train.drop(train.index[train['content_id'] == 10033], inplace = True)
questions.drop(questions.index[questions['question_id'] == 10033], inplace = True)

# remove users with less than 10 interactions
users = train['user_id'].unique()
for idx, u in enumerate(users):
    if train[train['user_id'] == u]['user_id'].count() < 10:
        train.drop(train.index[train['user_id'] == u], inplace=True)
		
# count number of lectures watched
time_frame = 1
users = train['user_id'].unique()
n_lectures_watched = []
for idx, u in enumerate(users):
    user_log = train[train['user_id'] == u]
    num_frame = np.ceil(len(user_log) / time_frame).astype(int)
    for i in range(num_frame):
        if time_frame * (i+1) <= len(user_log):
            lecture_sum = user_log.iloc[0:time_frame * (i+1)]['content_type_id'].sum()
            n_lectures_watched.extend(np.repeat(lecture_sum, time_frame))
        else:
            lecture_sum =  user_log['content_type_id'].sum()
            remainder = np.mod(len(user_log), time_frame)
            n_lectures_watched.extend(np.repeat(lecture_sum, remainder))
train['n_lectures_watched'] = np.array(n_lectures_watched)

# drop lecture entries
train = train[train['content_type_id'] == 0]

# count number of times an explanation is asked
time_frame = 1
users = train['user_id'].unique()
n_question_explained = []
for idx, u in enumerate(users):
    user_log = train[train['user_id'] == u]
    num_frame = np.ceil(len(user_log) / time_frame).astype(int)
    for i in range(num_frame):
        if time_frame * (i+1) <= len(user_log):
            explanation_sum = user_log.iloc[0:time_frame * (i+1)]['prior_question_had_explanation'].sum()
            n_question_explained.extend(np.repeat(explanation_sum, time_frame))
        else:
            explanation_sum =  user_log['prior_question_had_explanation'].sum()
            remainder = np.mod(len(user_log), time_frame)
            n_question_explained.extend(np.repeat(explanation_sum, remainder))
train['n_question_explained'] = np.array(n_question_explained)

# fill nan values with column mean
train['prior_question_elapsed_time'].fillna((train['prior_question_elapsed_time'].mean()), inplace=True)

# merge questions with communities and parts
community_part = questions[['question_id', 'part', 'community']]
train = pd.merge(train[train['content_type_id'] == 0], 
community_part, left_on = 'content_id', right_on = 'question_id', how = 'left') 

# count number of times an explanation is asked
time_frame = 1
users = train['user_id'].unique()
success_per_skill = []
for idx, u in enumerate(users):
    user_log = train[train['user_id'] == u]
    num_frame = np.ceil(len(user_log) / time_frame).astype(int)
    for i in range(num_frame):
        if time_frame * (i+1) <= len(user_log):
            part = user_log.iloc[time_frame*(i)]['part']
            community = user_log.iloc[time_frame*(i)]['community']
            avr = user_log.iloc[0:time_frame * (i+1)].groupby(['part', 'community'])['answered_correctly'].mean().reset_index()
            success = avr[(avr['part'] == part) & (avr['community'] == community)]['answered_correctly'].to_numpy()[0]
            success_per_skill.append(success)
        else:
            part = user_log.iloc[-1]['part']
            community = user_log.iloc[-1]['community']
            avr = user_log.groupby(['part', 'community'])['answered_correctly'].mean().reset_index()
            success = avr[(avr['part'] == part) & (avr['community'] == community)]['answered_correctly'].to_numpy()[0]
            success_per_skill.append(success)
train['success_per_skill'] = np.array(success_per_skill)

# one hot encoding of part and community
train = pd.concat([train, pd.get_dummies(train['part'], prefix='part')], axis=1)
train = pd.concat([train, pd.get_dummies(train['community'], prefix='community')], axis=1)

# normalise time ellapsed
time_elapsed = train['prior_question_elapsed_time'].values
time_scaled = (time_elapsed - np.min(time_elapsed)) / (np.max(time_elapsed) - np.min(time_elapsed))
train['prior_question_elapsed_time'] = time_scaled

# save data frame as csv
train_columns = ['answered_correctly', 'prior_question_elapsed_time', 'prior_question_had_explanation',
                'n_lectures_watched', 'n_question_explained', 'success_per_skill', 'part_1', 'part_2',
                'part_3', 'part_4', 'part_5', 'part_5', 'part_6', 'part_7', 'community_0', 'community_1',
                'community_2', 'community_3']
train_select = train[train_columns]
train_select.to_csv(save_folder, index=False)
