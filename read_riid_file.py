import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class read_riid_file():
    def __init__(self, file, nrows=100000, lectures=False):
        self.data = pd.read_csv(file, nrows=nrows)
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.lectures = not(lectures)
        self.CATEGORAL_COLUMNS = ['content_id', 'task_container_id']
        self.CONTINUOUS_COLUMNS = ['prior_question_had_explanation', 'prior_question_elapsed_time']
        self.dtype = {
            'content_id': 'int16',
            'task_container_id': 'int16',
            'answered_correctly': 'int8',
            'prior_question_had_explanation': 'int8',
            'prior_question_elapsed_time': 'float32'
        }

    def return_features(self, features):
        self.data = self.data[features]
        self.data = self.data[self.dtype.keys()]

    def remove_lectures(self):
        self.data = self.data[self.data['content_type_id'] == 0]
        self.data = self.data[self.data['answered_correctly'] != -1]

    def return_content_mean(self):
        content_mean_final = self.data[['content_id', 'answered_correctly']].groupby(['content_id']).agg(['mean'])
        content_mean_final.columns = ["answered_correctly_content_mean"]
        return content_mean_final

    def return_user_mean(self):
        user_mean_final = self.data[['user_id', 'answered_correctly']].groupby(['user_id']).agg(
            ['mean', 'sum', 'count'])
        user_mean_final.columns = ["answered_correctly_user_mean", 'sum_correct', 'count']
        return user_mean_final

    def sort_time(self, data):
        return data.sort_values(['timestamp'], ascending=True).reset_index(drop=True)

    def scale(self, column):
        col = self.scaler.fit_transform(self.data[column].values.reshape(-1,1))
        self.data[column] = col

    def encode(self, column):
        self.data[column] = self.encoder.fit_transform(self.data[column].values)

    def split_data(self, val_split=0.2):
        split_train, split_validation = train_test_split(self.data, test_size=val_split, shuffle=False)
        split_train = self.sort_time(split_train)
        split_validation = self.sort_time(split_validation)
        return split_train, split_validation

    def fill_na(self):
        # take the column mean
        mean = np.floor(self.data['prior_question_elapsed_time'].mean())
        self.data['prior_question_elapsed_time'].fillna(mean, inplace=True)
        self.data['prior_question_had_explanation'].fillna(0, inplace=True)

    def change_column_type_to_categorical(self):
        self.data[self.CATEGORAL_COLUMNS] = self.data[self.CATEGORAL_COLUMNS].astype('category')

    def prepare_data(self, features, scale_col):
        if self.lectures:
            self.remove_lectures()
        self.return_features(features)
        self.fill_na()
        self.scale(scale_col)
        self.change_column_type_to_categorical()

    def get_embeddings(self):
        #embedded_cols = {n: len(col.cat.categories) for n, col in self.data[self.CATEGORAL_COLUMNS].items()}
        embedded_cols = {'content_id': 32737, 'task_container_id': 10000}
        embedding_sizes = [(n_categories, min(50, (n_categories + 1) // 2)) for _, n_categories in
                           embedded_cols.items()]
        return embedding_sizes