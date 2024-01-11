import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class dataLoader(Dataset):
    def __init__(self, df): 
        feature_columns = ['prior_question_elapsed_time', 'prior_question_had_explanation', 'n_lectures_watched',
                          'success_per_skill']
        self.features = df.loc[:, feature_columns].to_numpy(dtype=np.float32)
        self.targets = df['answered_correctly'].to_numpy(dtype=np.float32)
        self.difficulty = df['difficulty'].to_numpy(dtype=np.float32)

        self.data_length = len(self.targets)
        
    def __getitem__(self, index):
        features = self.features[index]
        targets = self.targets[index]
        difficulty = self.difficulty[index]

        return features, targets, difficulty
    
    def __len__(self):
        return self.data_length
