import numpy as np
from torch.utils.data import Dataset

# global variables
CATEGORAL_COLUMNS = ['content_id', 'task_container_id']
CONTINUOUS_COLUMNS = ['prior_question_had_explanation','prior_question_elapsed_time']
TIME_STEP_SIZE = 100

# define data loader class for RIID files
class Riid_RNN_Dataset(Dataset):
    def __init__(self, data):
        self.categorical_data = data.loc[:, CATEGORAL_COLUMNS].to_numpy(dtype=np.int64)
        self.continuous_data = data.loc[:, CONTINUOUS_COLUMNS].to_numpy(dtype=np.float32)
        self.targets = data['answered_correctly'].to_numpy(dtype=np.float32)

        self.data_length = len(self.targets) - TIME_STEP_SIZE

    def __getitem__(self, index):
        X1 = self.categorical_data[index: index + TIME_STEP_SIZE]
        X2 = self.continuous_data[index: index + TIME_STEP_SIZE]
        y = self.targets[index + TIME_STEP_SIZE, np.newaxis]

        return (X1, X2), y

    def __len__(self):
        return self.data_length



