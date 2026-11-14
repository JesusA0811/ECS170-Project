'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np

class Setting_Train_Test_Split(setting):
    fold = 3
    
    def load_run_save_evaluate(self):
        
        # load dataset
        self.dataset.dataset_source_file_name = 'train.csv'
        train_data = self.dataset.load()

        # Use the separate test file to evaluate on examples not used for training.
        self.dataset.dataset_source_file_name = 'test.csv'
        test_data = self.dataset.load()

        # run MethodModule
        # use the provided training and testing datasets
        self.method.data = {
            'train': train_data,
            'test': test_data
        }

        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.fold_count = 0
        self.result.save()

        # Evaluate results
        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None
