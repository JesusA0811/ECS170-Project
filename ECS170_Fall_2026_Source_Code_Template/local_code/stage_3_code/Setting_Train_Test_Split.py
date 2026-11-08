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

        self.dataset.dataset_source_file_name = 'test.csv'
        test_data = self.dataset.load()

        # Pass both datasets to the MLP
        self.method.data = {
            'train': train_data,
            'test': test_data
        }

        # Train and test the model
        learned_result = self.method.run()

        # Save results
        self.result.data = learned_result
        self.result.fold_count = 0
        self.result.save()

        # Evaluate results
        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None