
from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- object initialization section -------------------
    data_obj = Dataset_Loader('stage 2 training', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.dataset_source_file_name = 'train.csv'

    method_obj = Method_MLP('multi-layer perceptron', '')

    # Initial experiment: 10 epochs
    method_obj.max_epoch = 10

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/'
    result_obj.result_destination_file_name = 'prediction_result'

    # Create result directory if it doesn't exist
    os.makedirs(
        result_obj.result_destination_folder_path,
        exist_ok=True
    )

    setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')

    setting_obj.prepare(
        data_obj,
        method_obj,
        result_obj,
        evaluate_obj
    )

    setting_obj.print_setup_summary()

    # Train, test, save predictions, and evaluate
    mean_score, std_score = setting_obj.load_run_save_evaluate()

    print('************ Overall Performance ************')
    print('MLP Test Accuracy:', mean_score)

    print('************ Finish ************')
    # ------------------------------------------------------
# ---- Training Convergence Plot ----
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(method_obj.loss_history) + 1),
    method_obj.loss_history,
    marker='o'
)

plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('MLP Training Convergence')
plt.grid(True)

plt.savefig(
    '../../result/stage_2_result/training_loss1.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()