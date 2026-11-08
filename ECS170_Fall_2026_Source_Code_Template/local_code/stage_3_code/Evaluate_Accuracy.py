'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class Evaluate_Accuracy(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')

        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        accuracy = accuracy_score(true_y, pred_y)
        precision = precision_score(
            true_y, pred_y, average='weighted',
            zero_division=0
        )
        recall = recall_score(
            true_y, pred_y, average='weighted',
            zero_division=0
        )
        f1 = f1_score(
            true_y, pred_y, average='weighted',
            zero_division=0
        )

        print('Accuracy:', accuracy)
        print('Weighted Precision:', precision)
        print('Weighted Recall:', recall)
        print('Weighted F1:', f1)

        return accuracy

        