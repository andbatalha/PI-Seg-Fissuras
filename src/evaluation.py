import numpy as np

def calculate_evaluation_counts(label, predicted_mask):
    results = {}

    for classe in range(6):
        real = label == classe
        predicted = predicted_mask == classe

        true_positive = int(np.sum(real & predicted))
        false_positive = int(np.sum(~real & predicted))
        false_negative = int(np.sum(real & ~predicted))

        precision = ( true_positive / (true_positive + false_positive)
                if (true_positive + false_positive) > 0 else 0.0
        )
        
        recall = ( true_positive / (true_positive + false_negative)
                if (true_positive + false_negative) > 0 else 0.0
        )


        IoU = ( true_positive / (true_positive + false_negative + false_positive)
                if (true_positive + false_negative + false_positive) > 0 else 0.0
        )

        f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0 else 0.0
        )

        results[classe] = {
            "TP": true_positive,
            "FP": false_positive,
            "FN": false_negative,
            "Precision": precision,
            "Recall": recall,
            "IoU": IoU,
            "F1": f1_score
        }

    return results

