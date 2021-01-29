"""
@Time   :   2021-01-21 12:01:32
@File   :   evaluations.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""


def compute_corrector_prf(results):
    """
    copy from https://github.com/sunnyqiny/Confusionset-guided-Pointer-Networks-for-Chinese-Spelling-Check/blob/master/utils/evaluation_metrics.py
    """
    TP = 0
    FP = 0
    FN = 0
    all_predict_true_index = []
    all_gold_index = []
    for item in results:
        src, tgt, predict = item
        gold_index = []
        each_true_index = []
        for i in range(len(list(src))):
            if src[i] == tgt[i]:
                continue
            else:
                gold_index.append(i)
        all_gold_index.append(gold_index)
        predict_index = []
        for i in range(len(list(src))):
            if src[i] == predict[i]:
                continue
            else:
                predict_index.append(i)

        for i in predict_index:
            if i in gold_index:
                TP += 1
                each_true_index.append(i)
            else:
                FP += 1
        for i in gold_index:
            if i in predict_index:
                continue
            else:
                FN += 1
        all_predict_true_index.append(each_true_index)

    # For the detection Precision, Recall and F1
    detection_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    detection_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    if detection_precision + detection_recall == 0:
        detection_f1 = 0
    else:
        detection_f1 = 2 * (detection_precision * detection_recall) / (detection_precision + detection_recall)
    print("The detection result is precision={}, recall={} and F1={}".format(detection_precision, detection_recall,
                                                                             detection_f1))

    TP = 0
    FP = 0
    FN = 0

    for i in range(len(all_predict_true_index)):
        # we only detect those correctly detected location, which is a different from the common metrics since
        # we wanna to see the precision improve by using the confusionset
        if len(all_predict_true_index[i]) > 0:
            predict_words = []
            for j in all_predict_true_index[i]:
                predict_words.append(results[i][2][j])
                if results[i][1][j] == results[i][2][j]:
                    TP += 1
                else:
                    FP += 1
            for j in all_gold_index[i]:
                if results[i][1][j] in predict_words:
                    continue
                else:
                    FN += 1

    # For the correction Precision, Recall and F1
    correction_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    correction_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    if correction_precision + correction_recall == 0:
        correction_f1 = 0
    else:
        correction_f1 = 2 * (correction_precision * correction_recall) / (correction_precision + correction_recall)
    print("The correction result is precision={}, recall={} and F1={}".format(correction_precision,
                                                                              correction_recall,
                                                                              correction_f1))

    return detection_f1, correction_f1
