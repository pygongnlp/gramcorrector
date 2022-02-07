def compute_metrics(results):
    TP, FP, FN = 0, 0, 0
    all_predict_true_index, all_gold_index = [], []
    for (src, tgt, predict) in results:
        assert len(src) == len(tgt) == len(predict), f"{src}   {tgt}   {predict}  {len(src)}/{len(tgt)}/{len(predict)}"
        gold_index, each_true_index = [], []
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

    detection_precision = TP / (TP + FP) if (TP+FP) > 0 else 0
    detection_recall = TP / (TP + FN) if (TP+FN) > 0 else 0
    detection_f1 = 2 * (detection_precision * detection_recall) / (detection_precision + detection_recall) \
        if (detection_precision + detection_recall) > 0 else 0

    TP, FP, FN = 0, 0, 0
    for i in range(len( all_predict_true_index)):
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
                if results[i][1][j]  in predict_words:
                    continue
                else:
                    FN += 1

    # For the correction Precision, Recall and F1
    correction_precision = TP / (TP + FP) if (TP+FP) > 0 else 0
    correction_recall = TP / (TP + FN) if (TP+FN) > 0 else 0
    correction_f1 = 2 * (correction_precision * correction_recall) / (correction_precision + correction_recall) \
        if (correction_precision + correction_recall) > 0 else 0

    return {
        "det_pre": detection_precision,
        "det_rec": detection_recall,
        "det_f1": detection_f1,
        "cor_pre": correction_precision,
        "cor_rec": correction_recall,
        "cor_f1": correction_f1
    }