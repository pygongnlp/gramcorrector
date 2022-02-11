import numpy as np
import operator


def compute_metrics(results):
    src = [pair[0] for pair in results]
    trg = [pair[1] for pair in results]
    predict = [pair[2] for pair in results]

    mod_sen = [src[i] != predict[i] for i in range(len(predict))]
    acc_sen = [trg[i] == predict[i] for i in range(len(predict))]
    tar_sen = [trg[i] != src[i] for i in range(len(src))]
    sen_mod = sum(mod_sen)
    sen_mod_acc = sum(np.multiply(np.array(mod_sen), np.array(acc_sen)))
    sen_tar_mod = sum(tar_sen)
    sen_acc = sum(acc_sen)
    setsum = len(src)

    prob_ = [[0 if predict[i][j] == src[i][j] else 1 for j in
              range(len(predict[i]))] for i in range(len(predict))]
    label = [[0 if src[i][j] == trg[i][j] else 1 for j in
              range(len(src[i]))] for i in range(len(src))]
    d_acc_sen = [operator.eq(prob_[i], label[i]) for i in range(len(prob_))]
    d_mod_sen = [0 if sum(prob_[i]) == 0 else 1 for i in range(len(prob_))]
    d_tar_sen = [0 if sum(label[i]) == 0 else 1 for i in range(len(label))]
    d_sen_mod = sum(d_mod_sen)
    d_sen_mod_acc = sum(np.multiply(np.array(d_mod_sen), np.array(d_acc_sen)))
    d_sen_tar_mod = sum(d_tar_sen)
    d_sen_acc = sum(d_acc_sen)

    d_precision = d_sen_mod_acc / d_sen_mod
    d_recall = d_sen_mod_acc / d_sen_tar_mod
    d_f1 = 2 * d_precision * d_recall / (d_precision + d_recall)
    c_precision = sen_mod_acc / sen_mod
    c_recall = sen_mod_acc / sen_tar_mod
    c_f1 = 2 * c_precision * c_recall / (c_precision + c_recall)

    return {
        "det_acc": d_sen_acc / setsum,
        "det_pre": d_precision,
        "det_rec": d_recall,
        "det_f1": d_f1,
        "cor_acc": sen_acc / setsum,
        "cor_pre": c_precision,
        "cor_rec": c_recall,
        "cor_f1": c_f1
    }