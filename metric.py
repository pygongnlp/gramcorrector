def compute_sent_metrics(results):
    err_sent = [pair[0] for pair in results]
    cor_sent = [pair[1] for pair in results]
    predict = [pair[2] for pair in results]
    assert len(err_sent) == len(cor_sent) == len(predict)

    c_tp, c_fp, c_tn, c_fn = 0, 0, 0, 0
    d_tp, d_fp, d_tn, d_fn = 0, 0, 0, 0
    for es, cs, ps in zip(err_sent, cor_sent, predict):
        cs_d = [e == c for e, c in zip(es, cs)]
        ps_d = [e == p for e, p in zip(es, ps)]
        if es != cs:
            if cs == ps:
                c_tp += 1
            else:
                c_fn += 1
            if cs_d == ps_d:
                d_tp += 1
            else:
                d_fn += 1
        if es == cs:
            if cs == ps:
                c_tn += 1
            else:
                c_fp += 1
            if cs_d == ps_d:
                d_tn += 1
            else:
                d_fp += 1

    d_acc = (d_tp + d_tn) / (d_tp + d_tn + d_fp + d_fn)
    d_pre = d_tp / (d_tp + d_fp)
    d_rec = d_tp / (d_tp + d_fn)
    d_f1 = 2 * d_pre * d_rec / (d_pre + d_rec)

    c_acc = (c_tp + c_tn) / (c_tp + c_tn + c_fp + c_fn)
    c_pre = c_tp / (c_tp + c_fp)
    c_rec = c_tp / (c_tp + c_fn)
    c_f1 = 2 * c_pre * c_rec / (c_pre + c_rec)


    return {
        "sent_d_acc": d_acc,
        "sent_d_pre": d_pre,
        "sent_d_rec": d_rec,
        "sent_d_f1": d_f1,
        "sent_c_acc": c_acc,
        "sent_c_pre": c_pre,
        "sent_c_rec": c_rec,
        "sent_c_f1": c_f1
    }


def compute_char_metrics(results):
    pass