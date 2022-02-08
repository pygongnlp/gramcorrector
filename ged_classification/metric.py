
def compute_metrics(predict_labels, labels):
    assert len(predict_labels) == len(labels)
    acc = 0
    for p, t in zip(predict_labels, labels):
        if p == t:
            acc += 1
    return acc / len(predict_labels)