def compute_acc(results):
    labels = [pair[1] for pair in results]
    predicts = [pair[2] for pair in results]
    assert len(labels) == len(predicts)

    acc = 0
    for label, predict in zip(labels, predicts):
        if label == predict:
            acc += 1

    return acc / len(labels)