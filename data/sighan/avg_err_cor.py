import json
import random

ratio = 0.5
pairs = json.load(open("data/sighan/dev.json", "r", encoding="utf8"))

avg_pairs = []
same_oral_cnt, same_avg_cnt = 0, 0
for pair in pairs:
    err_sent = pair["original_text"]
    cor_sent = pair["correct_text"]

    if err_sent == cor_sent:
        avg_pairs.append(pair)
        same_oral_cnt += 1
        same_avg_cnt += 1
    elif random.random() > ratio:
        pair["original_text"] = cor_sent
        avg_pairs.append(pair)
        same_avg_cnt += 1
    else:
        avg_pairs.append(pair)

assert len(pairs) == len(avg_pairs)
print(f"{len(pairs)} pairs, {same_oral_cnt} are same, after avg, {same_avg_cnt} are same")

json.dump(avg_pairs, open("data/sighan/train_avg.json", "w", encoding="utf8"), indent=2, ensure_ascii=False)

