import json
import pandas as pd
import random
from tqdm import tqdm
from datasets import Dataset, DatasetInfo, DatasetDict, Features, Value
from load import load_cla_data

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def preprocess_data(dataset, dates):
    train_dataset, valid_dataset, test_dataset = load_cla_data(
        f'data/stock_movement/raw/{dataset}/price',
        f'data/stock_movement/raw/{dataset}/tweet',
        dates[0], dates[1], dates[2], seq=10,
    )
    return train_dataset, valid_dataset, test_dataset

def build_instructions(dataset, templates, ds, datum_id, truncate=True, inference=False):
    results = []
    lengths = []
    truncate_num = 0
    options = ["Rise", "Fall"]
    for datum in tqdm(zip(*dataset)):
        tid = "$"+datum[4].lower()
        string_row = ["date,open,high,low,close,adj-close,inc-5,inc-10,inc-15,inc-20,inc-25,inc-30"]
        for t in datum[0]:
            t = [f"{a:.1f}" if index > 0 else a for index, a in enumerate(t)]
            string_row.append(','.join(t))
        string_row = "\n".join(string_row)
        text = datum[3]
        text = [": ".join([val[0], val[1].replace("\n", "").replace("URL", "")[:200]]) for val in zip(text["date"], text["text"])]
        text = "\n".join(text)
        context = f"{string_row}\n\n{text}"
        ans = "Rise" if datum[2] > 0 else "Fall"

        template = random.choice(templates).format(tid=tid, point=datum[-1])
        prompt = f"{template}\nContext: {context}\nAnswer:"
        temp = {"id": f"{ds[:-2]}sm{datum_id}",
                "query": prompt, "answer": ans, "text": context,
                "choices": options, "gold": options.index(ans),}
        datum_id += 1
        # tokens = tokenizer.encode(temp["conversations"][0]["value"])
        # if len(tokens) > 2000 and truncate:
        #     truncate_num += 1
        #     continue
        results.append(temp)
    results = pd.DataFrame(results)
    return results, datum_id

def dump_json(dataset, name, ds):
    results = []
    for datum in dataset:
        results.append(json.dumps(datum))
    with open(f"data/stock_movement/{ds}/{name}.json", "w") as f:
        f.writelines("\n".join(results))

def dump_evaluate_data(data, dataset_name, dataset):
    for datum in data:
        datum["text"] = datum["conversations"][0]["value"].split("\n")[1]
        datum["label"] = datum["conversations"][1]["value"]
    d = [json.dumps(val) for val in data]
    with open(f"data/stock_movement/{dataset}/{dataset_name}.jsonl", "w") as f:
        f.writelines("\n".join(d))


instructions = [
    "Analyze the information and social media posts to determine if the closing price of {tid} will ascend or descend at {point}. Please respond with either Rise or Fall.",
    "Reflect on the provided data and tweets to anticipate if the closing price of {tid} is going to increase or decrease at {point}. Kindly respond with either Rise or Fall.",
    "With the help of the data and tweets given, can you forecast whether the closing price of {tid} will climb or drop at {point}? Please state either Rise or Fall.",
    "By reviewing the data and tweets, can we predict if the closing price of {tid} will go upwards or downwards at {point}? Please indicate either Rise or Fall.",
    "Utilize the data and tweets at hand to foresee if the closing price of {tid} will elevate or diminish at {point}. Only answer with Rise or Fall.",
    "Given the data and tweets, could you project whether the closing price of {tid} will grow or shrink at {point}? Please specify either Rise or Fall.",
    "Contemplate the data and tweets to guess whether the closing price of {tid} will surge or decline at {point}. Please declare either Rise or Fall.",
    "Assess the data and tweets to estimate whether the closing price of {tid} will escalate or deflate at {point}. Respond with either Rise or Fall.",
    "Scrutinize the data and tweets to envisage if the closing price of {tid} will swell or contract at {point}. Please affirm either Rise or Fall.",
    "Examine the data and tweets to deduce if the closing price of {tid} will boost or lower at {point}. Kindly confirm either Rise or Fall."
]

dataset_configs = {
    "bigdata22": ['2020-04-01', '2020-10-01', '2020-11-02'],
    "acl18": ['2014-01-02', '2015-08-03', '2015-10-01'],
    "cikm18": ['2017-01-03', '2017-10-02', '2017-11-02'],
}
for dataset, dates in dataset_configs.items():
    train_dataset, valid_dataset, test_dataset = preprocess_data(dataset, dates)
    id_ = 0
    train_dataset, id_ = build_instructions(train_dataset, instructions, dataset, id_)
    valid_dataset, id_ = build_instructions(valid_dataset, instructions, dataset, id_)
    test_dataset, id_ = build_instructions(test_dataset, instructions, dataset, id_, False, True)


    dataset_dict = {}
    all_data = [train_dataset, valid_dataset, test_dataset]
    for data, name in zip(all_data, ["train", "valid", "test"]):
        dataset_dict[name] = Dataset.from_pandas(data)

    my_dataset = DatasetDict(dataset_dict)

    # Push to Hugging Face
    my_dataset.push_to_hub(f"ChanceFocus/flare-sm-{dataset[:-2]}")
