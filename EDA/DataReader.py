from os import path
import pandas as pd
from sklearn.model_selection import train_test_split

from mint import get_data_path
from string import punctuation
from underthesea import word_tokenize


def clean(text: str) -> str:
    text = text.replace("\u200b", "")
    text = text.replace("\n", "")
    text = text.translate(str.maketrans("", "", punctuation))
    text = text.lower()
    return word_tokenize(text, format("text"))


def clean_corpus(corpus: list) -> list:
    cleaned_corpus = [clean(cor) for cor in corpus]
    return cleaned_corpus


class DataReader:
    def __init__(self, p: str):
        self.data_path_root = get_data_path(p)
        if p != "":
            self.df_train = self.read_file("train")
            self.df_test = self.read_file("test")
            self.df_dev = self.read_file("dev")
            self.df_total = pd.concat([self.df_train, self.df_dev, self.df_test], axis=0)
        else:
            self.df_train = None
            self.df_test = None
            self.df_dev = None

    def read_file(self, t: str) -> pd.DataFrame:
        data_path = path.join(self.data_path_root, t)
        sents = pd.read_csv(path.join(data_path, "sents.txt"), delimiter='\r', header=None, )
        sentiments = pd.read_csv(path.join(data_path, "sentiments.txt"), delimiter='\r', header=None)
        topics = pd.read_csv(path.join(data_path, "topics.txt"), delimiter='\r', header=None)
        df = pd.DataFrame()
        df["corpus"] = clean_corpus(sents[0].tolist())
        df["label"] = sentiments[0].tolist()
        df["topics"] = topics[0].tolist()
        df["length"] = [len(corpus) for corpus in df["corpus"]]
        return df

    def read_csv_file(self, t: str):
        data_path = path.join(self.data_path_root, t)
        self.df_total = pd.read_csv(data_path, delimiter=',')
        self.df_total["corpus"] = clean_corpus(self.df_total["text"])
        self.split_train_test()

    def split_train_test(self):
        if self.df_total is None or self.df_total.empty:
            raise ValueError("DataFrame self.df is None or empty!")

        train_corpus, test_corpus, train_label, test_label = train_test_split(
            self.df_total["corpus"], self.df_total["label"], test_size=0.2, random_state=42
        )

        # Tạo DataFrame nếu chưa có
        self.df_train = pd.DataFrame({"corpus": train_corpus, "label": train_label})
        self.df_test = pd.DataFrame({"corpus": test_corpus, "label": test_label})
