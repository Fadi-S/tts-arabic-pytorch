import text
import pandas as pd
from tqdm import tqdm


def preprocess(txt):
    txt = txt.replace(".", "")
    txt = txt.replace("!", "")
    txt = txt.replace(",", "")
    t_phon = text.arabic_to_buckwalter(txt)
    t_phon = text.buckwalter_to_phonemes(t_phon)
    return t_phon


lines = pd.read_csv("index.csv")["text"]

output = []

for i in tqdm(range(len(lines))):
    phonemes = preprocess(lines[i])
    tokens = text.phonemes_to_tokens(phonemes)
    token_ids = text.tokens_to_ids(tokens)
    # output.append(token_ids)

# print(output)
