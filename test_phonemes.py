import text
import pandas as pd
from tqdm import tqdm
#
#
def preprocess(txt):
    txt = txt.replace(".", "")
    txt = txt.replace("!", "")
    txt = txt.replace(",", "")
    t_phon = text.arabic_to_buckwalter(txt)
    t_phon = text.buckwalter_to_phonemes(t_phon)
    return t_phon

#
# lines = pd.read_csv("index.csv")["text"]
#
# output = []
#
# for i in tqdm(range(len(lines))):
#     phonemes = preprocess(lines[i])
#     tokens = text.phonemes_to_tokens(phonemes)
#     token_ids = text.tokens_to_ids(tokens)
#     # output.append(token_ids)
#
# # print(output)


ids = [40, 14, 37, 30,  4, 16, 41, 14, 30,  4, 28, 30, 29, 41, 30,  4, 28, 22,
        25, 12, 30,  4, 40, 28,  8, 41, 14, 30,  4, 27, 27,  9, 12, 27,  4,  6,
        16,  4,  1]

tokens = text.ids_to_tokens(ids)

print(" ".join(tokens))
