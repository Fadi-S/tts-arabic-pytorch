import text


def preprocess(txt):
    t_phon = text.arabic_to_buckwalter(txt)
    t_phon = text.buckwalter_to_phonemes(t_phon)
    return t_phon


lines = [
    "ازيكم يا جماعه عاملين ايه يا رب تكونوا بخير",
    "منورين يا حبايبي",
    "بخير وصحه وسعاده وهنا وكل حاجه حلوه طبعا",
]

output = []

for i in range(len(lines)):
    phonemes = preprocess(lines[i])
    tokens = text.phonemes_to_tokens(phonemes)
    token_ids = text.tokens_to_ids(tokens)
    output.append(token_ids)

print(output)
