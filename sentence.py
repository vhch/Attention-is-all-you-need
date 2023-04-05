import pandas as pd
import sentencepiece as spm
import csv

en = pd.read_fwf('baseline-1M_train.en', header = None, keep_default_na = False)
en = en.loc[:,0]

fr = pd.read_fwf('baseline-1M_train.fr', header = None, keep_default_na = False)
fr = fr.loc[:,0]
print(fr)



with open('translate_english.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(en))

spm.SentencePieceTrainer.Train(
input='translate_english.txt', model_prefix='translate_english', vocab_size=30000, model_type='bpe', pad_id=0, unk_id=1, bos_id=2, eos_id=3)

with open('translate_french.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(fr))

spm.SentencePieceTrainer.Train(
input='translate_french.txt', model_prefix='translate_french', vocab_size=30000, model_type='bpe', pad_id=0, unk_id=1, bos_id=2, eos_id=3)

sp_e = spm.SentencePieceProcessor()
vocab_file = "translate_english.model"
sp_e.load(vocab_file)

sp_f = spm.SentencePieceProcessor()
vocab_file = "translate_french.model"
sp_f.load(vocab_file)

# print(sp.IdToPiece(397))
# print(sp.IdToPiece(31))
# print(sp.IdToPiece(223))
# print(sp.IdToPiece(121))
# print(sp.IdToPiece(5))
# print(sp.IdToPiece(4574))



# sequence = "씨티은행에서 일하세요?"

sp_e.SetEncodeExtraOptions('bos')
sequence2 = "Do you work at a City bank?"
print(sp_e.encode(sequence2, out_type=int))
# print(sp.DecodeIds([1079, 29046]))


sp_e.SetEncodeExtraOptions('eos')
sequence2 = "Do you work at a City bank?"
print(sp_e.encode('sequence2', out_type=int))
