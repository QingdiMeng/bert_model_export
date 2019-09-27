import gluonnlp as nlp;
import mxnet as mx;
import time

model, vocab = nlp.model.get_model('bert_12_768_12', dataset_name='wiki_cn_cased')
tokenizer = nlp.data.BERTTokenizer(vocab, lower=True)
transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=25, pair=False, pad=True)
sample = transform(['我们的世界是什么'])
words, valid_len, segments = mx.nd.array([sample[0]]), mx.nd.array([sample[1]]), mx.nd.array([sample[2]])

masked_positions = mx.nd.array([[
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0
]], dtype="int32")

print(words)
print(valid_len)
print(segments)

for i in range(1):
    start = time.time()
    seq_encoding = model(words, segments, valid_len, masked_positions)

    print(seq_encoding[1])
    print(time.time() - start)
