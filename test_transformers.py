import torch
from transformers import BertTokenizer, BertModel
import time

import logging
logging.basicConfig(level=logging.INFO)

tokenizer = BertTokenizer.from_pretrained("./transformers_pre_trained_bert_base_chinese")
model = BertModel.from_pretrained("./transformers_pre_trained_bert_base_chinese")
model.eval()

# print(tokenizer.encode("我们的世界是什么", add_special_tokens=True))
encode_plus = tokenizer.encode_plus("我们的世界是什么", add_special_tokens=True, max_length=25)


print(encode_plus)

input_ids = encode_plus["input_ids"]
token_type_ids = encode_plus["token_type_ids"]
input_mask_ids = [1 for x in input_ids]


if len(input_ids) < 25:
    input_ids = input_ids + [0 for i in range(25 - len(input_ids))]
    token_type_ids = token_type_ids + [0 for i in range(25 - len(token_type_ids))]
    input_mask_ids = input_mask_ids + [0 for i in range(25 - len(input_mask_ids))]

print(input_ids)
print(token_type_ids)
print(input_mask_ids)

input_ids_tensor = torch.tensor([input_ids])
token_type_ids_tensor = torch.tensor([token_type_ids])
input_mask_ids_tensor = torch.tensor([input_mask_ids])

with torch.no_grad():
    start = time.time()
    print(model(input_ids=input_ids_tensor, attention_mask=input_mask_ids_tensor, token_type_ids=token_type_ids_tensor))
    print(time.time() - start)


# tokenizer.save_pretrained("./transformers_pre_trained_bert_base_chinese")
# model.save_pretrained("./transformers_pre_trained_bert_base_chinese")