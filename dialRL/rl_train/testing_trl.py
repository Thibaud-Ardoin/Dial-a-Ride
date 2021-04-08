import torch
from transformers import GPT2Tokenizer
from dialRL.rl_train.trl.GPT2_trl import GPT2HeadWithValueModel, respond_to_batch
from dialRL.rl_train.trl.ppo_trl import PPOTrainer

# get models
gpt2_model = GPT2HeadWithValueModel.from_pretrained('gpt2')
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# initialize trainer
ppo_config = {'batch_size': 1, 'forward_batch_size': 1}
ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, **ppo_config)

# encode a query
query_txt = "This morning I went to the "
query_tensor = gpt2_tokenizer.encode(query_txt, return_tensors="pt")

# get model response
response_tensor  = respond_to_batch(gpt2_model, query_tensor, txt_len=1)
response_txt = gpt2_tokenizer.decode(response_tensor[0,:])

# define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = torch.tensor([-1.0])

# train model with ppo
train_stats = ppo_trainer.step(query_tensor, response_tensor, reward)
print(train_stats)
