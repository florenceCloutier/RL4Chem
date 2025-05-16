import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import pandas as pd
import json

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TransPolicy(nn.Module):
    def __init__(self, vocab, max_len, n_heads, n_embed, n_layers, dropout):
        super(TransPolicy, self).__init__()

        self.vocab = vocab
        self.max_len = max_len        
        self.n_heads = n_heads
        self.n_embed = n_embed
        self._dropout = dropout
        self.n_layers = n_layers

        print('TRANS: arguments are stored')

        self.embedding = nn.Embedding(len(vocab), n_embed, padding_idx=vocab.pad, dtype=torch.float32)
        self.position_embedding = nn.Embedding(max_len, n_embed, dtype=torch.float32)
        
        print('TRANS: embeddings are initialised')

        encoder_layer = nn.TransformerEncoderLayer(n_embed, n_heads, 4 * n_embed, dropout=dropout, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers, nn.LayerNorm(n_embed))

        print('TRANS: transformer encoder layers are initialised')

        self.linear = nn.Linear(n_embed, len(vocab)) 
        ### Change action type
        # self.op_head = nn.Linear(n_embed, 3)  # for op_type
        # self.token_head = nn.Linear(n_embed, len(vocab))  # for token
        # self.pos_head = nn.Linear(n_embed, max_len)  # for position
        ###
        
        print('TRANS: linear layer is initialised')

        self.register_buffer('triu', torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1))

        print('TRANS: buffer is registered')
        print('TRANS: done')

    def forward(self, x):
        L, B = x.shape
        x_tok = self.embedding(x)  #L, B
        x_pos = self.position_embedding(torch.arange(L, device=x.device)).view(L, 1, -1).expand(-1, B, -1)
        x = x_tok + x_pos
        x = self.encoder(x, self.triu[:L, :L])
        logits = self.linear(x)
        
        ### Change action type
        # x_pooled = x.mean(dim=0)
        
        # pos_logits = self.pos_head(x_pooled)
        # start_idx = 17
        # mask = torch.arange(self.max_len, device=x.device) < start_idx
        # pos_logits[:, mask] = float('-inf')
        
        # op_logits = self.op_head(x_pooled)
        # token_logits = self.token_head(x_pooled)
        
        # return {
        #     "op_dist": td.Categorical(logits=op_logits),
        #     "token_dist": td.Categorical(logits=token_logits),
        #     "pos_dist": td.Categorical(logits=pos_logits),
        # }
        ###
        
        return td.Categorical(logits=logits)

    def autoregress(self, x):
        L, B = x.shape
        x_tok = self.embedding(x)  #L, B
        x_pos = self.position_embedding(torch.arange(L, device=x.device)).view(L, 1, -1).expand(-1, B, -1)
        x = x_tok + x_pos
        x = self.encoder(x, self.triu[:L, :L])
        
        ### Change action type
        # return self.forward(x)
        # x_pooled = x.mean(dim=0)  # or x[-1] for the last token

        # op_logits = self.op_head(x_pooled)
        # token_logits = self.token_head(x_pooled)
        # pos_logits = self.pos_head(x_pooled)

        # return {
        #     "op_dist": td.Categorical(logits=op_logits),
        #     "token_dist": td.Categorical(logits=token_logits),
        #     "pos_dist": td.Categorical(logits=pos_logits),
        # }
        ###
        logits = self.linear(x)[-1]
        return td.Categorical(logits=logits)

    def sample(self, batch_size, max_length, device):
        assert max_length <= self.max_len
        preds = self.vocab.bos * torch.ones((1, batch_size), dtype=torch.long, device=device)
        finished = torch.zeros((batch_size), dtype=torch.bool, device=device)
        imag_smiles_lens = torch.ones((batch_size),  device=device)
        
        ### Change action type
        # sequences = [torch.tensor([self.vocab.bos], dtype=torch.long, device=device) for _ in range(batch_size)]
        # finished = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        # seq_lens = torch.ones((batch_size,), dtype=torch.long, device=device)

        # max_steps = 100
        
        # with torch.no_grad():
        #     for step in range(max_steps):
        #         for i in range(batch_size):
        #             if finished[i]:
        #                 continue
                    
        #             seq = preds[i]
        #             seq_input = seq.unsqueeze(1)
                    
        #             preds_dist = self.forward(seq_input)
        #             op = preds_dist["op_dist"].sample().item()
        #             token = preds_dist["token_dist"].sample().item()
        #             pos = preds_dist["pos_dist"].sample().item()
        #             new_seq = self.apply_action(seq, op, token, pos)
        #             preds[i] = new_seq
        #             seq_lens[i] += 1
        #             imag_smiles_lens += ~finished

        #             if (new_seq[-1] == self.vocab.eos).item():
        #                 finished[i] = True
        #             if torch.prod(finished) == 1: break
        #             # EOS_sampled = (preds[-1] == self.vocab.eos)
        #             # finished = torch.ge(finished + EOS_sampled, 1)
        #             # if torch.prod(finished) == 1: break
        # # Pad and stack sequences to return a uniform tensor (optional)
        # max_len = max([len(seq) for seq in sequences])
        # padded_seqs = torch.full((batch_size, max_len), self.vocab.pad, dtype=torch.long, device=device)
        # for i, seq in enumerate(sequences):
        #     padded_seqs[i, :len(seq)] = seq

        # # Convert to list of lists
        # return padded_seqs.tolist(), seq_lens.tolist()
        ###

        with torch.no_grad():
            for i in range(1, max_length + 1):
                preds_dist = self.forward(preds)
                next_preds = preds_dist.sample()[-1].view(1, -1)
                preds = torch.cat([preds, next_preds], dim=0)
                imag_smiles_lens += ~finished

                EOS_sampled = (preds[-1] == self.vocab.eos)
                finished = torch.ge(finished + EOS_sampled, 1)
        #         if torch.prod(finished) == 1: break
        
        imag_smiles = preds.T.tolist()     
        return imag_smiles, imag_smiles_lens[0].tolist()
    
    def apply_action(self, seq, op, token=None, pos=None):
        new_seq = seq.clone()
        if op == 0 and pos < len(seq): # ADD
            new_seq = torch.cat([new_seq[:pos], torch.tensor([token], device=seq.device), new_seq[pos:]])
        elif op == 1 and pos < len(seq):
            new_seq = torch.cat([new_seq[:pos], new_seq[pos+1:]])
        # elif op == "REPLACE" and pos < len(seq):
        #     new_seq[pos] = token
        return new_seq

    def get_likelihood(self, obs, nonterms, actions=None):      
        ### Change action type
        # dist = self.forward(obs[:-1])  # Get action distributions for each step

        # logprob_op = dist["op_dist"].log_prob(actions["op"])     # (T, B)
        # logprob_token = dist["token_dist"].log_prob(actions["token"])  # (T, B)
        # logprob_pos = dist["pos_dist"].log_prob(actions["pos"])   # (T, B)

        # total_logprob = (logprob_op + logprob_token + logprob_pos) * nonterms

        # return total_logprob
        ###
        
        dist = self.forward(obs[:-1])
        logprobs = dist.log_prob(obs[1:]) * nonterms[:-1]
        # print(logprobs.shape)
        # print(dist.logits.shape)
        # print(dist.probs.shape)
        # exit()
        log_of_probs = F.log_softmax(dist.logits, dim=-1) * nonterms[:-1].unsqueeze(-1)
        action_probs = dist.probs * nonterms[:-1].unsqueeze(-1)

        # print(log_of_probs)
        # print(action_probs)
        # print(torch.log(action_probs))
        # exit()

        return logprobs, log_of_probs, action_probs

    def get_data(self, batch_size, max_length, device, train_steps, last_eval=False, max_steps=100):
        if max_length is None:
            max_length = self.max_len
        else:
            assert max_length <= self.max_len
            
        obs = torch.zeros((max_length + 1, batch_size), dtype=torch.long, device=device)
        obs[0] = self.vocab.bos
        ### Change action type
        # actions = {
        #     "op": torch.zeros((max_steps, batch_size), dtype=torch.long, device=device),
        #     "token": torch.zeros((max_steps, batch_size), dtype=torch.long, device=device),
        #     "pos": torch.zeros((max_steps, batch_size), dtype=torch.long, device=device)
        # }
        ###
        
        if last_eval:
            qed_target = [0.2761] * batch_size
            logp_target = [8.1940] * batch_size
            logp_target= pd.Series(list(logp_target)).apply(self.adjust_value)
            logp_target = [self.tokenize_target_LOGP(target) for target in logp_target]
            qed_target = pd.Series(list(qed_target)).apply(self.adjust_value)
            qed_target = [self.tokenize_target_QED(target) for target in qed_target]
        
            column_idx = 0
            for logp_target, qed_target in zip(logp_target, qed_target):
                logp_tensor = torch.tensor(self.vocab.encode(logp_target))
                qed_tensor = torch.tensor(self.vocab.encode(qed_target))
                
                obs[0: len(logp_tensor), column_idx] = logp_tensor
                obs[len(logp_tensor): len(logp_tensor) + len(qed_tensor), column_idx] = qed_tensor
                obs[len(logp_tensor) + len(qed_tensor), column_idx] = self.vocab.bos
                column_idx += 1
        else: 
            batch_n = train_steps // 100
            self.add_targets(batch_size, obs, batch_n)
        start_idx = 17
        # start_idx = 1
        
        nonterms = torch.zeros((max_length + 1, batch_size), dtype=torch.bool, device=device)
        rewards = torch.zeros((max_length, batch_size), dtype=torch.float32, device=device)
        end_flags = torch.zeros((1, batch_size), dtype=torch.bool, device=device)

        for i in range(start_idx, max_length+1):
            preds_dist = self.autoregress(obs[:i])
            
            preds = preds_dist.sample()
 
            obs[i] = preds
            nonterms[i-1] = ~end_flags
            
            EOS_sampled = (preds == self.vocab.eos)
            rewards[i-1] = EOS_sampled * (~end_flags)

            #check if all sequences are done
            end_flags = torch.ge(end_flags + EOS_sampled, 1)
            if torch.prod(end_flags) == 1: break

        if i == max_length:
            rewards[-1] = rewards[-1] + (~end_flags)

        #remove assertion afterwards
        assert rewards.sum() == batch_size

        obs = obs[:i+1]
        nonterms = nonterms[:i+1]
        rewards = rewards[:i]
        episode_lens = nonterms.sum(0).cpu()

        return obs, rewards, nonterms, episode_lens, None
    
    def add_targets(self, batch_size, obs, batch_n):
             
        with open("data/batches_512.json", "r") as f:
            batches = json.load(f)
        
        sampled_qed_targets_512 = [float(x) for x in batches[0][0]]
        sampled_logp_targets_512 = [float(x) for x in batches[0][1]]
        
        if batch_n > 0:
            # Get 16 random targets from the batches[batch_n] and switch them with other targets in the sampled targets
            for i in range(batch_n - 1 * 16, batch_n * 16):
                sampled_qed_targets_512[i] = float(batches[batch_n][0][i])
                sampled_logp_targets_512[i] = float(batches[batch_n][1][i])
                 
        # sampled_qed_targets_64 = [0.8065, 0.8125, 0.8080, 0.8937, 0.7975, 0.8121, 0.8245, 0.8051, 0.8178, 0.8009, 0.8464, 0.8011, 0.8060, 0.8051, 0.8007, 0.9408, 0.9210, 0.8031, 0.8290, 0.7949, 0.8299, 0.9125, 0.80, 0.7962, 0.7766, 0.8169, 0.7721, 0.7348, 0.8064, 0.8065, 0.7984, 0.8542, 0.8837, 0.8025, 0.8497, 0.7848, 0.9411, 0.8067, 0.8010, 0.8633, 0.8338, 0.9171, 0.8077, 0.7721, 0.6116, 0.7301, 0.8038, 0.7602, 0.8552, 0.6304, 0.7980, 0.8476, 0.8395, 0.9217, 0.8061, 0.7952, 0.8040, 0.8044, 0.8485, 0.8009, 0.7998, 0.9304, 0.8479, 0.8754]
        # sampled_logp_targets_64 = [3.1364, 3.4829, 2.9671, 2.9141, 2.2230, 3.2770, 3.6220, 2.5314, 3.3136, 3.3588, 3.1831, 2.8280, 3.4449, 3.4814, 3.6057, 3.0780, 3.4048, 2.6587, 3.4527, 1.9232, 2.8381, 2.7728, 3.7658, 3.7898, 3.2520, 3.1077, 3.1905, 3.1236, 3.4464, 2.5680, 3.2973, 3.1092, 2.7750, 2.5282, 3.2481, 1.9146, 3.2343, 2.8764, 2.2845, 3.4176, 2.6688, 3.3950, 3.1730, 1.3096, 3.6431, 2.9403, 3.4364, 3.3885, 3.0314, 3.0678, 3.5384, 2.8467, 2.9031, 3.5439, 3.3121, 2.5766, 2.8871, 2.8366, 2.5171, 3.6672, 3.3059, 3.3036, 3.0726, 2.9964]
        
        sampled_qed_targets_128 = [0.8065, 0.8125, 0.8080, 0.8937, 0.7975, 0.8121, 0.8245, 0.8051, 0.8178, 0.8009, 0.8464, 0.8011, 0.8060, 0.8051, 0.8007, 0.9408, 0.9210, 0.8031, 0.8290, 0.7949, 0.8299, 0.9125, 0.80, 0.7962, 0.7766, 0.8169, 0.7721, 0.7348, 0.8064, 0.8065, 0.7984, 0.8542, 0.8837, 0.8025, 0.8497, 0.7848, 0.9411, 0.8067, 0.8010, 0.8633, 0.8338, 0.9171, 0.8077, 0.7721, 0.6116, 0.7301, 0.8038, 0.7602, 0.8552, 0.6304, 0.7980, 0.8476, 0.8395, 0.9217, 0.8061, 0.7952, 0.8040, 0.8044, 0.8485, 0.8009, 0.7998, 0.9304, 0.8479, 0.8754, 0.7929, 0.8708, 0.8420, 0.8331, 0.8269, 0.7491, 0.7738, 0.8210, 0.8622, 0.8074, 0.9251, 0.6118, 0.9233, 0.7808, 0.8003, 0.7706, 0.7946, 0.8039, 0.7801, 0.8040, 0.7739, 0.8808, 0.8388, 0.6253, 0.7875, 0.8982, 0.8079, 0.8052, 0.9291, 0.8938, 0.8856, 0.8006, 0.8011, 0.7982, 0.7876, 0.8088, 0.8465, 0.7672, 0.8792, 0.7904, 0.8035, 0.8473, 0.9432, 0.9197, 0.7429, 0.8890, 0.8567, 0.9147, 0.8036, 0.6274, 0.7837, 0.8403, 0.8037, 0.7983, 0.8590, 0.7569, 0.7975, 0.7454, 0.8524, 0.7617, 0.7358, 0.7220, 0.8054, 0.9229] 
        sampled_logp_targets_128 = [3.1364, 3.4829, 2.9671, 2.9141, 2.2230, 3.2770, 3.6220, 2.5314, 3.3136, 3.3588, 3.1831, 2.8280, 3.4449, 3.4814, 3.6057, 3.0780, 3.4048, 2.6587, 3.4527, 1.9232, 2.8381, 2.7728, 3.7658, 3.7898, 3.2520, 3.1077, 3.1905, 3.1236, 3.4464, 2.5680, 3.2973, 3.1092, 2.7750, 2.5282, 3.2481, 1.9146, 3.2343, 2.8764, 2.2845, 3.4176, 2.6688, 3.3950, 3.1730, 1.3096, 3.6431, 2.9403, 3.4364, 3.3885, 3.0314, 3.0678, 3.5384, 2.8467, 2.9031, 3.5439, 3.3121, 2.5766, 2.8871, 2.8366, 2.5171, 3.6672, 3.3059, 3.3036, 3.0726, 2.9964, 2.6771, 3.3562, 3.1465, 3.5920, 2.9055, 2.7410, 2.8452, 2.9384, 2.7991, 3.2755, 2.9959, 3.4830, 3.1178, 2.9836, 2.7538, 1.9318, 3.0504, 2.6673, 2.5368, 2.9970, 1.6009, 3.0144, 2.5383, 3.3762, 1.6180, 2.4644, 3.1062, 2.3621, 2.6595, 3.4550, 2.8451, 3.2637, 3.4660, 3.2214, 3.490, 2.7978, 2.6864, 3.3145, 2.3393, 2.5196, 2.5057, 3.2116, 2.9435, 3.4948, 2.6838, 2.7648, 3.1507, 2.7159, 2.3364, 3.6728, 3.8468, 2.9841, 3.0365, 3.1816, 3.5235, 3.9561, 3.6205, 3.4320, 2.9033, 2.2955, 2.5186, 3.6149, 3.4028, 3.2257]
        
        sampled_qed_targets_256 = [0.8065, 0.8125, 0.8080, 0.8937, 0.7975, 0.8121, 0.8245, 0.8051, 0.8178, 0.8009, 0.8464, 0.8011, 0.8060, 0.8051, 0.8007, 0.9408, 0.9210, 0.8031, 0.8290, 0.7949, 0.8299, 0.9125, 0.80, 0.7962, 0.7766, 0.8169, 0.7721, 0.7348, 0.8064, 0.8065, 0.7984, 0.8542, 0.8837, 0.8025, 0.8497, 0.7848, 0.9411, 0.8067, 0.8010, 0.8633, 0.8338, 0.9171, 0.8077, 0.7721, 0.6116, 0.7301, 0.8038, 0.7602, 0.8552, 0.6304, 0.7980, 0.8476, 0.8395, 0.9217, 0.8061, 0.7952, 0.8040, 0.8044, 0.8485, 0.8009, 0.7998, 0.9304, 0.8479, 0.8754, 0.7929, 0.8708, 0.8420, 0.8331, 0.8269, 0.7491, 0.7738, 0.8210, 0.8622, 0.8074, 0.9251, 0.6118, 0.9233, 0.7808, 0.8003, 0.7706, 0.7946, 0.8039, 0.7801, 0.8040, 0.7739, 0.8808, 0.8388, 0.6253, 0.7875, 0.8982, 0.8079, 0.8052, 0.9291, 0.8938, 0.8856, 0.8006, 0.8011, 0.7982, 0.7876, 0.8088, 0.8465, 0.7672, 0.8792, 0.7904, 0.8035, 0.8473, 0.9432, 0.9197, 0.7429, 0.8890, 0.8567, 0.9147, 0.8036, 0.6274, 0.7837, 0.8403, 0.8037, 0.7983, 0.8590, 0.7569, 0.7975, 0.7454, 0.8524, 0.7617, 0.7358, 0.7220, 0.8054, 0.9229, 0.9142, 0.7946, 0.9384, 0.7940, 0.7758, 0.8151, 0.9084, 0.8427, 0.7838, 0.8010, 0.8015, 0.8188, 0.7548, 0.8571, 0.8001, 0.8994, 0.8029, 0.7928, 0.8873, 0.9020, 0.7996, 0.8075, 0.6099, 0.7771, 0.9351, 0.9123, 0.7732, 0.9178, 0.8170, 0.7630, 0.7746, 0.8056, 0.8088, 0.8901, 0.9479, 0.9044, 0.7690, 0.7801, 0.8004, 0.9245, 0.7498, 0.8508, 0.9083, 0.9095, 0.8058, 0.8352, 0.7691, 0.7528, 0.7971, 0.9141, 0.8014, 0.8256, 0.8891, 0.7997, 0.8295, 0.7881, 0.7998, 0.8046, 0.7835, 0.9208, 0.7781, 0.7911, 0.8591, 0.9191, 0.6665, 0.7957, 0.8037, 0.8896, 0.6019, 0.8904, 0.8327, 0.7641, 0.8184, 0.7949, 0.7718, 0.7981, 0.8052, 0.7974, 0.8497, 0.6226, 0.940, 0.8490, 0.7756, 0.8518, 0.8725, 0.7609, 0.7534, 0.7448, 0.6170, 0.8916, 0.8506, 0.8494, 0.7989, 0.7861, 0.8256, 0.9174, 0.8602, 0.7765, 0.7966, 0.6125, 0.9048, 0.7923, 0.7055, 0.9160, 0.6133, 0.9309, 0.7752, 0.7693, 0.7708, 0.8286, 0.9148, 0.8280, 0.9143, 0.7616, 0.8250, 0.9117, 0.8305, 0.7914, 0.9360, 0.7899, 0.7761, 0.7878, 0.7761, 0.7075, 0.8042, 0.8406, 0.8003, 0.9406] 
        sampled_logp_targets_256 = [3.1364, 3.4829, 2.9671, 2.9141, 2.2230, 3.2770, 3.6220, 2.5314, 3.3136, 3.3588, 3.1831, 2.8280, 3.4449, 3.4814, 3.6057, 3.0780, 3.4048, 2.6587, 3.4527, 1.9232, 2.8381, 2.7728, 3.7658, 3.7898, 3.2520, 3.1077, 3.1905, 3.1236, 3.4464, 2.5680, 3.2973, 3.1092, 2.7750, 2.5282, 3.2481, 1.9146, 3.2343, 2.8764, 2.2845, 3.4176, 2.6688, 3.3950, 3.1730, 1.3096, 3.6431, 2.9403, 3.4364, 3.3885, 3.0314, 3.0678, 3.5384, 2.8467, 2.9031, 3.5439, 3.3121, 2.5766, 2.8871, 2.8366, 2.5171, 3.6672, 3.3059, 3.3036, 3.0726, 2.9964, 2.6771, 3.3562, 3.1465, 3.5920, 2.9055, 2.7410, 2.8452, 2.9384, 2.7991, 3.2755, 2.9959, 3.4830, 3.1178, 2.9836, 2.7538, 1.9318, 3.0504, 2.6673, 2.5368, 2.9970, 1.6009, 3.0144, 2.5383, 3.3762, 1.6180, 2.4644, 3.1062, 2.3621, 2.6595, 3.4550, 2.8451, 3.2637, 3.4660, 3.2214, 3.490, 2.7978, 2.6864, 3.3145, 2.3393, 2.5196, 2.5057, 3.2116, 2.9435, 3.4948, 2.6838, 2.7648, 3.1507, 2.7159, 2.3364, 3.6728, 3.8468, 2.9841, 3.0365, 3.1816, 3.5235, 3.9561, 3.6205, 3.4320, 2.9033, 2.2955, 2.5186, 3.6149, 3.4028, 3.2257, 2.9787, 3.3054, 3.6035, 3.2821, 3.0061, 3.4746, 2.9699, 3.1405, 2.6752, 3.5043, 3.1280, 3.7913, 3.3462, 3.0706, 2.9757, 3.0888, 3.2834, 3.7744, 3.4274, 3.6407, 3.4574, 3.1979, 2.8780, 3.3480, 3.1926, 3.3181, 3.0324, 2.9521, 2.8779, 3.7955, 3.1879, 2.8507, 2.6520, 3.2793, 2.9592, 3.1876, 3.4746, 2.6350, 3.1366, 3.6492, 3.2627, 3.1917, 3.4477, 3.6031, 2.7071, 3.4227, 3.6969, 3.3165, 1.9007, 2.3511, 3.6423, 2.4698, 2.4198, 3.6415, 2.4340, 1.9677, 2.9334, 2.1928, 3.0625, 3.1141, 3.7653, 3.4271, 2.5042, 2.5091, 3.6164, 1.7314, 3.5965, 2.6758, 3.7786, 2.8083, 2.9117, 3.1037, 2.4624, 2.7136, 2.5015, 2.0537, 2.7012, 3.4957, 2.3478, 3.4135, 3.6825, 2.6774, 2.6815, 3.0788, 3.2424, 2.8256, 3.0934, 2.7839, 3.8455, 3.0834, 3.4915, 2.9168, 2.2316, 3.1476, 2.8218, 3.1934, 2.7689, 3.6118, 3.3054, 3.0381, 3.6253, 2.4529, 3.3777, 2.7828, 2.4087, 3.2907, 2.5984, 3.6262, 1.6593, 2.6033, 3.1382, 3.2932, 3.0410, 3.8946, 2.8077, 2.9694, 2.3075, 1.6795, 2.8060, 2.6382, 2.8324, 2.9434, 3.0939, 3.6861, 2.8895, 3.0446, 2.9975, 3.0740]
        
        target_indices = np.random.uniform(0, 512, size=batch_size)
        qed_targets = [sampled_qed_targets_512[int(idx)] for idx in target_indices]
        logp_targets = [sampled_logp_targets_512[int(idx)] for idx in target_indices]
        
        # logp_targets = np.random.uniform(-3, 20, size=batch_size)
        logp_targets= pd.Series(list(logp_targets)).apply(self.adjust_value)
        logp_targets = [self.tokenize_target_LOGP(target) for target in logp_targets]
        
        # qed_targets = np.random.choice(sampled_targets, size=batch_size)  # Sample from list
        # qed_target = np.random.choice(sampled_targets, size=1)  # Sample one value
        # qed_targets = np.full(batch_size, qed_target)  # Repeat it for the batch
        # qed_targets = np.random.uniform(0, 1, size=batch_size)
        qed_targets = pd.Series(list(qed_targets)).apply(self.adjust_value)
        qed_targets = [self.tokenize_target_QED(target) for target in qed_targets]
        
        column_idx = 0
        for logp_target, qed_target in zip(logp_targets, qed_targets):
            logp_tensor = torch.tensor(self.vocab.encode(logp_target))
            qed_tensor = torch.tensor(self.vocab.encode(qed_target))
            
            obs[0: len(logp_tensor), column_idx] = logp_tensor
            obs[len(logp_tensor): len(logp_tensor) + len(qed_tensor), column_idx] = qed_tensor
            obs[len(logp_tensor) + len(qed_tensor), column_idx] = self.vocab.bos
            column_idx += 1
            # row[0: len(logp_target)] = logp_target
            # row[len(logp_target): len(logp_target) + len(qed_target)] = qed_target
            # row[len(logp_target) + len(qed_target)] = self.vocab.bos
            
        return obs
    
    def adjust_value(self, val, length=6):
        """Adjust value to match the specified length in string representation."""
        # Round the value to match the desired number of decimals
        decimals = max(0, length - len(str(int(val ))) - 2)  # Account for "0." or "-0."
        rounded_val = round(val, decimals)
        
        # Convert to string and pad with trailing zeros if needed
        value_str = f"{rounded_val:.{decimals}f}"  # Ensure the value has the correct decimal places
        if len(value_str) < length:
            # Pad with zeros if the string is shorter than the desired length
            padding = length - len(value_str)
            value_str += "0" * padding
        
        # Convert back to float and return
        return value_str

    def tokenize_target_LOGP(self, target):
        # item_str = str(target)
        transformed = ['LOGP'] + list(target) + ['\\LOGP']
        return transformed
    
    def tokenize_target_QED(self, target):
        item_str = str(target)
        transformed = ['QED'] + list(item_str) + ['\\QED']
        return transformed
    
    def tokenize_target_HOMO_LUMO(self, target):
        item_str = str(target)
        transformed = ['HOMO_LUMO'] + list(item_str) + ['\\HOMO_LUMO']
        return transformed
    
    def get_save_dict(self):
        return {
            "embedding": self.embedding.state_dict(),
            "position_embedding": self.position_embedding.state_dict(),
            "encoder": self.encoder.state_dict(),
            ### Change action type
            # "op_head": self.op_head.state_dict(),
            # "token_head": self.token_head.state_dict(),
            # "pos_head": self.pos_head.state_dict(),
            ###
            "linear": self.linear.state_dict(),
        }

    def load_save_dict(self, saved_dict):
        self.embedding.load_state_dict(saved_dict["embedding"])
        self.position_embedding.load_state_dict(saved_dict["position_embedding"])
        self.encoder.load_state_dict(saved_dict["encoder"])
        
        ### Change action type
        # Use the old linear weights for token prediction head
        # old_linear = saved_dict["linear"]
        # self.token_head.load_state_dict(old_linear)
        ###
        
        self.linear.load_state_dict(saved_dict["linear"])

class RnnPolicy(nn.Module):
    def __init__(self, vocab, embedding_size, hidden_size, num_layers):
        super(RnnPolicy, self).__init__()
        self.vocab = vocab
        self.embedding = nn.Embedding(len(vocab), embedding_size, padding_idx=vocab.pad, dtype=torch.float32)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, len(vocab))
    
    def forward(self, x, lengths, hiddens=None):
        x = self.embedding(x)
        x = rnn_utils.pack_padded_sequence(x, lengths, enforce_sorted=False)
        x, hiddens = self.rnn(x, hiddens)
        x, _ = rnn_utils.pad_packed_sequence(x)
        logits = self.linear(x)
        return td.Categorical(logits=logits), lengths, hiddens

    def sample(self, batch_size, max_length, device):
        starts = self.vocab.bos * torch.ones((1, batch_size), dtype=torch.long, device=device)
        finished = torch.zeros((1, batch_size), dtype=torch.bool, device=device)

        imag_smiles = [starts]
        imag_smiles_lens = torch.ones((1, batch_size),  device=device)
        
        input_lens = torch.ones(batch_size)
        hiddens = None
        with torch.no_grad():
            for i in range(1, max_length + 1):
                preds_dist, _, hiddens = self.forward(starts, input_lens, hiddens)
                preds = preds_dist.sample()
                imag_smiles.append(preds)
                imag_smiles_lens += ~finished
                starts = preds

                EOS_sampled = (preds == self.vocab.eos)
                finished = torch.ge(finished + EOS_sampled, 1)
                if torch.prod(finished) == 1: break

        imag_smiles = torch.cat(imag_smiles, 0).T.tolist()
        return imag_smiles, imag_smiles_lens[0].tolist()
    
    def get_likelihood(self, obs, smiles_len, nonterms):      
        dist, _, _ = self.forward(obs[:-1], smiles_len)
        logprobs = dist.log_prob(obs[1:]) * nonterms[:-1]
        return logprobs

    def get_data(self, batch_size, max_length, device):
        obs = torch.zeros((max_length + 1, batch_size), dtype=torch.long, device=device)
        obs[0] = self.vocab.bos
        nonterms = torch.zeros((max_length + 1, batch_size), dtype=torch.bool, device=device)
        rewards = torch.zeros((max_length, batch_size), dtype=torch.float32, device=device)
        end_flags = torch.zeros((1, batch_size), dtype=torch.bool, device=device)

        input_lens = torch.ones(batch_size)
        hiddens = None
        for i in range(1, max_length+1):
            preds_dist, _, hiddens = self.forward(obs[i-1].view(1, -1), input_lens, hiddens)
            preds = preds_dist.sample()

            obs[i] = preds
            nonterms[i-1] = ~end_flags
            
            EOS_sampled = (preds == self.vocab.eos)
            rewards[i-1] = EOS_sampled * (~end_flags)

            #check if all sequences are done
            end_flags = torch.ge(end_flags + EOS_sampled, 1)
            if torch.prod(end_flags) == 1: break

        if i == max_length:
            rewards[-1] = rewards[-1] + (~end_flags)

        #remove assertion afterwards
        assert rewards.sum() == batch_size

        obs = obs[:i+1]
        nonterms = nonterms[:i+1]
        rewards = rewards[:i]
        episode_lens = nonterms.sum(0).cpu()

        return obs, rewards, nonterms, episode_lens
    
    def get_save_dict(self):
        return {
            "embedding": self.embedding.state_dict(),
            "rnn": self.rnn.state_dict(),
            "linear": self.linear.state_dict(),
        }

    def load_save_dict(self, saved_dict):
        self.embedding.load_state_dict(saved_dict["embedding"])
        self.rnn.load_state_dict(saved_dict["rnn"])
        self.linear.load_state_dict(saved_dict["linear"])

class FcPolicy(nn.Module):
    def __init__(self, vocab, max_len, embedding_size, hidden_size):
        super(FcPolicy, self).__init__()
        self.vocab = vocab
        self.max_len = max_len
        self.embedding = nn.Embedding(len(vocab), embedding_size, padding_idx=vocab.pad, dtype=torch.float32)
        self.fc1 = nn.Linear(embedding_size * max_len, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, len(vocab))

        self.register_buffer('triu', torch.tril(torch.ones(max_len, max_len))) #L,L

    def forward(self, x):
        B, L = x.shape
        x = self.embedding(x) #B, L, C

        x = (x.view(B, 1, L, -1) * self.triu.unsqueeze(-1)).flatten(start_dim=-2) #(B, 1, L, C) * (L, L, 1) -> (B, L, L, C).flatten(start_dim=-2) -> (B, L, L*C)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return td.Categorical(logits=logits)

    def forward_last(self, x):
        x = self.embedding(x).flatten(start_dim=-2) #(B, L)-> (B, L*C)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return td.Categorical(logits=logits)
    
    def sample(self, batch_size, max_length, device):
        assert max_length <= self.max_len

        imag_smiles = self.vocab.pad * torch.ones((self.max_len + 1, batch_size), dtype=torch.long, device=device)
        imag_smiles_lens = torch.ones((1, batch_size),  device=device)
        imag_smiles[0] = self.vocab.bos
        finished = torch.zeros((1, batch_size), dtype=torch.bool, device=device)
        with torch.no_grad():
            for i in range(1, max_length + 1):
                preds_dist = self.forward_last(imag_smiles[:-1].T)
                preds = preds_dist.sample()
                imag_smiles[i] = preds
                imag_smiles_lens += ~finished

                EOS_sampled = (preds == self.vocab.eos)
                finished = torch.ge(finished + EOS_sampled, 1)
                if torch.prod(finished) == 1: break
    
        imag_smiles = imag_smiles.T.tolist()
        return imag_smiles, imag_smiles_lens[0].tolist()

    def get_likelihood(self, obs, nonterms):      
        dist = self.forward(obs[:-1].T)
        logprobs = dist.log_prob(obs[1:].T) * nonterms[:-1].T
        return logprobs.T

    def get_data(self, batch_size, max_length, device):
        if max_length is None:
            max_length = self.max_len
        else:
            assert max_length <= self.max_len
        
        obs = self.vocab.pad * torch.ones((self.max_len + 1, batch_size), dtype=torch.long, device=device)
        obs[0] = self.vocab.bos
        nonterms = torch.zeros((max_length + 1, batch_size), dtype=torch.bool, device=device)
        rewards = torch.zeros((max_length, batch_size), dtype=torch.float32, device=device)
        end_flags = torch.zeros((1, batch_size), dtype=torch.bool, device=device)

        for i in range(1, max_length+1):
            preds_dist = self.forward_last(obs[:-1].T)
            preds = preds_dist.sample()
            
            obs[i] = preds
            nonterms[i-1] = ~end_flags
            
            EOS_sampled = (preds == self.vocab.eos)
            rewards[i-1] = EOS_sampled * (~end_flags)

            #check if all sequences are done
            end_flags = torch.ge(end_flags + EOS_sampled, 1)
            
            if torch.prod(end_flags) == 1: break
        
        if i == max_length:
            rewards[-1] = rewards[-1] + (~end_flags)
        
        #remove assertion afterwards
        assert rewards.sum() == batch_size

        obs = obs#[:i+1]
        nonterms = nonterms#[:i+1]
        rewards = rewards#[:i]
        episode_lens = nonterms[:i+1].sum(0).cpu()

        return obs, rewards, nonterms, episode_lens
    
    def get_save_dict(self):
        return {
            "embedding": self.embedding.state_dict(),
            "fc1": self.fc1.state_dict(),
            "fc2": self.fc2.state_dict(),
            "fc3": self.fc3.state_dict(),
        }

    def load_save_dict(self, saved_dict):
        self.embedding.load_state_dict(saved_dict["embedding"])
        self.fc1.load_state_dict(saved_dict["fc1"])
        self.fc2.load_state_dict(saved_dict["fc2"])
        self.fc3.load_state_dict(saved_dict["fc3"])
        
        
# sampled_targets = [0.1310, 0.1378, 0.1273, 0.1306, 0.1309, 0.1302,
    #    0.1269, 0.1651, 0.1308, 0.1380, 0.1310, 0.1726,
    #    0.1331, 0.1363, 0.1303, 0.1726, 0.1310, 0.1301,
    #    0.1309, 0.1269, 0.1644, 0.1648, 0.1306, 0.1686,
    #    0.1363, 0.1725, 0.1302, 0.1303, 0.1274, 0.1363,
    #    0.1333, 0.1310, 0.1330, 0.1649, 0.1686, 0.1412,
    #    0.1686, 0.1413, 0.1273, 0.1278, 0.1614, 0.1278,
    #    0.1270, 0.1644, 0.1333, 0.1688, 0.1329, 0.1307,
    #    0.1307, 0.1305, 0.1273, 0.1726, 0.1331, 0.1299,
    #    0.130, 0.1332, 0.1652, 0.1309, 0.1727, 0.1686,
    #    0.1333, 0.1310, 0.1686, 0.1310] # 64 samples
        
    #     sampled_targets = [0.1608, 0.1334, 0.1273, 0.1301, 0.1270, 0.1274, 0.1651, 0.1788
    #    , 0.1410, 0.1654, 0.1728, 0.1657, 0.1687, 0.1369, 0.1332, 0.1723
    #    , 0.1608, 0.1689, 0.1299, 0.1612, 0.1659, 0.1286, 0.1652, 0.1321
    #    , 0.1369, 0.1362, 0.1272, 0.1304, 0.1306, 0.1369, 0.1274, 0.1728
    #    , 0.1317, 0.1654, 0.1321, 0.1311, 0.1657, 0.1613, 0.1303, 0.1789
    #    , 0.1288, 0.1789, 0.1649, 0.1664, 0.1269, 0.1314, 0.1618, 0.1380
    #    , 0.1649, 0.1411, 0.1303, 0.1313, 0.1687, 0.130, 0.1329, 0.1333
    #    , 0.1656, 0.1270, 0.1745, 0.1377, 0.1269, 0.1608, 0.1321, 0.1608
    #    , 0.1302, 0.1654, 0.1685, 0.1648, 0.1736, 0.1745, 0.1292, 0.1299
    #    , 0.1299, 0.1365, 0.1410, 0.1306, 0.1684, 0.1653, 0.1789, 0.1334
    #    , 0.1617, 0.1729, 0.1789, 0.1653, 0.130, 0.1362, 0.1644, 0.1661
    #    , 0.1685, 0.1322, 0.1331, 0.1662, 0.1413, 0.1658, 0.1278, 0.1650
    #    , 0.1292, 0.1656, 0.1273, 0.1322, 0.1727, 0.1327, 0.1617, 0.1645
    #    , 0.1618, 0.1729, 0.1303, 0.1684, 0.1321, 0.1609, 0.1412, 0.1299
    #    , 0.1264, 0.1321, 0.1305, 0.1319, 0.1789, 0.1673, 0.1724, 0.1275
    #    , 0.1303, 0.1332, 0.1332, 0.1688, 0.1735, 0.1654, 0.1747, 0.1231] # 128 samples
        
    #     sampled_targets = [0.1319, 0.1285, 0.1273, 0.1747, 0.130, 0.1306, 0.1788, 0.1746
    #    , 0.1334, 0.1785, 0.1316, 0.1668, 0.1686, 0.1627, 0.1302, 0.1641
    #    , 0.1316, 0.1720, 0.1298, 0.1645, 0.1372, 0.1674, 0.1656, 0.1719
    #    , 0.1627, 0.1219, 0.1307, 0.1309, 0.1652, 0.1627, 0.1651, 0.1682
    #    , 0.1213, 0.1630, 0.1719, 0.1656, 0.1668, 0.1366, 0.1304, 0.1374
    #    , 0.1234, 0.1294, 0.1330, 0.1667, 0.1276, 0.1324, 0.1624, 0.1614
    #    , 0.1327, 0.1650, 0.1304, 0.1737, 0.1378, 0.1725, 0.1618, 0.1274
    #    , 0.1263, 0.130, 0.1315, 0.1683, 0.1651, 0.1319, 0.1719, 0.1319
    #    , 0.1274, 0.1630, 0.1414, 0.1736, 0.1557, 0.1665, 0.1672, 0.1328
    #    , 0.1328, 0.1670, 0.1748, 0.1301, 0.1635, 0.1746, 0.1374, 0.1370
    #    , 0.1418, 0.1265, 0.1374, 0.1746, 0.1329, 0.1604, 0.1325, 0.1260
    #    , 0.1414, 0.1265, 0.1687, 0.1220, 0.1643, 0.1409, 0.1297, 0.1271
    #    , 0.1672, 0.1263, 0.1273, 0.1681, 0.1230, 0.1661, 0.1640, 0.1287
    #    , 0.1624, 0.1265, 0.1332, 0.1635, 0.1610, 0.1381, 0.1727, 0.1328
    #    , 0.1634, 0.1293, 0.1275, 0.1669, 0.1669, 0.1602, 0.1226, 0.1310
    #    , 0.1304, 0.1302, 0.1302, 0.1413, 0.1359, 0.1785, 0.1732, 0.1633
    #    , 0.1310, 0.1330, 0.1665, 0.1674, 0.1720, 0.1304, 0.1372, 0.1644
    #    , 0.1670, 0.1724, 0.1636, 0.1294, 0.1740, 0.1328, 0.1331, 0.1310
    #    , 0.1272, 0.1612, 0.1608, 0.1642, 0.1720, 0.1332, 0.1411, 0.1674
    #    , 0.1734, 0.1686, 0.1647, 0.1606, 0.1645, 0.1318, 0.1674, 0.1739
    #    , 0.1271, 0.1618, 0.1666, 0.1738, 0.1649, 0.1304, 0.1411, 0.1617
    #    , 0.1790, 0.1734, 0.1745, 0.1234, 0.1409, 0.1321, 0.1607, 0.1294
    #    , 0.1641, 0.1725, 0.1275, 0.1313, 0.1289, 0.1277, 0.1669, 0.1275
    #    , 0.1363, 0.1619, 0.1653, 0.1371, 0.1317, 0.0979, 0.1725, 0.1275
    #    , 0.1375, 0.1605, 0.1624, 0.1634, 0.1332, 0.1644, 0.1681, 0.1220
    #    , 0.1628, 0.1740, 0.1781, 0.1296, 0.1611, 0.1617, 0.1790, 0.1607
    #    , 0.0979, 0.1621, 0.1334, 0.1228, 0.1727, 0.1650, 0.1688, 0.1673
    #    , 0.1304, 0.1220, 0.1738, 0.1332, 0.1311, 0.1660, 0.1688, 0.1675
    #    , 0.1729, 0.1233, 0.1318, 0.1227, 0.1739, 0.1672, 0.1613, 0.1609
    #    , 0.1372, 0.1648, 0.1362, 0.1279, 0.1332, 0.1281, 0.1730, 0.1670
    #    , 0.1656, 0.1219, 0.1409, 0.1379, 0.1667, 0.1744, 0.1274, 0.1316
    #    , 0.1737, 0.1621, 0.1273, 0.1601, 0.1657, 0.1626, 0.1790, 0.1738] # 256 samples
    
    #     sampled_targets = [0.1310, 0.1378, 0.1273, 0.1306, 0.1309, 0.1302, 0.1269, 0.1276
    #    , 0.1308, 0.1380, 0.1310, 0.1726, 0.1331, 0.1363, 0.1303, 0.1726
    #    , 0.1310, 0.1301, 0.1309, 0.1269, 0.1644, 0.1648, 0.1306, 0.1686
    #    , 0.1363, 0.1725, 0.1302, 0.1303, 0.1274, 0.1363, 0.1333, 0.1310
    #    , 0.8463, 0.8544, 0.8576, 0.7991, 0.8576, 0.8501, 0.8516, 0.8557
    #    , 0.8003, 0.8557, 0.8539, 0.8505, 0.8479, 0.8485, 0.7862, 0.8539
    #    , 0.8539, 0.8464, 0.8516, 0.7869, 0.8465, 0.8520, 0.8010, 0.8484
    #    , 0.8508, 0.8038, 0.8497, 0.8538, 0.8479, 0.8518, 0.8576, 0.8518] # 64 samples from 2 ranges