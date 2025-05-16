import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import pandas as pd
import json
import random
from torch_geometric.utils import from_smiles

from gnn_intermediate_reward import GNNModel

class TransConstructPolicy(nn.Module):
    def __init__(self, vocab, max_len, n_heads, n_embed, n_layers, dropout):
        super(TransConstructPolicy, self).__init__()

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

        # self.linear = nn.Linear(n_embed, len(vocab)) 
        ### Change action type
        self.op_head = nn.Linear(n_embed, 3)  # for op_type
        self.token_head = nn.Linear(n_embed, len(vocab))  # for token
        self.pos_head = nn.Linear(n_embed, max_len)  # for position
        ###
        
        print('TRANS: linear layer is initialised')

        self.register_buffer('triu', torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1))
        
        self.model = GNNModel(num_node_features=9, dropout=0.2)
        self.model.load_state_dict(torch.load("saved/intermediate_rewards_gnn_qed/best_1.pt"))

        print('TRANS: buffer is registered')
        print('TRANS: done')

    def forward(self, x, start_idx=0):
        L, B = x.shape
        x_tok = self.embedding(x)  #L, B
        x_pos = self.position_embedding(torch.arange(L, device=x.device)).view(L, 1, -1).expand(-1, B, -1)
        x = x_tok + x_pos
        x = self.encoder(x, self.triu[:L, :L])
        # logits = self.linear(x)
        
        ### Change action type
        x_pooled = x.mean(dim=0)
        
        pos_logits = self.pos_head(x_pooled)
        
        if start_idx > 0:
            mask = torch.arange(self.max_len, device=x.device) < start_idx
            pos_logits[:, mask] = float('-inf')
        
        return {
            "op_dist": td.Categorical(logits=self.op_head(x_pooled)),
            "token_dist": td.Categorical(logits=self.token_head(x_pooled)),
            "pos_dist": td.Categorical(logits=pos_logits),
        }
        ###
        
        return td.Categorical(logits=logits)

    def autoregress(self, x, start_idx=0):
        # L, B = x.shape
        # x_tok = self.embedding(x)  #L, B
        # x_pos = self.position_embedding(torch.arange(L, device=x.device)).view(L, 1, -1).expand(-1, B, -1)
        # x = x_tok + x_pos
        # x = self.encoder(x, self.triu[:L, :L])
        
        ### Change action type
        return self.forward(x, start_idx=start_idx)
        x_pooled = x.mean(dim=0)  # or x[-1] for the last token

        op_logits = self.op_head(x_pooled)
        token_logits = self.token_head(x_pooled)
        pos_logits = self.pos_head(x_pooled)

        return {
            "op_dist": td.Categorical(logits=op_logits),
            "token_dist": td.Categorical(logits=token_logits),
            "pos_dist": td.Categorical(logits=pos_logits),
        }
        ###
        logits = self.linear(x)[-1]
        return td.Categorical(logits=logits)

    def sample(self, batch_size, max_length, device):
        assert max_length <= self.max_len
        preds = self.vocab.bos * torch.ones((1, batch_size), dtype=torch.long, device=device)
        finished = torch.zeros((batch_size), dtype=torch.bool, device=device)
        imag_smiles_lens = torch.ones((batch_size),  device=device)
        
        ### Change action type
        sequences = [torch.tensor([self.vocab.bos], dtype=torch.long, device=device) for _ in range(batch_size)]
        finished = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        seq_lens = torch.ones((batch_size,), dtype=torch.long, device=device)

        max_steps = 100
        
        with torch.no_grad():
            for step in range(max_steps):
                for i in range(batch_size):
                    if finished[i]:
                        continue
                    
                    seq = preds[i]
                    seq_input = seq.unsqueeze(1)
                    
                    preds_dist = self.forward(seq_input)
                    op = preds_dist["op_dist"].sample().item()
                    token = preds_dist["token_dist"].sample().item()
                    pos = preds_dist["pos_dist"].sample().item()
                    new_seq = self.apply_action(seq, op, token, pos)
                    preds[i] = new_seq
                    seq_lens[i] += 1
                    imag_smiles_lens += ~finished

                    if (new_seq[-1] == self.vocab.eos).item():
                        finished[i] = True
                    if torch.prod(finished) == 1: break
                    # EOS_sampled = (preds[-1] == self.vocab.eos)
                    # finished = torch.ge(finished + EOS_sampled, 1)
                    # if torch.prod(finished) == 1: break
        # Pad and stack sequences to return a uniform tensor (optional)
        max_len = max([len(seq) for seq in sequences])
        padded_seqs = torch.full((batch_size, max_len), self.vocab.pad, dtype=torch.long, device=device)
        for i, seq in enumerate(sequences):
            padded_seqs[i, :len(seq)] = seq

        # Convert to list of lists
        return padded_seqs.tolist(), seq_lens.tolist()
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
        pad_token = self.vocab.pad
        if op == 0 and pos < len(seq): # ADD
            new_seq = torch.cat([new_seq[:pos], torch.tensor([token], device=seq.device), new_seq[pos:-1]])
        elif op == 1 and pos < len(seq): # REMOVE
            new_seq = torch.cat([new_seq[:pos], new_seq[pos+1:], torch.tensor([pad_token], device=seq.device)])[:len(seq)]
        elif op == 2 and pos < len(seq): # REPLACE
            new_seq[pos] = token
        return new_seq

    def get_likelihood(self, obs, nonterms, actions=None):      
        ### Change action type
        dist = self.forward(obs[:-1])  # Get action distributions for each step

        # 1. Log-probs for sampled actions
        logprob_op = dist["op_dist"].log_prob(actions["op"])     # (T, B)
        logprob_token = dist["token_dist"].log_prob(actions["token"])  # (T, B)
        logprob_pos = dist["pos_dist"].log_prob(actions["pos"])   # (T, B)

        total_logprob = (logprob_op + logprob_token + logprob_pos) * nonterms
        
        # 2. Log of probs
        log_of_probs = {
            "op": F.log_softmax(dist["op_dist"].logits, dim=-1) * nonterms.unsqueeze(-1),
            "token": F.log_softmax(dist["token_dist"].logits, dim=-1) * nonterms.unsqueeze(-1),
            "pos": F.log_softmax(dist["pos_dist"].logits, dim=-1) * nonterms.unsqueeze(-1),
        }

        # 3. Action probabilities
        action_probs = {
            "op": dist["op_dist"].probs * nonterms.unsqueeze(-1),
            "token": dist["token_dist"].probs * nonterms.unsqueeze(-1),
            "pos": dist["pos_dist"].probs * nonterms.unsqueeze(-1),
        }

        return total_logprob, log_of_probs, action_probs

        return total_logprob
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
            
        obs = torch.zeros((max_length, batch_size), dtype=torch.long, device=device)
        obs[0] = self.vocab.bos
        ### Change action type
        actions = {
            "op": torch.zeros((max_length, batch_size), dtype=torch.long, device=device),
            "token": torch.zeros((max_length, batch_size), dtype=torch.long, device=device),
            "pos": torch.zeros((max_length, batch_size), dtype=torch.long, device=device)
        }
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
        
        nonterms = torch.zeros((max_length, batch_size), dtype=torch.bool, device=device)
        rewards = torch.zeros((max_length, batch_size), dtype=torch.float32, device=device)
        end_flags = torch.zeros((1, batch_size), dtype=torch.bool, device=device)

        interval_intermediate_reward = 20 
        for i in range(max_steps):
            # TODO @Ali here is where I would want to get an intermediate reward!
            if i != 0 and i % interval_intermediate_reward == 0:
                pass
                # data_as_graph = from_smiles()
                # pred = self.model()
                
            # preds_dist = self.autoregress(obs[:i])
            
            ### Change action type
            batch_size = obs.shape[1]
            new_obs = obs.clone()
            
            action_dists = self.autoregress(obs[:], start_idx=start_idx)
            ops = action_dists["op_dist"].sample()
            tokens = action_dists["token_dist"].sample()
            positions = action_dists["pos_dist"].sample()
            
            actions["op"][i] = ops
            actions["token"][i] = tokens
            actions["pos"][i] = positions
            
            new_obs = obs.clone()
            
            for b in range(batch_size):
                if end_flags[0, b]:
                    continue
                
                seq = obs[:, b]
                op = ops[b].item()
                token = tokens[b].item()
                pos = positions[b].item()
                
                if pos < start_idx:
                    continue
                new_seq = self.apply_action(seq, op, token, pos)
                # new_len = new_seq.shape[0]
                new_obs[:, b] = new_seq
                # if new_len < obs.shape[0]:
                #     new_obs[new_len:, b] = self.vocab.pad
                
                nonterms[i-1, b] = not end_flags[0, b]
                if (new_seq[-1] == self.vocab.eos):
                    rewards[i-1, b] = 1
                    end_flags[0, b] = True
                    
            obs = new_obs
            if end_flags.all():
                break
            
        if i == max_length:
            rewards[-1] += (~end_flags).squeeze(0)
            
        # assert rewards.sum() == batch_size
            
        obs = obs[:i+1]
        nonterms = nonterms[:i+1]
        rewards = rewards[:i]
        episode_lens = nonterms.sum(0).cpu()
        actions = {k: v[:i-start_idx+1] for k, v in actions.items()}
        nonterms = nonterms[start_idx:i+1]
        rewards = rewards[start_idx - 1:i]
        
        return obs, rewards, nonterms, episode_lens, actions
        
    def add_targets(self, batch_size, obs, batch_n):
        
        with open("data/joint_samples_init_mol/batches_512.json", "r") as f:
            batches = json.load(f)

        # Get the correct batch (fall back to last if batch_n too large)
        batch = batches[min(batch_n, len(batches) - 1)]
        qed_targets = batch["qed"]
        logp_targets = batch["logp"]
        molecule_seqs = batch["data"]

        # Zip and sample
        combined = list(zip(qed_targets, logp_targets, molecule_seqs))
        sampled = random.choices(combined, k=batch_size)

        for column_idx, (qed, logp, molecules) in enumerate(sampled):
            # Normalize and tokenize targets
            qed = self.adjust_value(qed)
            logp = self.adjust_value(logp)
            qed_tokens = self.tokenize_target_QED(qed)
            logp_tokens = self.tokenize_target_LOGP(logp)
            mol_tokens = [self.tokenize_molecule(mol) for mol in molecules]

            # Encode all parts
            logp_tensor = torch.tensor(self.vocab.encode(logp_tokens))
            qed_tensor = torch.tensor(self.vocab.encode(qed_tokens))
            
            for mol in molecules:
                if column_idx >= obs.shape[1]:
                    break  # Stop if we've filled the batch

                # Tokenize and encode molecule
                mol_tokens = self.tokenize_molecule(mol)
                mol_tensor = torch.tensor(self.vocab.encode(mol_tokens), dtype=torch.long)

                # Compose full sequence
                total_len = len(logp_tensor) + len(qed_tensor) + 1 + len(mol_tensor)  # +1 for BOS

                if total_len > obs.shape[0]:
                    print(f"Skipping molecule {mol} due to length overflow.")
                    continue  # Avoid overflowing obs height

                obs[0:len(logp_tensor), column_idx] = logp_tensor
                obs[len(logp_tensor):len(logp_tensor)+len(qed_tensor), column_idx] = qed_tensor

                bos_idx = len(logp_tensor) + len(qed_tensor)
                obs[bos_idx, column_idx] = self.vocab.bos

                obs[bos_idx + 1:bos_idx + 1 + len(mol_tensor), column_idx] = mol_tensor

                column_idx += 1  # Move to the next column (batch example)

            # mol_tensor = torch.tensor(self.vocab.encode(mol_tokens))

            # # Assemble: [logp][qed][BOS][molecule]
            # obs[0:len(logp_tensor), column_idx] = logp_tensor
            # obs[len(logp_tensor):len(logp_tensor)+len(qed_tensor), column_idx] = qed_tensor
            # bos_idx = len(logp_tensor) + len(qed_tensor)
            # obs[bos_idx, column_idx] = self.vocab.bos
            # obs[bos_idx + 1:bos_idx + 1 + len(mol_tensor), column_idx] = mol_tensor

        return obs
    
    def adjust_value(self, val, length=6):
        """Adjust value to match the specified length in string representation."""
        # Round the value to match the desired number of decimals
        val = float(val)
        decimals = max(0, length - len(str(int(val))) - 2)  # Account for "0." or "-0."
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
    
    def tokenize_molecule(self, molecule):
        alphabet = sorted(self.vocab.alphabet_list, key=len, reverse=True) # Longest match first
        tokens = []
        new_molecule = molecule.replace("Cl", "L").replace("Br", "R")
        i = 0
        while i < len(new_molecule):
            matched = False
            for token in alphabet:
                if new_molecule.startswith(token, i):
                    tokens.append(token)
                    i += len(token)
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Unrecognized token in molecule at position {i}: {molecule[i]}")
        return tokens
    
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
            "op_head": self.op_head.state_dict(),
            "token_head": self.token_head.state_dict(),
            "pos_head": self.pos_head.state_dict(),
            ###
            # "linear": self.linear.state_dict(),
        }

    def load_save_dict(self, saved_dict):
        self.embedding.load_state_dict(saved_dict["embedding"])
        self.position_embedding.load_state_dict(saved_dict["position_embedding"])
        self.encoder.load_state_dict(saved_dict["encoder"])
        
        ### Change action type
        # Use the old linear weights for token prediction head
        old_linear = saved_dict["linear"]
        self.token_head.load_state_dict(old_linear)
        ###
        
        # self.linear.load_state_dict(saved_dict["linear"])
