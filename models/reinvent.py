import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import torch.nn.utils.rnn as rnn_utils

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
        end_flags = torch.zeros((1, batch_size), dtype=torch.bool, device=device)

        input_lens = torch.ones(batch_size)
        hiddens = None
        for i in range(1, max_length+1):
            preds_dist, _, hiddens = self.forward(obs[i-1].view(1, -1), input_lens, hiddens)
            preds = preds_dist.sample()

            obs[i] = preds
            nonterms[i-1] = ~end_flags
            
            EOS_sampled = (preds == self.vocab.eos)

            #check if all sequences are done
            end_flags = torch.ge(end_flags + EOS_sampled, 1)
            if torch.prod(end_flags) == 1: break

        obs = obs[:i+1]
        nonterms = nonterms[:i+1]
        episode_lens = nonterms.sum(0).cpu()

        return obs, nonterms, episode_lens

    def get_data_with_prior(self, prior, batch_size, max_length, device):
        obs = torch.zeros((max_length + 1, batch_size), dtype=torch.long, device=device)
        obs[0] = self.vocab.bos
        log_probs = torch.zeros((max_length, batch_size), dtype=torch.float32, device=device)
        nonterms = torch.zeros((max_length + 1, batch_size), dtype=torch.bool, device=device)
        rewards = torch.zeros((max_length, batch_size), dtype=torch.float32, device=device)
        end_flags = torch.zeros((1, batch_size), dtype=torch.bool, device=device)

        input_lens = torch.ones(batch_size)
        hiddens = None
        for i in range(1, max_length+1):
            preds_dist, _, hiddens = self.forward(obs[i-1].view(1, -1), input_lens, hiddens)
            preds = preds_dist.sample()

            obs[i] = preds
            log_probs[i-1] = preds_dist.log_prob(preds) * (~end_flags)
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
        log_probs = log_probs[:i]
        rewards = rewards[:i]
        episode_lens = nonterms.sum(0).cpu()

        with torch.no_grad():
            prior_dist, _, _ = prior(obs[:-1], episode_lens)
            prior_log_probs = (prior_dist.log_prob(obs[1:]) * nonterms[:-1])

        return obs, rewards, nonterms, log_probs, prior_log_probs, episode_lens

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