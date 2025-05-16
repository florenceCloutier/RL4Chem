import os
import sys
import wandb
import hydra
import torch
import random
import re
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from omegaconf import DictConfig
from optimizer import BaseOptimizer
path_here = os.path.dirname(os.path.realpath(__file__))

from models.reinforce import TransPolicy
from models.reinforce_mol_construct import TransConstructPolicy
from data import smiles_vocabulary, selfies_vocabulary
from diversity import compute_internal_diversity

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_params(model):
    return (p for p in model.parameters() if p.requires_grad)

def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()

class reinforce_optimizer(BaseOptimizer):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.agent_name = cfg.agent_name

    def _init(self, cfg):
        if cfg.dataset == 'molgen_oled_1':
            saved_path = 'saved/' + cfg.dataset + '/' + cfg.model_name + '_' + cfg.rep + '/' + 'chembl0.25_zinc0.25_moses0.25_oled0.25.pt'
            vocab_path = 'data/molgen_oled_1/molgen_' + cfg.rep + '_vocab.txt'
            if cfg.rep=='smiles': 
                max_dataset_len = 366 # 112
            elif cfg.rep=='selfies':
                max_dataset_len = 106
            if cfg.max_len > max_dataset_len:
                cfg.max_len = max_dataset_len
                print('*** Changing the maximum length of sampled molecules because it was set to be greater than the maximum length seen during training ***')
        elif cfg.dataset == 'molgen_oled_2':
            saved_path = 'saved/' + cfg.dataset + '/' + cfg.model_name + '_' + cfg.rep + '/' + 'chembl0.2_zinc0.2_moses0.2_oled0.4.pt'
            vocab_path = 'data/molgen_oled_2/molgen_' + cfg.rep + '_vocab.txt'
            if cfg.rep=='smiles': 
                max_dataset_len = 366 # 112
            elif cfg.rep=='selfies':
                max_dataset_len = 106
            if cfg.max_len > max_dataset_len:
                cfg.max_len = max_dataset_len
                print('*** Changing the maximum length of sampled molecules because it was set to be greater than the maximum length seen during training ***')
        elif cfg.dataset == 'molgen':
            saved_path = 'saved/' + cfg.dataset + '/' + cfg.model_name + '_' + cfg.rep + '/' + 'chembl0.1_zinc0.3_moses0.6.pt'
            vocab_path = 'data/molgen/molgen_' + cfg.rep + '_vocab.txt'
            if cfg.rep=='smiles': 
                max_dataset_len = 112
            elif cfg.rep=='selfies':
                max_dataset_len = 106
            if cfg.max_len > max_dataset_len:
                cfg.max_len = max_dataset_len
                print('*** Changing the maximum length of sampled molecules because it was set to be greater than the maximum length seen during training ***')
        elif cfg.dataset == 'molgen_oled_cond_gen':
            saved_path = 'saved/' + cfg.dataset + '/' + cfg.model_name + '_' + cfg.rep + '/' + 'final_1.pt'
            vocab_path = 'data/molgen_oled_1/molgen_' + cfg.rep + '_vocab.txt'
            if cfg.rep=='smiles': 
                max_dataset_len = 381 # 406 # 112
            elif cfg.rep=='selfies':
                max_dataset_len = 106
            if cfg.max_len > max_dataset_len:
                cfg.max_len = max_dataset_len
                print('*** Changing the maximum length of sampled molecules because it was set to be greater than the maximum length seen during training ***')
          
        elif cfg.dataset == 'chembl':
            saved_path = 'saved/' + cfg.dataset + '/' + cfg.model_name + '_' + cfg.rep + '/' + cfg.saved_name
            vocab_path = 'data/chembl/chembl_' + cfg.rep + '_vocab.txt'
            if cfg.rep=='smiles': 
                max_dataset_len = 112
            elif cfg.rep=='selfies':
                max_dataset_len = 106
            if cfg.max_len > max_dataset_len:
                cfg.max_len = max_dataset_len
                print('*** Changing the maximum length of sampled molecules because it was set to be greater than the maximum length seen during training ***')
        
        elif cfg.dataset == 'zinc250k':
            saved_path = 'saved/' + cfg.dataset + '/' + cfg.model_name + '_' + cfg.rep + '/' + cfg.saved_name
            vocab_path = 'data/zinc250k/zinc_' + cfg.rep + '_vocab.txt'
            max_dataset_len = 73
            if cfg.max_len > max_dataset_len:
                cfg.max_len = max_dataset_len
                print('*** Changing the maximum length of sampled molecules because it was set to be greater than the maximum length seen during training ***')
        
        elif cfg.dataset == 'zinc1m':
            saved_path = 'saved/' + cfg.dataset + '/' + cfg.model_name + '_' + cfg.rep + '/' + cfg.saved_name
            vocab_path = 'data/zinc1m/zinc_' + cfg.rep + '_vocab_1M.txt'
            max_dataset_len = 74
            if cfg.max_len > max_dataset_len:
                cfg.max_len = max_dataset_len
                print('*** Changing the maximum length of sampled molecules because it was set to be greater than the maximum length seen during training ***')
        
        elif cfg.dataset == 'zinc10m':
            saved_path = 'saved/' + cfg.dataset + '/' + cfg.model_name + '_' + cfg.rep + '/' + cfg.saved_name
            vocab_path = 'data/zinc10m/zinc_' + cfg.rep + '_vocab.txt'
            if cfg.rep=='smiles': 
                max_dataset_len = 85
            elif cfg.rep=='selfies':
                max_dataset_len = 88
            if cfg.max_len > max_dataset_len:
                cfg.max_len = max_dataset_len
                print('*** Changing the maximum length of sampled molecules because it was set to be greater than the maximum length seen during training ***')
        
        elif cfg.dataset == 'zinc100m':
            saved_path = 'saved/' + cfg.dataset + '/' + cfg.model_name + '_' + cfg.rep + '/' + cfg.saved_name
            vocab_path = 'data/zinc100m/zinc_' + cfg.rep + '_vocab.txt'
            if cfg.rep=='smiles': 
                max_dataset_len = 85
            elif cfg.rep=='selfies':
                max_dataset_len = 88
            if cfg.max_len > max_dataset_len:
                cfg.max_len = max_dataset_len
                print('*** Changing the maximum length of sampled molecules because it was set to be greater than the maximum length seen during training ***')
        else:
            raise NotImplementedError
        
        #get data
        if cfg.rep == 'smiles':
            self.vocab = smiles_vocabulary(vocab_path=os.path.join(path_here, vocab_path))
        elif cfg.rep == 'selfies':
            self.vocab = selfies_vocabulary(vocab_path=os.path.join(path_here, vocab_path))
        else:
            raise NotImplementedError
    
        print('Vocab assigned')

        if cfg.gen_type == "generate":
            self.target_entropy = - 0.98 * torch.log(1 / torch.tensor(len(self.vocab)))
        elif cfg.gen_type == "construct":
            self.target_entropy = {
                "op": -0.98 * torch.log(1 / torch.tensor(2.0)),                         # 3 ops: ADD, REMOVE, Nope: REPLACE
                "token": -0.98 * torch.log(1 / torch.tensor(len(self.vocab))),          # vocab size
                "pos": -0.98 * torch.log(1 / torch.tensor(cfg.max_len, dtype=torch.float32)),  # max positions
            }
            
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = optim.Adam([self.log_alpha], lr=3e-4, eps=1e-4)

        assert cfg.model_name == 'char_trans'
        #get prior
        prior_saved_dict = torch.load(os.path.join(path_here, saved_path))
        print('Prior loaded')

        # get agent
        if cfg.gen_type == "generate":
            self.agent = TransPolicy(self.vocab, max_dataset_len, cfg.n_heads, cfg.n_embed, cfg.n_layers, dropout=cfg.dropout)
            self.prior = TransPolicy(self.vocab, max_dataset_len, cfg.n_heads, cfg.n_embed, cfg.n_layers, dropout=cfg.dropout)
        if cfg.gen_type == "construct":
            self.agent = TransConstructPolicy(self.vocab, max_dataset_len, cfg.n_heads, cfg.n_embed, cfg.n_layers, dropout=cfg.dropout)
            self.prior = TransConstructPolicy(self.vocab, max_dataset_len, cfg.n_heads, cfg.n_embed, cfg.n_layers, dropout=cfg.dropout)
        
        print('Agent class initialised')

        self.agent.to(self.device)
        self.prior.to(self.device)

        print('Agent class transferred to cuda memory')

        self.agent.load_save_dict(prior_saved_dict)
        self.prior.load_save_dict(prior_saved_dict)

        print('Prior weights initialised')

        # get optimizers
        self.optimizer = torch.optim.Adam(get_params(self.agent), lr=cfg['learning_rate'])
        # lr scheduler
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.99)

        print('Initialisation of optimizer is done!')
    
    def update(self, obs, rewards, nonterms, episode_lens, cfg, metrics, log, actions=None):
        rev_returns = torch.cumsum(rewards, dim=0) 
        advantages = rewards - rev_returns + rev_returns[-1:]

        logprobs, log_of_probs, action_probs = self.agent.get_likelihood(obs, nonterms, actions)

        # print(logprobs)
        # print(act_probs)
        # print(logprobs.shape)
        # print(act_probs.shape)
        # exit()
        
        # Get likelihoods for old policy (before the update)
        # with torch.no_grad():
        #     old_logprobs, old_log_of_probs, old_action_probs = self.prior.get_likelihood(obs, nonterms, actions)

        loss_pg = -advantages * logprobs
        loss_pg = loss_pg.sum(0, keepdim=True).mean()
        
        # Compute KL divergence
        # kl_div = torch.sum(action_probs + (log_of_probs - old_log_of_probs), dim=-1)
        # kl_loss = kl_div.mean()
        # beta = 0.001
        
        #loss_p = - (1 / logprobs.sum(0, keepdim=True)).mean()
        loss = loss_pg #+ cfg.lp_coef * loss_p 
        loss = loss_pg + self.alpha * logprobs.sum(0, keepdim=True).mean() # + beta * kl_loss

        # Calculate gradients and make an update to the network weights
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.optimizer.step()
        
        # Anneal lr
        self.scheduler.step()

        if cfg.gen_type == "construct":
            alpha_loss = (action_probs['token'].detach() * (-self.log_alpha.exp() * (log_of_probs['token'] + self.target_entropy['token']).detach())).mean()
        elif cfg.gen_type == "generate":
            alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_of_probs + self.target_entropy).detach())).mean()
        
        self.a_optimizer.zero_grad()
        alpha_loss.backward()
        self.a_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        if log:
            metrics['pg_loss'] = loss_pg.item()       
            metrics['agent_likelihood'] = logprobs.sum(0).mean().item()
            metrics['grad_norm'] = grad_norm.item() 
            metrics['smiles_len'] = episode_lens.float().mean().item()
            # metrics['loss_p'] = loss_p.item()
            metrics['alpha'] = self.alpha
            metrics['alpha_loss'] = alpha_loss.detach().item()
            print('logging!')
            wandb.log(metrics)

    def optimize(self, cfg):
        # if cfg.wandb_log:
        #     self.define_wandb_metrics()

        #set device
        self.device = torch.device(cfg.device)

        self._init(cfg)

        train_steps = 0
        eval_strings = 0
        metrics = dict() 
        print('Start training ... ')
        # qed_target = cfg.target_values[0]
        # logp_target = cfg.target_values[1]
        while eval_strings < cfg.max_strings:
            # Get target objectives with a normalized range for every episode
            # homo_lumo_gap_targets = np.random.uniform(0, 15, size=cfg.batch_size)

            with torch.no_grad():
                # sample experience
                obs, rewards, nonterms, episode_lens, actions = self.agent.get_data(cfg.batch_size, cfg.max_len, self.device, train_steps)

            smiles_list = []
            qed_targets = []
            logp_targets = []
            homo_lumo_targets = []
            for en_sms in obs.cpu().numpy().T:
                sms = self.vocab.decode_padded(en_sms)
                
                # TODO put in function: extract target values
                pattern = 'lOGP(-?[0-9.]+)\\\\ClOGP|QED([0-9.]+)\\\\QED'
                matches = re.findall(pattern, sms[0:30])
                logp_targets.append(float(next((m[0] for m in matches if m[0]), None))) # Value from CLOGP
                qed_targets.append(float(next((m[1] for m in matches if m[1]), None))) # Value from QED
                sms = sms.split('BOS', 1)[1]
                smiles_list.append(sms)
            
            score, score_qed, score_logp, score_homo_lumo = self.predict(smiles_list, [list(pair) for pair in zip(qed_targets, logp_targets)], homo_lumo=cfg.homo_lumo)
            score_array = np.array(score)
            scores = torch.tensor(score_array, dtype=torch.float32, device=self.device).unsqueeze(0)

            if self.finish:
                print('max oracle hit')
                break
                wandb.finish()
                sys.exit(0)

            train_steps += 1
            eval_strings += cfg.batch_size

            log = False
            if cfg.wandb_log and train_steps % cfg.train_log_interval == 0:
                log = True
                metrics = dict()
                metrics['eval_strings'] = eval_strings
                score_mask = np.where(score_array != 0.0)[0]
                metrics['mean_score'] = np.mean(score_array[score_mask])
                metrics['max_score'] = np.max(score)
                metrics['min_score'] = np.min(score)
                # metrics['mean_qed_score'] = np.mean(score_qed)
                # metrics['mean_logp_score'] = np.mean(score_logp)
                metrics['mean_episode_lens'] = np.mean(episode_lens.tolist())
                metrics['max_episode_lens'] = np.max(episode_lens.tolist())
                metrics['min_episode_lens'] = np.min(episode_lens.tolist())
                metrics['invalid_count'] = self.invalid_count
                metrics['diversity'] = compute_internal_diversity(smiles_list)
                
                # Evaluation loop
                # if train_steps % cfg.eval_interval == 0:
                diff_qed = []
                diff_logp = []
                # with torch.no_grad():
                    # sample experience
                    # obs, rewards, nonterms, episode_lens = self.agent.get_data(cfg.batch_size, cfg.max_len, self.device)

                smiles_list = []
                qed_eval_targets = []
                logp_eval_targets = []
                for en_sms in obs.cpu().numpy().T:
                    sms = self.vocab.decode_padded(en_sms)
                    # TODO put in function: extract target values
                    pattern = 'ClOGP(-?[0-9.]+)\\\\ClOGP|QED([0-9.]+)\\\\QED'
                    matches = re.findall(pattern, sms[0:30])
                    # TODO!!! Normalize logp values
                    logp_eval_targets.append(float(next((m[0] for m in matches if m[0]), None))) # Value from CLOGP
                    qed_eval_targets.append(float(next((m[1] for m in matches if m[1]), None))) # Value from QED
                    sms = sms.split('BOS', 1)[1]
                    smiles_list.append(sms)

                # _, score_qed, score_logp, score_homo_lumo = self.predict(smiles_list, [list(pair) for pair in zip(qed_eval_targets, logp_eval_targets)], homo_lumo=cfg.homo_lumo)
                mask = np.where(np.array(score_qed) != 0.0)[0]
                diff_qed = np.array(qed_targets)[mask] - np.array(score_qed)[mask]
                diff_logp = np.array(logp_targets) - np.array(list(map(self._denormalize_logp, score_logp)))
                    
                metrics['val_qed_MAE'] = np.mean(np.abs(diff_qed))
                metrics['val_logp_MAE'] = np.mean(np.abs(diff_logp))
                if len(diff_qed) > 0:
                    metrics['min_qed_AE'] = np.min(np.abs(diff_qed))
                else:
                    metrics['min_qed_AE'] = None
                # metrics['min_qed_AE'] = np.min(np.abs(diff_qed))
                if len(diff_logp) > 0:
                    metrics['min_logp_AE'] = np.min(np.abs(diff_logp))
                else:
                    metrics['min_logp_AE'] = None
                metrics['batch_n'] = train_steps // 100
                
                # metrics['homo_lumo_gap'] = np.mean(score_homo_lumo)
                    
                wandb.log(metrics)
            
            rewards = rewards * scores
            self.update(obs, rewards, nonterms, episode_lens, cfg, metrics, log, actions=actions)

        obs, rewards, nonterms, episode_lens, actions = self.agent.get_data(2000, cfg.max_len, self.device, train_steps=0, last_eval=True)
        smiles_list = []
        qed_targets = []
        logp_targets = []
        homo_lumo_targets = []
        for en_sms in obs.cpu().numpy().T:
            sms = self.vocab.decode_padded(en_sms)
            
            # TODO put in function: extract target values
            pattern = 'lOGP(-?[0-9.]+)\\\\ClOGP|QED([0-9.]+)\\\\QED'
            matches = re.findall(pattern, sms[0:30])
            logp_targets.append(float(next((m[0] for m in matches if m[0]), None))) # Value from CLOGP
            qed_targets.append(float(next((m[1] for m in matches if m[1]), None))) # Value from QED
            sms = sms.split('BOS', 1)[1]
            smiles_list.append(sms)
        
        score, score_qed, score_logp, score_homo_lumo = self.predict(smiles_list, [list(pair) for pair in zip(qed_targets, logp_targets)], homo_lumo=cfg.homo_lumo, last_eval=True)
        score_array = np.array(score)
        scores = torch.tensor(score_array, dtype=torch.float32, device=self.device).unsqueeze(0)
        print("Scores: ", scores)
        print("Score qed: ", score_qed)
        print("Score logp: ", score_logp)
        print('max training string hit')
        
        mask_qed = np.where(np.array(score_qed) != 0.0)[0]
        mask_logp = np.where(np.array(score_logp) != 0.0)[0]
        diff_qed = np.array(qed_targets)[mask_qed] - np.array(score_qed)[mask_qed]
        diff_logp = np.array(logp_targets)[mask_logp] - np.array(list(map(self._denormalize_logp, score_logp)))[mask_logp]
            
        print('Val_qed_MAE', np.mean(np.abs(diff_qed)))
        print('val_logp_MAE', np.mean(np.abs(diff_logp)))
        print('min_qed_AE', np.min(np.abs(diff_qed)))
        print('min_logp_AE', np.min(np.abs(diff_logp)))
        wandb.finish()
        sys.exit(0)

@hydra.main(config_path='cfgs', config_name='reinforce_trans', version_base=None)
def main(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    
    if cfg.wandb_log:
        # project_name = cfg.task + '_' + cfg.target
        if cfg.wandb_dir is not None:
            cfg.wandb_dir = path_here 
        else:
            cfg.wandb_dir = hydra_cfg['runtime']['output_dir']
        wandb.init(project="llms-materials-rl")
        # wandb.init(project="llms-materials-rl", entity=cfg.wandb_entity, config=dict(cfg), dir=cfg.wandb_dir)
        wandb.run.name = cfg.wandb_run_name
        
    set_seed(cfg.seed)
    cfg.output_dir = hydra_cfg['runtime']['output_dir']

    optimizer = reinforce_optimizer(cfg)
    optimizer.optimize(cfg)
    sys.exit(0)
    
if __name__ == '__main__':
    main()
    sys.exit(0)
    exit()