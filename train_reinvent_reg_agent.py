import os
import sys
import wandb
import hydra
import torch
import random
import numpy as np
import selfies as sf
from omegaconf import DictConfig
from optimizer import BaseOptimizer
path_here = os.path.dirname(os.path.realpath(__file__))

from models.reinvent import RnnPolicy
from data import smiles_vocabulary, selfies_vocabulary

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
    

class reinvent_optimizer(BaseOptimizer):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.agent_name = cfg.agent_name

    def _init(self, cfg):
        if cfg.dataset == 'zinc250k':
            saved_path = 'saved/' + cfg.dataset + '/' + cfg.model_name + '_' + cfg.rep + '/' + cfg.saved_name
            vocab_path = 'data/zinc250k/zinc_' + cfg.rep + '_vocab.txt'
            max_dataset_len = 73
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
       
        assert cfg.model_name == 'char_rnn'
        #get pretrained weights
        prior_saved_dict = torch.load(os.path.join(path_here, saved_path))

        # get agent
        self.agent = RnnPolicy(self.vocab, cfg.embedding_size, cfg.hidden_size, cfg.num_layers).to(self.device)
        self.agent.load_save_dict(prior_saved_dict)

        # get optimizers
        self.optimizer = torch.optim.Adam(get_params(self.agent), lr=cfg['learning_rate'])

    def update(self, obs, scores, nonterms, episode_lens, cfg, metrics, log): 
        
        logprobs = self.agent.get_likelihood(obs, episode_lens, nonterms)

        loss_pg = -scores * logprobs
        loss_pg = loss_pg.sum(0, keepdim=True).mean()

        if cfg.reg == 'inv_lopp':
            # loss_p regularisation
            loss_p = - (1 / logprobs.sum(0, keepdim=True)).mean()
            loss = loss_pg + cfg.inv_p_coef * loss_p 
        elif cfg.reg == 'ent':
            # entropy regularisation
            loss_p = loss_pg.sum(0, keepdim=True).mean()
            loss = loss_pg + cfg.ent_coef * loss_p 
        else:
            loss_p = 0
            loss = loss_pg

        # Calculate gradients and make an update to the network weights
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.optimizer.step()

        if log:
            metrics['pg_loss'] = loss_pg.item()       
            metrics['agent_likelihood'] = logprobs.sum(0).mean().item()
            metrics['grad_norm'] = grad_norm.item() 
            metrics['smiles_len'] = episode_lens.float().mean().item()
            metrics['loss_p'] = 0 if cfg.reg == None else loss_p.item()
            print('logging!')
            wandb.log(metrics)

    def optimize(self, cfg):
        if cfg.wandb_log:
            self.define_wandb_metrics()

        #set device
        self.device = torch.device(cfg.device)

        self._init(cfg)

        train_steps = 0
        eval_strings = 0
        metrics = dict() 
        while eval_strings < cfg.max_strings:

            with torch.no_grad():
                # sample experience
                obs, nonterms, episode_lens = self.agent.get_data(cfg.batch_size, cfg.max_len, self.device)
               
            if cfg.rep == 'selfies':            
                smiles_list = []
                for en_sms in obs.cpu().numpy().T:
                    sms = self.vocab.decode_padded(en_sms)
                    smiles_list.append(sms)

                score = np.array(self.predict(smiles_list))
                scores = torch.tensor(score, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                smiles_list = []
                for en_sms in obs.cpu().numpy().T:
                    sms = self.vocab.decode_padded(en_sms)
                    smiles_list.append(sms)

                score = np.array(self.predict(smiles_list))
                scores = torch.tensor(score, dtype=torch.float32, device=self.device).unsqueeze(0)

            if self.finish:
                print('max oracle hit')
                wandb.finish()
                sys.exit(0)

            train_steps += 1
            eval_strings += cfg.batch_size

            log = False
            if cfg.wandb_log and train_steps % cfg.train_log_interval == 0:
                log = True
                metrics = dict()
                metrics['eval_strings'] = eval_strings
                metrics['mean_score'] = np.mean(score)
                metrics['max_score'] = np.max(score)
                metrics['min_score'] = np.min(score)
                metrics['mean_episode_lens'] = np.mean(episode_lens.tolist())
                metrics['max_episode_lens'] = np.max(episode_lens.tolist())
                metrics['min_episode_lens'] = np.min(episode_lens.tolist())
                wandb.log(metrics)

            self.update(obs, scores, nonterms, episode_lens, cfg, metrics, log)
        
        print('max training string hit')
        wandb.finish()
        sys.exit(0)

@hydra.main(config_path='cfgs', config_name='reinvent_rnn', version_base=None)
def main(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    
    if cfg.wandb_log:
        project_name = cfg.task + '_' + cfg.target
        if cfg.wandb_dir is not None:
            cfg.wandb_dir = path_here 
        else:
            cfg.wandb_dir = hydra_cfg['runtime']['output_dir']
        wandb.init(project=project_name, entity=cfg.wandb_entity, config=dict(cfg), dir=cfg.wandb_dir)
        wandb.run.name = cfg.wandb_run_name
    
    set_seed(cfg.seed)
    cfg.output_dir = hydra_cfg['runtime']['output_dir']

    optimizer = reinvent_optimizer(cfg)
    optimizer.optimize(cfg)
    sys.exit(0)

if __name__ == '__main__':
    main()
    exit()