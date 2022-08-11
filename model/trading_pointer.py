import sys, os
sys.path.insert(0, os.getcwd())

import random
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from utils.lstm import script_lnlstm
from utils.nn_module import fc_block
from model.vac import VAC


class PointerNetwork(nn.Module):

    def __init__(self, cfg):
        super(PointerNetwork, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model
        self.activation_type = self.cfg.activation
        self.entity_reduce_type = self.cfg.entity_reduce_type

        self.MAX_SELECTED_UNITS_NUM = self.cfg.get('max_selected_units_num', 64)
        self.MAX_ENTITY_NUM = self.cfg.get('max_entity_num', 512)

        self.key_fc = fc_block(self.cfg.entity_embedding_dim, self.cfg.key_dim, activation=None, norm_type=None)

        self.query_mlp = nn.Sequential(
            *[
                fc_block(self.cfg.input_dim, self.cfg.func_dim, activation=self.activation_type),
                fc_block(self.cfg.func_dim, self.cfg.key_dim, activation=None),
            ]
        )
        self.embed_mlp = nn.Sequential(
            *[
                fc_block(self.cfg.key_dim, self.cfg.func_dim, activation=self.activation_type, norm_type=None),
                fc_block(self.cfg.func_dim, self.cfg.input_dim, activation=None, norm_type=None)
            ]
        )

        self.lstm_num_layers = self.cfg.lstm_num_layers
        self.lstm_hidden_dim = self.cfg.lstm_hidden_dim
        self.lstm = script_lnlstm(self.cfg.key_dim, self.lstm_hidden_dim, self.lstm_num_layers)
        self.value_net = VAC(obs_shape=self.cfg.input_dim, action_shape=1)

    def _get_key_mask(self, entity_embedding, entity_mask):
        key = self.key_fc(entity_embedding)  # b, n, c
        if self.entity_reduce_type == 'entity_num':
            key_reduce = torch.div(key, entity_mask.sum(dim=-1).reshape(-1, 1, 1))
            key_embeddings = self.embed_mlp(key_reduce)
        elif self.entity_reduce_type == 'constant':
            key_reduce = torch.div(key, self.MAX_ENTITY_NUM)
            key_embeddings = self.embed_mlp(key_reduce)
        elif self.entity_reduce_type == 'selected_units_num':
            key_embeddings = key
        else:
            raise NotImplementedError
        return key, key_embeddings

    def _get_pred_with_logit(
        self,
        logit,
    ):
        dist = torch.distributions.Categorical(logits=logit)
        units = dist.sample()
        return units

    def _query(
        self,
        key: Tensor,
        autoregressive_embedding: Tensor,
        entity_mask: Tensor,
        key_embeddings: Tensor,
        temperature: float = 1
    ):

        ae = autoregressive_embedding
        bs = ae.shape[0]  # batch size

        results_list, logits_list = [], []
        result: Optional[Tensor] = None

        # initialize hidden state
        state = [
            (
                torch.zeros(bs, self.lstm_hidden_dim,
                            device=ae.device), torch.zeros(bs, self.lstm_hidden_dim, device=ae.device)
            ) for _ in range(self.lstm_num_layers)
        ]

        selected_units_one_hot = torch.zeros(*key_embeddings.shape[:2], device=ae.device)  # bs, n+1,1
        for selected_step in range(self.MAX_SELECTED_UNITS_NUM):
            if result is not None:
                entity_mask[torch.arange(bs), result.detach()] = False  # mask selected units
            lstm_input = self.query_mlp(ae).unsqueeze(0)  # 1, bs, lstm_hidden_size
            lstm_output, state = self.lstm(lstm_input, state)

            # dot product of queries and key -> logits
            queries = lstm_output.permute(1, 0, 2)  # 1, bs, lstm_hidden_size -> bs, 1,lstm_hidden_size
            # get logits
            step_logits = (queries * key).sum(dim=-1)  # b, n
            step_logits.div_(temperature)
            step_logits = step_logits.masked_fill(~entity_mask, -1e9)
            logits_list.append(step_logits)

            result = self._get_pred_with_logit(step_logits, )
            # if not end and choose end flag set selected units_num
            results_list.append(result)
            if self.entity_reduce_type == 'selected_units_num':
                # put selected_units in cut step to selected_units_on_hot
                selected_units_one_hot[torch.arange(bs), result, ] = 1

                # take average of selected_units_embedding according to selected_units_num
                selected_units_emebedding = (key_embeddings * selected_units_one_hot.unsqueeze(-1)).sum(dim=1)
                selected_units_emebedding = selected_units_emebedding / (selected_step + 1)

                selected_units_emebedding = self.embed_mlp(selected_units_emebedding)
                ae = autoregressive_embedding + selected_units_emebedding
            else:
                ae = ae + key_embeddings[torch.arange(bs), result]

        results = torch.stack(results_list, dim=1)
        logits = torch.stack(logits_list, dim=1)

        return logits, results, ae

    def _train_query(
        self,
        key: Tensor,
        autoregressive_embedding: Tensor,
        entity_mask: Tensor,
        key_embeddings: Tensor,
        selected_units: Tensor,
        temperature: float = 1
    ):
        ae = autoregressive_embedding
        bs = ae.shape[0]

        logits_list = []
        # entity_mask = entity_mask.repeat(max(seq_len, 1), 1, 1)  # b, n -> s, b, n
        selected_units_one_hot = torch.zeros(*key_embeddings.shape[:2], device=ae.device).unsqueeze(dim=2)

        # initialize hidden state
        state = [
            (
                torch.zeros(bs, self.lstm_hidden_dim,
                            device=ae.device), torch.zeros(bs, self.lstm_hidden_dim, device=ae.device)
            ) for _ in range(self.lstm_num_layers)
        ]

        for selected_step in range(self.MAX_SELECTED_UNITS_NUM):
            if selected_step > 0:
                entity_mask[torch.arange(bs), selected_units[:, selected_step - 1]] = 0  # mask selected units
            lstm_input = self.query_mlp(ae).unsqueeze(0)
            lstm_output, state = self.lstm(lstm_input, state)

            queries = lstm_output.permute(1, 0, 2)  # 1, bs, lstm_hidden_size -> bs, 1,lstm_hidden_size
            step_logits = (queries.squeeze(0) * key).sum(dim=-1)  # b, n
            step_logits.div_(temperature)
            step_logits = step_logits.masked_fill(~entity_mask, -1e9)

            logits_list.append(step_logits)

            if self.entity_reduce_type == 'selected_units_num' in self.entity_reduce_type:
                new_selected_units_one_hot = selected_units_one_hot.clone()  # inplace operation can not backward !!!
                new_selected_units_one_hot[torch.arange(bs), selected_units[:, selected_step], :] = 1
                # take average of selected_units_embedding according to selected_units_nu
                selected_units_emebedding = (key_embeddings * new_selected_units_one_hot).sum(dim=1)
                selected_units_emebedding = selected_units_emebedding / (selected_step + 1)
                selected_units_emebedding = self.embed_mlp(selected_units_emebedding)
                ae = autoregressive_embedding + selected_units_emebedding
                selected_units_one_hot = new_selected_units_one_hot.clone()
            else:
                ae = ae + key_embeddings[torch.arange(bs), selected_units[:, selected_step]]
        logits = torch.stack(logits_list, dim=1)
        return logits, None, ae

    def forward(
        self,
        embedding,
        entity_embedding,
        entity_mask,
        selected_units=None,
        temperature=1,
    ):
        key, key_embeddings = self._get_key_mask(entity_embedding, entity_mask)
        entity_mask = entity_mask.clone()
        if selected_units is not None:  # train
            logits, units, embedding = self._train_query(
                key, embedding, entity_mask, key_embeddings, selected_units, temperature=temperature
            )
        else:
            logits, units, embedding = self._query(key, embedding, entity_mask, key_embeddings, temperature=temperature)

        output = self.value_net.forward(embedding, mode='compute_actor_critic')
        value_logit, value = output['logit'], output['value']
        return logits, units, embedding, value_logit, value


import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    from easydict import EasyDict

    default_model_config = EasyDict(
        {
            'model': {
                'input_dim': 14,
                'max_selected_units_num': 400,
                'max_entity_num': 2000,
                'entity_embedding_dim': 256,
                'key_dim': 32,
                'func_dim': 256,
                'lstm_hidden_dim': 32,
                'lstm_num_layers': 1,
                'activation': 'relu',
                'entity_reduce_type': 'selected_units_num',  # ['constant', 'entity_num', 'selected_units_num']
            }
        }
    )
    net = PointerNetwork(default_model_config)
    entity_embedding_dim = net.cfg.entity_embedding_dim
    input_dim = net.cfg.input_dim
    MaxEntityNum = net.cfg.max_entity_num
    batch_size = 10
    embedding = torch.rand(size=(
        batch_size,
        input_dim,
    ))
    # entity_mask = torch.ones(size=(batch_size,MaxEntityNum),dtype=torch.bool)
    entity_mask = torch.rand(size=(
        batch_size,
        MaxEntityNum,
    )) > 0
    entity_embedings = torch.rand(size=(
        batch_size,
        MaxEntityNum,
        entity_embedding_dim,
    ))

    setup_seed(20)
    logits0, units, embedding0 = net.forward(embedding, entity_embedings, entity_mask, temperature=0.8)  # test

    setup_seed(20)
    logits1, _, embedding1 = net.forward(
        embedding,
        entity_embedings,
        entity_mask,
        selected_units=units,  # trainer, entity mask --> boolean
        temperature=0.8
    )

    print((logits1 - logits0).abs().max())
    print((embedding1 - embedding0).abs().max())
