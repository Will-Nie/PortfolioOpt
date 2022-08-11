import sys
import os
sys.path.append("..")
sys.path.insert(0, os.getcwd())
import math
import torch
import importlib
import os
import threading
import time
import traceback
from copy import deepcopy
from typing import Any, ValuesView
import random
import platform
import numpy as np

import portpicker
import torch.multiprocessing as tm

from functools import partial
from bigrl.core.torch_utils.data_helper import to_device, to_share, to_contiguous, to_pin_memory
from bigrl.core.utils import EasyTimer
from bigrl.core.utils.import_helper import import_pipeline_module
from bigrl.core.utils.file_helper import loads
from bigrl.core.utils.redis_utils import start_redis_server, shutdown_redis, get_redis_ip_port_connect
from bigrl.core.worker.learner.data_collector import start_data_collector
from bigrl.core.worker.adapter.adapter import Adapter

from bigrl.core.rl_utils.vtrace_util import vtrace_from_importance_weights
from bigrl.core.torch_utils.collate_fn import default_collate_with_dim



class RLDataLoader(object):
    def __init__(self, learner, ) -> None:
        torch.set_num_threads(1)
        self.whole_cfg = learner.whole_cfg
        self.env_name = learner.env_name
        self.pipeline = learner.pipeline
        self.dir_path = learner.dir_path
        self.use_cuda = learner.use_cuda
        self.device = learner.device
        self.rank = learner.rank
        self.world_size = learner.world_size
        self.ip = learner.ip
        self.player_id = learner.player_id
        self.logger = learner.logger
        self.variable_record = learner.variable_record
        self.timer = EasyTimer(self.use_cuda)
        self.fake_dataloader = self.whole_cfg.learner.data.get('fake_dataloader', False)
        self.debug_mode = self.whole_cfg.learner.get('debug_mode', False)

        if self.fake_dataloader:
            self.use_adapter = False
        else:
            self.use_adapter = platform.system() == 'Windows' or self.whole_cfg.communication.get('use_adapter', False)

        if self.fake_dataloader:
            print(f'use fake dataloader {self.fake_dataloader}')
        elif self.use_adapter:
            self.max_buffer_size = self.whole_cfg.learner.data.get('max_buffer_size', 100)
        else:
            self.start_redis_server(learner)
            self.start_grpc_server(learner)

        if self.debug_mode:
            self.batch_size = 1
            self.worker_num = 1
        else:
            self.batch_size = self.whole_cfg.learner.data.get('batch_size', 1)
            self.worker_num = self.whole_cfg.learner.data.get('worker_num', 1)
        self.unroll_len = self.whole_cfg.learner.data.get('unroll_len', 1)
        Features = import_pipeline_module(self.env_name, self.pipeline, 'Features')
        features = Features(self.whole_cfg)
        get_rl_batch_data = features.get_rl_batch_data
        self.use_pin_memory = self.whole_cfg.learner.data.get('pin_memory', False) and self.use_cuda
        size = self.batch_size if not self.use_adapter else self.max_buffer_size
        self.shared_data = to_share(
            to_contiguous(get_rl_batch_data(unroll_len=self.unroll_len, batch_size=size )))
        if self.use_pin_memory:
            self.shared_data = to_pin_memory(self.shared_data)

        if not self.fake_dataloader:
            self.start_worker_process()

    def start_worker_process(self):
        self.worker_processes = []
        size = self.batch_size if not self.use_adapter else self.max_buffer_size
        self.signal_queue = tm.Queue(maxsize=size)
        self.done_flags = torch.tensor([False for _ in range(size)]).share_memory_()
        if not self.use_adapter:
            worker_loop = partial(worker_loop_grpc, signal_queue=self.signal_queue, done_flags=self.done_flags,
                                  shared_data=self.shared_data, redis_port=self.redis_port, cfg=self.whole_cfg,
                                  variable_record=self.variable_record)
        else:
            self.used_count = torch.zeros(self.max_buffer_size)
            worker_loop = partial(worker_loop_adapter, signal_queue=self.signal_queue, done_flags=self.done_flags,
                                  shared_data=self.shared_data, cfg=self.whole_cfg, player_id=self.player_id,
                                  variable_record=self.variable_record)
        for worker_idx in range(self.worker_num):
            if not self.debug_mode:
                worker_process = tm.Process(target=worker_loop,
                                            args=(),
                                            daemon=True)
            else:
                worker_process = threading.Thread(target=worker_loop,
                                                  args=(),
                                                  daemon=True)
            worker_process.start()
            self.worker_processes.append(worker_process)

        for idx in range(size):
            self.signal_queue.put(idx)

    def start_redis_server(self, learner):
        self.redis_address_dir = os.path.join(self.dir_path, 'redis_address')
        learner.setup_dir(self.redis_address_dir)
        self.redis_port = portpicker.pick_unused_port()
        start_redis_server(self.redis_port, self.redis_address_dir)

    def start_grpc_server(self, learner):
        self.grpc_address_dir = os.path.join(self.dir_path, 'grpc_address')
        learner.setup_dir(self.grpc_address_dir)
        self.grpc_port = portpicker.pick_unused_port()
        self.data_collector_process = start_data_collector(self.ip, self.redis_port, self.grpc_port,
                                                           self.grpc_address_dir, self.whole_cfg, )

    def get_data(self):
        if not self.use_adapter:
            return self._get_data_grpc()
        else:
            return self._get_data_adapter()

    def _get_data_grpc(self) -> Any:
        if not self.fake_dataloader:
            while True:
                if (self.done_flags == True).all():
                    break
                else:
                    time.sleep(0.001)
        if self.use_cuda:
            with self.timer:
                batch_data = to_device(self.shared_data, self.device)
            self.variable_record.update_var({'to_device': self.timer.value})
        else:
            with self.timer:
                batch_data = deepcopy(self.shared_data)
            self.variable_record.update_var({'to_device': self.timer.value})
        if not self.fake_dataloader:
            self.done_flags.copy_(torch.zeros_like(self.done_flags))
            for batch_idx in range(self.batch_size):
                self.signal_queue.put(batch_idx)
        return batch_data

    def _get_data_adapter(self) -> Any:
        while True:
            start = random.randint(0, self.max_buffer_size - self.batch_size)
            end = start + self.batch_size
            if self.done_flags[start: end].all():
                self.used_count[start: end] += 1
                for i in range(start, end):
                    if self.used_count[i] >= self.whole_cfg.learner.data.max_use:
                        self.done_flags[i] = False
                        self.used_count[i] = 0
                        self.signal_queue.put(i)
                break
            else:
                time.sleep(0.001)
        if self.use_cuda:
            with self.timer:
                batch_data = to_device(slice_data(self.shared_data, start, end), self.device)
            self.variable_record.update_var({'to_device': self.timer.value})
        else:
            with self.timer:
                batch_data = deepcopy(self.shared_data)
            self.variable_record.update_var({'to_device': self.timer.value})
        return batch_data


    def close(self):
        if not self.fake_dataloader:
            shutdown_redis(self.redis_port)
            self.data_collector_process.terminate()
            if not self.debug_mode:
                for p in self.worker_processes:
                    p.terminate()
                for p in self.worker_processes:
                    p.join()
            time.sleep(1)
            print('has already close all subprocess in RLdataloader')
        return True


def worker_loop_grpc(signal_queue, done_flags, shared_data, redis_port, cfg, variable_record):
    torch.set_num_threads(1)
    min_sample_size = cfg.learner.data.get('min_sample_size', 20)
    timer = EasyTimer(cuda=False)
    lua_script = """
                local buffer_size = tonumber(redis.call('scard', 'traj_paths'))
                local min_sample_size = tonumber(KEYS[1])
                if buffer_size < min_sample_size then
                    return 0
                end
                local randkey = redis.call('srandmember', 'traj_paths')
                local return_data = redis.call('get',randkey)
                local count = tonumber(redis.call('hget', 'counts', randkey))
                if count == 1 then
                    redis.call('del',randkey)
                    redis.call('srem', 'traj_paths', randkey)
                    redis.call('hdel', 'counts', randkey)
                else
                    redis.call('hset', 'counts', randkey, count-1)
                end
                return return_data
                """

    redis_connect = get_redis_ip_port_connect(ip='127.0.0.1', redis_port=redis_port)
    cmd = redis_connect.register_script(lua_script)

    while True:
        if signal_queue.qsize() > 0:
            batch_idx = signal_queue.get()
            while True:
                try:
                    with timer:
                        while True:
                            get_data = cmd([min_sample_size])
                            if not get_data:
                                time.sleep(0.001)
                            else:
                                break
                    variable_record.update_var({'get_data': timer.value})

                    with timer:
                        traj_data = loads(get_data, fs_type='pyarrow')
                    variable_record.update_var({'loads': timer.value})

                    with timer:
                        copy_data(batch_idx, traj_data=traj_data, shared_data=shared_data)
                    variable_record.update_var({'collate_fn': timer.value})

                    done_flags[batch_idx]=True
                    break
                except Exception as e:
                    print(f'[Loader Error]{e}', flush=True)
                    redis_connect = get_redis_ip_port_connect(ip='127.0.0.1', redis_port=redis_port)
                    cmd = redis_connect.register_script(lua_script)
                    time.sleep(0.001)
        else:
            time.sleep(0.001)


def worker_loop_adapter(signal_queue, done_flags, shared_data, cfg, player_id, variable_record):
    torch.set_num_threads(1)
    adapter = Adapter(cfg)
    timer = EasyTimer(cuda=False)
    worker_num = cfg.communication.get('adapter_traj_worker_num', 1)
    traj_fs_type = cfg.communication.get('traj_fs_type','nppickle')
    while True:
        with timer:
            data = adapter.pull(token=player_id + 'traj', fs_type=traj_fs_type, sleep_time=0.5, worker_num=worker_num)
        variable_record.update_var({'get_data': timer.value})
        while True:
            if signal_queue.qsize() > 0:
                batch_idx = signal_queue.get()
                with timer:
                    copy_data(batch_idx, traj_data=data, shared_data=shared_data)
                variable_record.update_var({'collate_fn': timer.value})
                done_flags[batch_idx] = True
                break
            else:
                time.sleep(0.001)


def _copy_data(dest_tensor, src_tensor, key=''):
    if dest_tensor.shape == src_tensor.shape:
        if dest_tensor.dtype != src_tensor.dtype:
            print(f'{key} dtype not same, dest: {dest_tensor.dtype}, src: {src_tensor.dtype}', flush=True)
        dest_tensor.copy_(src_tensor)
        return True
    else:
        print(key, dest_tensor.shape, src_tensor.shape)
        print(key, dest_tensor.dtype, src_tensor.dtype)
        raise NotImplementedError
        return False


def copy_data(batch_idx, traj_data, shared_data):
    for k, v in traj_data.items():
        if isinstance(v, torch.Tensor):
            try:
                _copy_data(shared_data[k][:, batch_idx], traj_data[k], key=k)
            except Exception as e:
                print(k, e, flush=True)
                print(''.join(traceback.format_tb(e.__traceback__)), flush=True)
        elif isinstance(v, dict):
            copy_data(batch_idx,v,shared_data[k])
        else:
            print(k, type(v))
            raise NotImplementedError

def slice_data(data, start, end):
    if isinstance(data, dict):
        return {k: slice_data(v, start, end) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data[:, start: end]



class ReinforcementLoss:
    def __init__(self, cfg: dict,) -> None:
        self.whole_cfg = cfg
        '''
        # loss parameters
        self.loss_parameters = self.whole_cfg.learner.get('loss_parameters',{})
        self.gamma = self.loss_parameters.get('gamma', 0.99)
        self.lambda_ = self.loss_parameters.get('lambda', 0.8)
        self.clip_rho_threshold = self.loss_parameters.get('clip_rho_threshold', 1)
        self.clip_pg_rho_threshold = self.loss_parameters.get('clip_pg_rho_threshold', 1)

        # loss weights
        self.loss_weights = self.whole_cfg.learner.get('loss_weights',{})
        self.policy_weight = self.loss_weights.get('policy', 1)
        self.value_weight = self.loss_weights.get('value', 0.5)
        self.entropy_weight = self.loss_weights.get('entropy', 0.01)
        '''

        # loss parameters
        self.loss_parameters = {}
        self.gamma = 0.99
        self.lambda_ = 0.8
        self.clip_rho_threshold = 1
        self.clip_pg_rho_threshold = 1

        # loss weights
        self.loss_weights = {}
        self.policy_weight = 1
        self.value_weight = 0.5
        self.entropy_weight = 0.01
        self.only_update_value = False

    def compute_loss(self, inputs):
        # take data from inputs
        actions = inputs['action']                                      # T, B,selected_units_num
        behaviour_policy_logits = inputs['behaviour_logit']             # T, B,selected_units_num, max_entity_num
        rewards = inputs['reward']                                      # T, B
        logits = inputs['logit']                                        # T, B, selected_units_num, max_entity_num
        values = inputs['value']                                        # T+1, B
        discounts = (1 - inputs['done'].float()) * self.gamma           # T, B

        # behaviour_policy_logits = inputs['logit']
        # pi_behaviour = torch.distributions.Categorical(logits=behaviour_policy_logits)

        unroll_len = rewards.shape[0]
        batch_size = rewards.shape[1]


        # reshape logits and values
        # logits = logits.reshape(unroll_len + 1, batch_size, -1)  # ((T+1), B,-1)
        # target_policy_logits = logits[:-1]  # ((T), B,-1)
        target_policy_logits = logits
        
        values = values.reshape(unroll_len + 1, batch_size)  # ((T+1), B)
        target_values, bootstrap_value = values[:-1], values[-1]  # ((T), B) ,(B)

        # get dist for behaviour policy and target policy
        pi_target = torch.distributions.Categorical(logits=target_policy_logits)
        target_action_log_probs = pi_target.log_prob(actions)
        entropy = (pi_target.entropy()/math.log(target_policy_logits.shape[-1])).mean(-1)

        pi_behaviour = torch.distributions.Categorical(logits=behaviour_policy_logits)
        behaviour_action_log_probs = pi_behaviour.log_prob(actions)

        with torch.no_grad():
            log_rhos = target_action_log_probs - behaviour_action_log_probs
            log_rhos = torch.sum(log_rhos, 2)

        # Make sure no gradients backpropagated through the returned values.
        with torch.no_grad():
            vtrace_returns = vtrace_from_importance_weights(log_rhos, discounts, rewards, target_values,
                                                            bootstrap_value,
                                                            clip_rho_threshold=self.clip_rho_threshold,
                                                            clip_pg_rho_threshold=self.clip_pg_rho_threshold)

        # Policy-gradient loss.
        policy_gradient_loss = -torch.mean((target_action_log_probs.sum(-1) * vtrace_returns.pg_advantages))

        # Critic loss.
        critic_loss = torch.mean(0.5 * ((target_values - vtrace_returns.vs) ** 2))

        # Entropy regulariser.
        entropy_loss = - torch.mean(entropy)

        # Combine weighted sum of actor & critic losses.
        if self.only_update_value:
            total_loss = self.value_weight * critic_loss
        else:
            total_loss = self.policy_weight * policy_gradient_loss + self.value_weight * critic_loss + self.entropy_weight * entropy_loss

        loss_info_dict = {
            'total_loss': total_loss.item(),
            'pg_loss': policy_gradient_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': -entropy_loss.item(),
            'reward': rewards.mean().item(),
            'value': values.mean().item(),
        }
        return total_loss, loss_info_dict


class Features:
    def __init__(self, cfg={}):
        self.cfg = cfg

    def get_rl_step_data(self,last=False):
        data = {}
        data['obs'] = torch.zeros(size=(1, 4, 84, 84), dtype=torch.float)
        if not last:
            data['action'] = torch.zeros(size=(1,),dtype=torch.long)
            data['action_logp'] = torch.zeros(size=(1,),dtype=torch.float)
            data['reward'] = torch.zeros(size=(1,),dtype=torch.float)
            data['done'] = torch.zeros(size=(1,),dtype=torch.bool)
            data['model_last_iter'] = torch.zeros(size=(1,),dtype=torch.float)
        return data

    def get_rl_traj_data(self,unroll_len):
        traj_data_list = []
        for _ in range(unroll_len):
            traj_data_list.append(self.get_rl_step_data())
        traj_data_list.append(self.get_rl_step_data(last=True))
        traj_data = default_collate_with_dim(traj_data_list, cat=True)
        return traj_data

    def get_rl_batch_data(self,unroll_len,batch_size):
        batch_data_list = []
        for _ in range(batch_size):
            batch_data_list.append(self.get_rl_traj_data(unroll_len))
        batch_data = default_collate_with_dim(batch_data_list,dim=1)
        return batch_data

# 串行， 一个env -- 但是可以通过在Feature层多写一个循环来多收集几个episode， 这样效果会好一些
# 先实现这个版本的话，暂时用不到 RLDataLoader
from model.trading_pointer import PointerNetwork
from model.vac import VAC
from easydict import EasyDict


from torch.optim.adam import Adam
from torch.optim.rmsprop import RMSprop


def build_optimizer(cfg, params):
    optimizer_type = cfg.get('type', 'adam')
    learning_rate = cfg.get('learning_rate', 0.001)
    weight_decay = cfg.get('weight_decay', 0)
    eps = float(cfg.get('eps', 1e-8))

    if optimizer_type == 'adam':
        decay = cfg.get('decay', 0.999)
        momentum = cfg.get('momentum', 0.9)
        betas = (momentum, decay)
        amsgrad = cfg.get('amsgrad', False)
        optimizer = Adam(params=params, lr=learning_rate, weight_decay=weight_decay, eps=eps, betas=betas,
                         amsgrad=amsgrad)
    elif optimizer_type == 'rmsprop':
        alpha = cfg.get('decay', 0.9)
        momentum = cfg.get('momentum', 0)
        optimizer = RMSprop(params=params, lr=learning_rate, alpha=alpha, weight_decay=weight_decay, eps=eps,
                            momentum=momentum)
    return optimizer


class FeaturesSerial:
    def __init__(self, env, kwargs, model_config, optimiser_config):
        import gym
        self.env = gym.make(env, **kwargs)
        self.batch_size = 8
        self.unroll_len = 10
        self.default_model_config = EasyDict(model_config)
        self.optimiser_config = EasyDict(optimiser_config)
        self.net = PointerNetwork(self.default_model_config)
        self.optimiser = build_optimizer(optimiser_config, self.net.parameters())
    
    def number_to_zero_one_action(self, action):
        action_num = self.default_model_config.model.max_entity_num
        action_zero_to_one = np.array(np.zeros(action_num))
        for i in action:
            action_zero_to_one[i] = 1
        return action_zero_to_one

    def collect_data(self):
        data={}
        dummy_state_unroll_len_batch_size = []
        dummy_action_unroll_len_batch_size = []
        dummy_action_logp_unroll_len_batch_size = []
        dummy_logits_unroll_len_batch_size = []
        dummy_reward_unroll_len_batch_size = []
        dummy_done_unroll_len_batch_size = []
        dummy_value_logit_unroll_len_batch_size = []
        dummy_value_unroll_len_batch_size = []

        for _ in range(self.batch_size):
            state = self.env.reset()
            entity_mask = torch.rand(size=(1, self.default_model_config.model.max_entity_num,)) > 0
            entity_embedings = torch.rand(size=(1, self.default_model_config.model.max_entity_num, self.default_model_config.model.entity_embedding_dim,))
            with torch.no_grad():
                value_logit, action, _, _, value = self.net(torch.tensor(state.reshape(1,140)).float(), entity_embedings, entity_mask) # value logit is the target logit
            logits1, _, _, _, _ = self.net(torch.tensor(state.reshape(1,140)).float(), entity_embedings, entity_mask, selected_units=action) # this is the learner
            dummy_state_unroll_len = []; dummy_state_unroll_len.append(state)
            dummy_action_unroll_len = []
            dummy_action_logp_unroll_len = []
            dummy_logits_unroll_len = []
            dummy_reward_unroll_len = []
            dummy_done_unroll_len = []
            dummy_value_logit_unroll_len = []
            dummy_value_unroll_len = []; dummy_value_unroll_len.append(value)
            for _ in range( self.unroll_len):
                with torch.no_grad():
                    value_logit, action, _, _, value = self.net(torch.tensor(state.reshape(1,140)).float(), entity_embedings, entity_mask)
                logits1, _, _, _, _ = self.net(torch.tensor(state.reshape(1,140)).float(), entity_embedings, entity_mask, selected_units=action)
                pi_behaviour = torch.distributions.Categorical(logits=logits1)
                action_logp = pi_behaviour.log_prob(action)
                next_obs, reward, done, info = self.env.step(self.number_to_zero_one_action(action))


                state = next_obs

                dummy_state_unroll_len.append(next_obs)
                dummy_action_unroll_len.append(action)
                dummy_action_logp_unroll_len.append(action_logp)
                dummy_logits_unroll_len.append(value_logit)
                dummy_reward_unroll_len.append(reward)
                dummy_done_unroll_len.append(done)
                dummy_value_logit_unroll_len.append(logits1)
                dummy_value_unroll_len.append(value)

                
            
            dummy_state_unroll_len_batch_size.append(dummy_state_unroll_len)
            dummy_value_logit_unroll_len_batch_size.append(torch.stack(dummy_value_logit_unroll_len, 0))
            dummy_value_unroll_len_batch_size.append(torch.stack(dummy_value_unroll_len,0))
            dummy_action_unroll_len_batch_size.append(torch.stack(dummy_action_unroll_len,0))
            dummy_action_logp_unroll_len_batch_size.append(torch.stack(dummy_action_logp_unroll_len,0))
            dummy_logits_unroll_len_batch_size.append(torch.stack(dummy_logits_unroll_len,0))
            dummy_reward_unroll_len_batch_size.append(dummy_reward_unroll_len)
            dummy_done_unroll_len_batch_size.append(dummy_done_unroll_len)



        data['obs'] = torch.tensor(np.stack(dummy_state_unroll_len_batch_size), dtype=torch.float).permute(1,0,2)
        data['logit'] = torch.stack(dummy_value_logit_unroll_len_batch_size, 1).squeeze(2)
        data['value'] = torch.stack(dummy_value_unroll_len_batch_size, 1).squeeze(2)
        data['action'] = torch.stack(dummy_action_unroll_len_batch_size, 1).squeeze(2)
        #data['action_logp'] = torch.tensor(torch.stack(dummy_action_logp_unroll_len_batch_size, 1), dtype=torch.float).squeeze(2)
        data['behaviour_logit'] = torch.stack(dummy_logits_unroll_len_batch_size, 1).squeeze(2)
        data['reward'] = torch.tensor(np.stack(dummy_reward_unroll_len_batch_size), dtype=torch.float).permute(1,0)
        data['done'] = torch.tensor(np.stack(dummy_done_unroll_len_batch_size), dtype=torch.float).permute(1,0)

        return data, self.optimiser


def main(FeaturesSerial, ReinforcementLoss, max_train_iter, env, kwargs, model_config, optimiser_config):
    env = FeaturesSerial(env, kwargs, model_config, optimiser_config)
    loss = ReinforcementLoss('tobemodified')
    for _ in range(max_train_iter):
        input, optimiser = env.collect_data()
        print('*'*20)
        print('Collect finished --> start optimise')
        print('*'*20)
        # you can write the learning forward code inside the collect_data file or write like the three lines below
        # flatten
        # logits1, _, _, _, _ = self.net(torch.tensor(state.reshape(1,140)).float(), entity_embedings, entity_mask, selected_units=action)
        # View
        total_loss, loss_info_dict = loss.compute_loss(input)
        print('*'*20)
        print('Optimisation ends --> start collecting')
        print('*'*20)
        optimiser.zero_grad()
        total_loss.backward()
        optimiser.step()

if __name__=='__main__':
    import gym
    import pandas as pd
    import numpy as np

    from env.gym_env.portfolio_env import StockPortfolioEnvStr1
    from data.data_demo import data_demo1

    # Process your data here [doing data  cleaning, features engineering here]
    tech_indicator_list = ['macd', 'rsi_30', 'cci_30', 'dx_30']
    train, test = data_demo1(tech_indicator_list)
    chosen_stock_number = 14
    max_step = 1000
    stock_dimension = len(train.tic.unique())
    state_space = stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    feature_dimension = len(tech_indicator_list)
    print(f"Feature Dimension: {feature_dimension}")


    env_train_kwargs = {
        'df': train,
        "hmax": 100, 
        "initial_amount": 1000000, 
        "transaction_cost_pct": 0, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": tech_indicator_list, 
        "action_space": stock_dimension, 
        "reward_scaling": 1e-1,
        "chosen_stock_num": chosen_stock_number,
        "max_step": max_step
    }

    model_config = {
    'model': {'input_dim': 140, 'max_selected_units_num': chosen_stock_number, 'max_entity_num': stock_dimension,
                'entity_embedding_dim': 16, 'key_dim': 32, 'func_dim': 256,
                'lstm_hidden_dim': 32, 'lstm_num_layers': 1,
                'activation': 'relu', 'entity_reduce_type': 'selected_units_num',# ['constant', 'entity_num', 'selected_units_num']
                }
     }

    optimiser_config = {'cfg':{'type':'adam', 'learning_rate': 0.001, 'weight_decay':0, 'eps':1e-8, 'decay':0.009, 'momentum': 0.9, 'amsgrad':False}}    


    main(FeaturesSerial, ReinforcementLoss, 100, 'trading-v1', env_train_kwargs, model_config, optimiser_config)