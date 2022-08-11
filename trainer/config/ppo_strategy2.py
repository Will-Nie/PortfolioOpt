from easydict import EasyDict

trading0_ppo_config = dict(
    exp_name='trading_strategy2_onppo_seed0',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=10000000000000000,
    ),
    policy=dict(
        cuda=False,
        action_space='continuous',
        model=dict(
            obs_shape=140, # 4*28
            action_shape=28,
            action_space='continuous',
            encoder_hidden_size_list=[64, 64, 128],
            critic_head_hidden_size=128,
            actor_head_hidden_size=128,
        ),
        learn=dict(
            epoch_per_collect=2,
            batch_size=64,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            learner=dict(hook=dict(save_ckpt_after_iter=100)),
        ),
        collect=dict(
            n_sample=256,
            unroll_len=1,
            discount_factor=0.9,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=100, ), ),
    ),
)
trading0_ppo_config = EasyDict(trading0_ppo_config)
main_config = trading0_ppo_config
trading0_ppo_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
)
trading0_ppo_create_config = EasyDict(trading0_ppo_create_config)
create_config = trading0_ppo_create_config
