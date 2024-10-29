uav_config = {
    ## Commented parameters are set to default values in ppo

    #==========  env config ==========
    'env': 'PongNoFrameskip-v4',  # environment name
    'continuous_action': False,  # action type of the environment
    'env_num': 1,  # number of the environment
    'seed': None,  # seed of the experiment
    'xparl_addr': None,  # xparl address for distributed training

    #==========  training config ==========
    'train_total_steps': int(4000),  # max training steps
    'step_nums': 300,  # data collecting time steps (ie. T in the paper)
    'num_minibatches': 4,  # number of training minibatches per update.
    'update_epochs': 4,  # number of epochs for updating (ie K in the paper)
    'eval_episode': 3,
    'test_every_steps': int(5e3),  # interval between evaluations

    #========== coefficient of ppo ==========
    'initial_lr': 2.5e-4,  # start learning rate
    'lr_decay': True,  # whether or not to use linear decay rl
    'clip_param': 0.1,  # epsilon in clipping loss
    'entropy_coef': 0.01,  # Entropy coefficient (ie. c_2 in the paper)
}