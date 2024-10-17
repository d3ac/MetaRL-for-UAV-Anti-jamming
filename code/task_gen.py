import torch
import random
import numpy as np
import random
import tqdm
import math
import pandas as pd
import pickle

from ENV.ENV_DDRQN import Environ

def generate_task(n_task, env):
    tasks = []
    Trange = tqdm.tqdm(range(n_task))
    type_id = np.random.randint(1,5, n_task)
    for i in Trange:
        task = env.generate_p_trans(type_id[i])
        tasks.append(task)
    return tasks

if __name__ == '__main__':
    env = Environ()
    tasks = generate_task(300, env)
    with open('tasks.pkl', 'wb') as f:
        pickle.dump(tasks, f)