from env import environment
from rl import DDPG

ITERATION= 500
STEP = 200
ON_TRAIN = False# TRUE IS TRAIN, FALSE IS PLOT

# set env
env = environment()
s_dim = env.state_dimension
a_dim = env.action_dimension
a_bound = env.action_boundary

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)


def train():
    # start training
    for i in range(ITERATION):
        s = env.reset() # INITIAL RESET EVERY ITERATION
        
        for j in range(STEP):
            env.render()#SHOW

            a = rl.choose_action(s)# rlchoose action 

            s_, r, done = env.step(a)# get feedback state and reward from environment

            rl.store_transition(s, a, r, s_)# store in memory

            
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == STEP-1:
                print('done')
                break
    rl.save()


def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    while True:
        s = env.reset()
        for _ in range(200):
            env.render()
            a = rl.choose_action(s)
            s, r, done = env.step(a)
            if done:
                break


if ON_TRAIN:
    train()
else:
    eval()



