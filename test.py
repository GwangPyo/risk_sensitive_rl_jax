from risk_sensitive_rl.rl_agents import CMVTD3
import gym

if __name__ == '__main__':
    import os
    os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    rewards_results = []

    for i in range(1, 3):

        model = CMVTD3(
            env=gym.make('Hopper-v3'),
            lr_critic=1e-3, lr_actor=1e-3,
            seed=i,
            batch_size=128, warmup_steps=10000,
            risk_param=0.5
         )
        model.learn(int(3e+5), log_interval=1)

        model.save('td3_cvar_worst_{}.npz'.format(i))
