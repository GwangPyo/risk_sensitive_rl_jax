
# risk_sensitive_rl_jax
Risk Sensitive RL packages for continuous control 

SAC: IQN + TQC based 
TD3: IQN + TQC based 

About IQN, 

See 
https://arxiv.org/abs/1806.06923
as well as 
https://arxiv.org/abs/2004.14547

About TQC 

See 
https://arxiv.org/abs/2005.04269


CMV SAC, TD3: Value based RL + Chaotic Mean variance Reduction algorithm, aversing martingale properties of rewards. 
See https://arxiv.org/abs/2006.12686

RCDSAC: Risk Conditioned SAC. 

See  https://arxiv.org/abs/2104.03111

NOTE: The docker file automatically install mujoco in the docker environment. If you do not need it, remove below the comment 
"# install mujco"
