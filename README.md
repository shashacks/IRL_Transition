# Submission_IRL_Transtion

## Descriptions
Humans decompose novel complex tasks into simpler ones to exploit previously learned skills. Analogously, hierarchical reinforcement learning seeks to leverage lower-level policies for simple tasks to solve complex ones. However, because each lower-level policy induces a different distribution of states, transitioning from one lower-level policy to another may fail due to an unexpected starting state. We introduce transition policies that smoothly connect lower-level policies by producing a distribution of states and actions that matches what is expected by the next policy. Training transition policies is challenging because the natural reward signal---whether the next policy can execute its subtask successfully---is sparse. By training transition policies via adversarial inverse reinforcement learning to match the distribution of expected states and actions, we avoid relying on task-based reward. To further improve performance, we use deep Q-learning with a binary action space to determine when to switch from a transition policy to the next pre-trained policy, using the success or failure of the next subtask as the reward. Although the reward is still sparse, the problem is less severe due to the simple binary action space. We demonstrate our method on continuous bipedal locomotion and arm manipulation tasks that require diverse skills. We show that it smoothly connects the lower-level policies, achieving higher success rates than previous methods that search for successful trajectories based on a reward function, but do not match the state distribution.

## Dependencies
To install the dependencies below:
* conda create -n [name] python=3.6
* conda install pytorch==1.3.0 torchvision==0.4.1 cudatoolkit=11.0 -c pytorch
* pip install tensorflow=1.14
* pip install mpi4py
* pip install joblib
* pip install tensorboard
* pip install scipy
* pip install cloudpickle
* pip install psutil
* pip install tqdm
* pip install -U 'mujoco-py<2.1,>=2.0'
* sudo apt-get update && sudo apt-get install libopenmpi-dev

#### Note that follow instruction_xxx.txt. We prepared the pre-trained policies provided by Lee et al. 2019.


### Collect data of pre-trained policy for Obstacle course (Walk, Jump Crawl)
* Walk for Jump (front)
<pre>
<code>
python -m main --hrl False --train False --complex_task obstacle --collect_exp_data True --env Walker2dJump-v1 --suffix wfj --primitive_env Walker2dForward-v1 --primitive_path data/Walker2dForward.forward_ICLR2019
</code>
</pre>
* Walk for Jump (rear)
<pre>
<code>
python -m main --hrl False --train False --complex_task obstacle --collect_exp_data True --env Walker2dJump-v1 --front False --suffix wrj --primitive_env Walker2dForward-v1 --primitive_path data/Walker2dForward.forward_ICLR2019
</code>
</pre>
* Walk for Crawl (front)
<pre>
<code>
python -m main --hrl False --train False --complex_task obstacle --collect_exp_data True --env Walker2dCrawl-v1 --suffix wfc --primitive_env Walker2dForward-v1 --primitive_path data/Walker2dForward.forward_ICLR2019
</code>
</pre>
* Walk for Crawl (rear)
<pre>
<code>
python -m main --hrl False --train False --complex_task obstacle --collect_exp_data True --env Walker2dCrawl-v1 --front False --suffix wrc --primitive_env Walker2dForward-v1 --primitive_path data/Walker2dForward.forward_ICLR201
</code>
</pre>
* Jump for Walk (front)
<pre>
<code>
python -m main --hrl False --train False --complex_task obstacle --collect_exp_data True --env Walker2dJump-v1 --suffix jf --primitive_env Walker2dJump-v1 --primitive_path data/Walker2dJump.jump_ICLR2019
</code>
</pre>
* Jump for Walk (rear)
<pre>
<code>
python -m main --hrl False --train False --complex_task obstacle --collect_exp_data True --env Walker2dJump-v1 --front False --suffix jr --primitive_env Walker2dJump-v1 --primitive_path data/Walker2dJump.jump_ICLR2019
</code>
</pre>
* Crawl for Walk (front)
<pre>
<code>
python -m main --hrl False --train False --complex_task obstacle --collect_exp_data True --env Walker2dCrawl-v1 --suffix cf --primitive_env Walker2dCrawl-v1 --primitive_path data/Walker2dCrawl.crawl_ICLR2019
</code>
</pre>
* Crawl for Walk (rear)
<pre>
<code>
python -m main --hrl False --train False --complex_task obstacle --collect_exp_data True --env Walker2dCrawl-v1 --front False --suffix cr --primitive_env Walker2dCrawl-v1 --primitive_path data/Walker2dCrawl.crawl_ICLR2019
</code>
</pre>

### Train transition policies for Obstacle course
* Walk -> Jump
<pre>
<code>
python -m main --hrl False --train True --complex_task obstacle --irl_training True --env Walker2dJump-v1 --exp_data_path_1 data/exp_demo/Walker2dForward-v1_wfj --exp_data_path_2 data/exp_demo/Walker2dJump-v1_jr --env_1 obstacle_walk --env_2 jump
</code>
</pre>
* Jump -> Walk
<pre>
<code>
python -m main --hrl False --train True --complex_task obstacle --irl_training True --env Walker2dJump-v1 --front False --exp_data_path_1 data/exp_demo/Walker2dJump-v1_jf --exp_data_path_2 data/exp_demo/Walker2dForward-v1_wrj --env_1 obstacle_jump --env_2 walk
</code>
</pre>
* Walk -> Crawl
<pre>
<code>
python -m main --hrl False --train True --complex_task obstacle --irl_training True --env Walker2dCrawl-v1 --exp_data_path_1 data/exp_demo/Walker2dForward-v1_wfc --exp_data_path_2 data/exp_demo/Walker2dCrawl-v1_cr --env_1 obstacle_walk --env_2 crawl
</code>
</pre>
* Crawl -> Walk
<pre>
<code>
python -m main --hrl False --train True --complex_task obstacle --irl_training True --env Walker2dCrawl-v1 --front False --exp_data_path_1 data/exp_demo/Walker2dCrawl-v1_cf --exp_data_path_2 data/exp_demo/Walker2dForward-v1_wrc --env_1 obstacle_crawl --env_2 walk
</code>
</pre>

### Train DQNs for Obstacle course
<pre>
<code>
python -m main --hrl True --train True --complex_task obstacle --env Walker2dObstacleCourse-v1 --pi1_env Walker2dForward-v1 --pi2_env Walker2dJump-v1 --pi3_env Walker2dCrawl-v1 --pi1 data/Walker2dForward.forward_ICLR2019 --pi2 data/Walker2dJump.jump_ICLR2019 --pi3 data/Walker2dCrawl.crawl_ICLR2019 --pi12 data/transition/obstacle_walk_jump/step10000000/model.pt --pi21 data/transition/obstacle_jump_walk/step10000000/model.pt --pi13 data/transition/obstacle_walk_crawl/step10000000/model.pt --pi31 data/transition/obstacle_crawl_walk/step10000000/model.pt --fname obstacle
</code>
</pre>

### Test trained networks
<pre>
<code>
python -m main --hrl True --train Fasle --complex_task obstacle --env Walker2dObstacleCourse-v1 --pi1_env Walker2dForward-v1 --pi2_env Walker2dJump-v1 --pi3_env Walker2dCrawl-v1 --pi1 data/Walker2dForward.forward_ICLR2019 --pi2 data/Walker2dJump.jump_ICLR2019 --pi3 data/Walker2dCrawl.crawl_ICLR2019 --pi12 data/transition/obstacle_walk_jump/step10000000/model.pt --pi21 data/transition/obstacle_jump_walk/step10000000/model.pt --pi13 data/transition/obstacle_walk_crawl/step10000000/model.pt --pi31 data/transition/obstacle_crawl_walk/step10000000/model.pt --q12 data/q_network/obstacle/30000_q12.pt --q21 data/q_network/obstacle/30000_q21.pt --q13 data/q_network/obstacle/30000_q13.pt --q31 data/q_network/obstacle/30000_q31.pt
</code>
</pre>


### References 
* https://github.com/openai/spinningup
* https://github.com/youngwoon/transition
* https://github.com/ku2482/gail-airl-ppo.pytorch/tree/master/gail_airl_ppo
