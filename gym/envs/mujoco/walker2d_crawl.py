import numpy as np
import mujoco_py

from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.walker2d import Walker2dEnv


class Walker2dCrawlEnv(Walker2dEnv):
    def __init__(self):
        super().__init__()

        # config
        self._config.update({
            "x_vel_reward": 2,
            "alive_reward": 1,
            "angle_reward": 0.1,
            "foot_reward": 0.01,
            "height_reward": 0,
            "x_vel_limit": 3,
            "collision_penalty": 10,
            "ceil_height": 1.0,
            "apply_force": 300,
            "random_steps": 5,
            "min_height": 0.3,
            "done_when_collide": 1,
        })

        # state
        self._ceils = None
        self._ceils_x = 2
        self._stage = 0
        self._collide = False
        self.t = 0

        # env info
        self.reward_type += ["x_vel_reward", "alive_reward", "angle_reward",
                             "foot_reward", "height_reward", "collision_penalty", "success",
                             "x_vel_mean", "height_mean", "nz_mean", "delta_h_mean"]
        self.ob_type = self.ob_shape.keys()
        self.collecting_demo = False
        self.randomness = 0.0

        mujoco_env.MujocoEnv.__init__(self, "walker_v2.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        x_before = self.data.qpos[0]
        right_foot_before = self.data.qpos[5]
        left_foot_before = self.data.qpos[8]
        self.do_simulation(a, self.frame_skip)
        x_after = self.data.qpos[0]
        right_foot_after = self.data.qpos[5]
        left_foot_after = self.data.qpos[8]

        self._reset_external_force()
        if np.random.rand(1) < self._config["prob_apply_force"]:
            self._apply_external_force()

        collision_penalty = 0
        if self.collision_detection('ceiling'):
            self._collide = True
            collision_penalty = -self._config["collision_penalty"]

        done = False
        x_vel_reward = 0
        angle_reward = 0
        height_reward = 0
        alive_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        height = self.data.qpos[1]
        angle = self.data.qpos[2]
        delta_h = self.data.body_xpos[1, 2] - max(self.data.body_xpos[4, 2], self.data.body_xpos[7, 2])
        nz = np.cos(angle)
        x_vel = (x_after - x_before) / self.dt
        x_vel = self._config["x_vel_limit"] - abs(x_vel - self._config["x_vel_limit"])
        right_foot_vel = abs(right_foot_after - right_foot_before) / self.dt
        left_foot_vel = abs(left_foot_after - left_foot_before) / self.dt

        # reward
        x_vel_reward = self._config["x_vel_reward"] * x_vel
        angle_reward = self._config["angle_reward"] * nz
        alive_reward = self._config["alive_reward"]
        foot_reward = -self._config["foot_reward"] * (right_foot_vel + left_foot_vel)
        reward = x_vel_reward + collision_penalty + angle_reward + \
            ctrl_reward + alive_reward + foot_reward

        done = height < self._config["min_height"] or (self._config["done_when_collide"] and self._collide)
        self.t += 1
        success = not done and self.t >= 1000
        if success: done = True

        ob = self._get_obs()
        info = {"x_vel_reward": x_vel_reward,
                "ctrl_reward": ctrl_reward,
                "angle_reward": angle_reward,
                "height_reward": height_reward,
                "alive_reward": alive_reward,
                "foot_reward": foot_reward,
                "collision_penalty": collision_penalty,
                "delta_h_mean": delta_h,
                "nz_mean": nz,
                "x_vel_mean": (x_after - x_before) / self.dt,
                "height_mean": height,
                "success": success}
        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        qacc = self.data.qacc
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10), qacc]).ravel()

    def get_ob_dict(self, ob):
        if len(ob.shape) > 1:
            return {
                'joint': ob[:, :17],
                'acc': ob[:, 17:26],
            }
        return {
            'joint': ob[:17],
            'acc': ob[17:26],
        }

    def reset_model(self):
        self._put_ceils()
        self._pass_state = False
        self._collide = False
        r = self._config["init_randomness"]
        self.set_state(
            self.init_qpos + np.random.uniform(low=-r, high=r, size=self.model.nq),
            self.init_qvel + np.random.uniform(low=-r, high=r, size=self.model.nv)
        )

        # more perturb
        for _ in range(int(self._config["random_steps"])):
            self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)
        self.t = 0

        return self._get_obs()

    def rollback(self, time, qpos, qvel, act, cur_pos):
        if cur_pos != -1:
            idx = self.model.geom_name2id('ceiling1')
            self.model.geom_pos[idx][0] = cur_pos
            self._ceils = {'pos': self.model.geom_pos[idx][0], 'size': self.model.geom_size[idx]}

        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(time, qpos, qvel,
                                         act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.do_simulation(act, self.frame_skip)

        self.sim.forward()     

        return self._get_obs()

    def _put_ceils(self):
        if not self.collecting_demo:
            idx = self.model.geom_name2id('ceiling_long')
        else:
            idx = self.model.geom_name2id('ceiling1')
        offset = np.random.uniform(-self.randomness, self.randomness)
        self.model.geom_pos[idx][0] = self._ceils_x + offset + self.model.geom_size[idx][0]
        self.model.geom_pos[idx][2] = self._config["ceil_height"]
        self._ceils = {'pos': self.model.geom_pos[idx][0], 'size': self.model.geom_size[idx]}

    def set_curbs_x_randomness_for_irl(self, task=None):
        self._ceils_x = 8.0
        self.collecting_demo = True
        self.randomness = 0.5
        self._config['apply_force'] = 100

    def get_sim_state(self):
        state = self.sim.get_state()
        obstacle_pos =  self._ceils['pos']
        return {'state': state, 'obstacle_pos': obstacle_pos}

    def get_x_pos(self):
        return self._get_walker2d_pos()

    def get_curb_pos(self):
        return self._ceils['pos']
    
    def get_dist(self):
        obs_obs = [5.1, 6.0]
        x_agent = self.get_x_pos()
        obs_start = self._ceils['pos'] - 1.5
        obs_end = self._ceils['pos'] +  1.5
        # print(obs_start, x_agent)
        if 0 < obs_start - x_agent and obs_start - x_agent < 3.0:
            obs_obs = [obs_start - x_agent, obs_end - x_agent]
        # print(obs_obs)
        return obs_obs
    
    def get_curbs_x_randomness(self):
        return self._ceils_x, self.randomness 

    def is_terminate(self, ob, init=False, env=None):
        if init:
            self._entered = False
        if ob[26] < 0:
            self._entered = True
        if ob[26] >= 5.1 and self._entered:
            return True
        return False

    def is_boundary_walk_front_for_obstacle(self):
        agent_x = self._get_walker2d_pos()
        dist = self._ceils['pos'] - 1.5 - agent_x
        start = 2.5
        end = 3.0
        if start <= dist and dist < end:
            return 0
        elif dist >= end:
            return 1
        else:
            return -1 # over   

    def is_boundary_crawl_rear_for_obstacle(self):
        agent_x = self._get_walker2d_pos()
        dist = self._ceils['pos'] - 1.5 - agent_x
        start = 1.0
        end = 3.0
        if start <= dist and dist < end:
            return 0
        elif dist >= end:
            return 1
        else:
            return -1 # over   

    def is_boundary_crawl_front_for_obstacle(self):
        agent_x = self._get_walker2d_pos()
        dist = agent_x - self._ceils['pos']
        start = 2.0
        end = 2.5
        if start <= dist and dist < end:
            return 0
        elif dist < start:
            return 1
        else:
            return -1 # over

    def is_boundary_walk_rear_for_obstacle(self):
        agent_x = self._get_walker2d_pos()
        start = -1.0
        end = 4.0
        
        if start <= agent_x and agent_x < end:
            return 0
        elif agent_x < start:
            return 1
        else:
            return -1 # over   

    def is_transition_boundary_for_obstacle(self):
        agent_x = self._get_walker2d_pos()
        dist = self._ceils['pos'] - 1.5 - agent_x
        start = 1.0
        end = 3.0
        if start <= dist and dist < end:
            return 0
        elif dist >= end:
            return 1
        else:
            return -1 # over
    
    def is_transition_boundary_rear_for_obstacle(self):
        agent_x = self._get_walker2d_pos()
        dist = agent_x - self._ceils['pos']
        start = 2.0
        end = 6.0
        if start <= dist and dist < end:
            return 0
        elif dist < start:
            return 1
        else:
            return -1 # over

