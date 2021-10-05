import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py
from gym.envs.mujoco.walker2d import Walker2dEnv


class Walker2dBackwardEnv(Walker2dEnv):
    def __init__(self):
        super().__init__()

        # config
        self._config.update({
            "x_vel_reward": 1,
            "alive_reward": 1,
            "angle_reward": 0.1,
            "foot_reward": 0.01,
            "height_reward": 1,
            "x_vel_limit": 3,
            "apply_force": 100,
            "random_steps": 5,
        })

        # state
        self.t = 0

        # env info
        self.reward_type += ["x_vel_reward", "alive_reward", "angle_reward",
                             "foot_reward", "height_reward", "success",
                             "x_vel_mean", "height_mean", "nz_mean", "delta_h_mean"]
        self.ob_type = self.ob_shape.keys()
        self.x_pos_pivot = 4.5 + np.random.rand(1)

        mujoco_env.MujocoEnv.__init__(self, 'walker_v1.xml', 4)
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

        done = False
        x_vel_reward = 0
        angle_reward = 0
        height_reward = 0
        alive_reward = 0
        foot_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        height = self.data.qpos[1]
        angle = self.data.qpos[2]
        delta_h = self.data.body_xpos[1, 2] - max(self.data.body_xpos[4, 2], self.data.body_xpos[7, 2])
        nz = np.cos(angle)
        x_vel = -(x_after - x_before) / self.dt
        x_vel = self._config["x_vel_limit"] - abs(x_vel - self._config["x_vel_limit"])
        right_foot_vel = abs(right_foot_after - right_foot_before) / self.dt
        left_foot_vel = abs(left_foot_after - left_foot_before) / self.dt

        # reward
        x_vel_reward = self._config["x_vel_reward"] * x_vel
        angle_reward = self._config["angle_reward"] * nz
        height_reward = -self._config["height_reward"] * abs(1.1 - delta_h)
        alive_reward = self._config["alive_reward"]
        foot_reward = -self._config["foot_reward"] * (right_foot_vel + left_foot_vel)
        reward = x_vel_reward + angle_reward + height_reward + \
            ctrl_reward + alive_reward + foot_reward

        # fail
        done = height < self._config["min_height"]
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
                "delta_h_mean": delta_h,
                "nz_mean": nz,
                "x_vel_mean": -(x_after - x_before) / self.dt,
                "height_mean": height,
                "success": success}
        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        qacc = self.data.qacc
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10), qacc]).ravel()

    def get_x_pos(self):
        return self._get_walker2d_pos()

    def get_q_pos(self):
        return self.data.qpos[0]

    def get_ob_dict(self, ob):
        if len(ob.shape) > 1:
            return {'joint': ob[:, :17], 'acc': ob[:, 17:26]}
        else:
            return {'joint': ob[:17], 'acc': ob[17:26]}

    def get_sim_state(self):
        state = self.sim.get_state()
        curb_obs = -1
        return {'state': state, 'obstacle_pos': curb_obs}

    def rollback(self, time, qpos, qvel, act, cur_pos):
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(time, qpos, qvel,
                                         act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.do_simulation(act, self.frame_skip)

        self.sim.forward()     
        return self._get_obs()

    def reset_model(self):
        init_randomness = self._config["init_randomness"]
        qpos = self.init_qpos + np.random.uniform(low=-init_randomness,
                                                  high=init_randomness,
                                                  size=self.model.nq)
        qvel = self.init_qvel + np.random.uniform(low=-init_randomness,
                                                  high=init_randomness,
                                                  size=self.model.nv)
        self.set_state(qpos, qvel)

        # init target
        self._set_pos('target_forward', (10, 0, 0))
        self._set_pos('target_backward', (-10, 0, 0))

        # more perturb
        for _ in range(int(self._config["random_steps"])):
            self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)
        self.t = 0
        self.x_pos_pivot = 4.5 + np.random.rand(1)
        return self._get_obs()

    def is_terminate(self, ob, init=False, env=None):
        # Forward-Backward
        return ob[26] > 0

    def is_terminate2(self):
        qpos = self.data.qpos
        # print('hi', qpos[0])
        return qpos[0] + 2.0 < 0

    def is_boundary_backward_front_for_patrol(self):
        qpos = self.data.qpos
        # print('hi', qpos[0])
        return 0 if qpos[0] + self.x_pos_pivot < 0 else 1

    def is_boundary_backward_rear_for_patrol(self):
        agent_x = self._get_walker2d_pos()
        start = 1.0
        end = -4.0
        # print(agent_x)
        if start >= agent_x and agent_x > end:
            return 0
        elif agent_x > start:
            return 1
        else:
            return -1 # over  

    def is_transition_boundary_for_patrol(self):
        return 0
    
    def is_transition_boundary_rear_for_patrol(self):
        agent_x = self._get_walker2d_pos()
        start = 1.0
        end = -4.0
        if start >= agent_x and agent_x > end:
            return 0
        elif start < agent_x:
            return 1
        else:
            return -1 # over

    def set_curbs_x_randomness_for_irl(self):
        return -1

    def get_curbs_x_randomness(self):
        return -1, -1
