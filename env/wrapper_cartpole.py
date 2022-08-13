import gym


class EnvCartpole(object):
    def __init__(self, env_name):
        self.env = env = gym.make(env_name)
        self.step = 0
        self.current_step_reward = 0
        self.current_step_termination = False

    # call this function to initialize the environment
    def get_initial_state(self):
        state_0 = self.env.reset()
        return state_0

    def transition(self, state, action):
        next_state, reward, done, _ = self.env.step(action)
        self.step += 1
        self.current_step_reward = reward
        self.current_step_termination = done
        return next_state

    def reward(self, state, action, next_state):
        if self.current_step_termination:
            if self.step != 499:
                additional_reward = -5
            else:
                additional_reward = 5
        else:
            additional_reward = 0
        return self.current_step_reward + additional_reward

    def check_if_terminal(self, state):
        return self.current_step_termination


class EnvCartpolePartial(object):
    def __init__(self, env_name):
        self.env = env = gym.make(env_name)
        self.step = 0
        self.current_step_reward = 0
        self.current_step_termination = False

    # call this function to initialize the environment
    def get_initial_state(self):
        state_0 = self.env.reset()[::2]
        return state_0

    def transition(self, state, action):
        next_state, reward, done, _ = self.env.step(action)
        self.step += 1
        self.current_step_reward = reward
        self.current_step_termination = done
        return next_state[::2]

    def reward(self, state, action, next_state):
        if self.current_step_termination:
            if self.step != 499:
                additional_reward = -5
            elif self.step == 499:
                additional_reward = 5
        else:
            additional_reward = 0
        return self.current_step_reward + additional_reward

    def check_if_terminal(self, state):
        return self.current_step_termination
