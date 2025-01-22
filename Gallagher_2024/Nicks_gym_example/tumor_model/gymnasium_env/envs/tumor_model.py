from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    off = 0
    on = 1


class tumor_model(gym.Env):
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None,s = 0.74, r = 0.01, r_s = 0.027, r_r_mult = 1, d_s = 0, d_r = 0, d_D = 1.5, k = 1, term_thresh = 1.2, s0 = 0.74, r0 = 0.01, N0 = 0.75, dt = 0.1):

        # Define model parameters
        self.s = s # Sensative cells
        self.r = r # resistant cells
        self.N = s+r # total number of cells
        self.r_s = r_s # Proliferation rate of sensative cells
        self.r_r = r_r_mult*r_s # Proliferation rate of resistant cells
        self.d_s = d_s # Death rate of sensative cells
        self.d_r = d_r # Death rate of resistant cells
        self.d_D = d_D # Death rate due to drug
        self.k = k # carrying capacity of the tumor
        self.term_thresh = term_thresh # Threshold for tumor size to terminate the episode
        self.s0 = s0 # Initial number of sensative cells
        self.r0 = r0 # Initial number of resistant cells
        self.N0 = N0
        self.dt = dt # Time step



        # Observations are dictionaries with the agent's and the target's location.
        # The observations are total tumor size

        self.observation_space = spaces.Box(low=0, high=self.k, shape=(1,), dtype=float)


    

        # We have 2 actions, corresponding to treatment on or off
        self.action_space = spaces.Discrete(2)

        """
        The following dictionary maps abstract actions from `self.action_space` to treatment on or off
        0 corresponds to "of", 1 to "on" etc.
        """
        self._action_to_direction = {
            Actions.off.value: 0,
            Actions.on.value: 1,

        }

        #assert render_mode is None or render_mode in self.metadata["render_modes"]
        #self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        #self.window = None
        #self.clock = None

    # Helper function to get the observation of the agent (total tumor size)
    def _get_obs(self):
        return np.array([self.N])

    # Helper function to get the info of the agent (sensative and resistant cells)
    def _get_info(self):
        return {
            "sensative": self.s,
            "resistant": self.r,
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # The agent always starts with a tumor of the same size
        self.s = self.s0
        self.r = self.r0
        self.N = self.s0 + self.r0

        # get the observation and info at initial state
        observation = self._get_obs()
        info = self._get_info()

        #if self.render_mode == "human":
        #    self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        treatment = self._action_to_direction[action]

        # Update the tumor size based on LV model
        self.s = self.s + self.dt * (self.r_s*self.s*(1- (self.s + self.r)/self.k) * (1 - self.d_D*treatment) - self.d_s*self.s )
        self.r = self.r + self.dt * ( self.r_r*self.r*(1-(self.s + self.r)/self.k) - self.d_r*self.r )
        self.N = self.s + self.r


        # An episode is done if the agent has reached the target size (1.2*N0)
        terminated = self.N >= self.term_thresh*self.N0
        #print(f"Tumor size: {self.N}, Treatment: {treatment}")   
        # reward for each step below threshold is 0.1
        reward = 0.1 if not terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        #if self.render_mode == "human":
        #    self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        print("Tumor size: ", self.N)
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
