import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)


def calculate_state(i):
    return i[0]*5+i[1]

class GridEnvironment():
    def __init__(self):
        self.observation_space = spaces.Discrete(25)
        self.action_space = spaces.Discrete(5)
        self.max_timesteps = 100
        self.state = None
        
    def reset(self):
        self.timestep = 0
        self.reward = 0
        self.agent_pos = [0, 0]
        self.goal_pos= [4,4] #villan position
        self.snake_pos=[3,3],[4,1],[1,1]
        self.pit_pos=[0,2],[1,4]
        self.ladder_pos=[2,1]
        self.environment_height = 5
        self.environment_width = 5
        self.state = np.zeros((5,5))
        self.state[tuple(self.agent_pos)] = 1
        self.state[tuple(self.goal_pos)] = 0.5
        return calculate_state(self.agent_pos)
    
    def step(self, action):
        self.state = np.random.choice(self.observation_space.n)
        self.state = np.random.choice(self.observation_space.n)
        if action == 3:
          self.agent_pos[0] += 1 #right
        if action == 2:
          self.agent_pos[0] -= 1 #left
        if action == 1:
          self.agent_pos[1] += 1 #up
        if action == 0:
          self.agent_pos[1] -= 1 #down
        
        self.reward = -1
        done = False
        self.agent_pos = np.clip(self.agent_pos, 0, 4)
        self.state = np.zeros((5,5))
        self.state[tuple(self.agent_pos)] = 1
        self.state[tuple(self.goal_pos)] = 0.5

        if (self.agent_pos == self.goal_pos).all():
          self.reward += 50
          self.agent_pos = [4,4]
          done = True
        elif any(np.array_equal(self.agent_pos,self.snake_pos[i]) for i in range(len(self.snake_pos))):
          self.reward += -20
          self.agent_pos = [0,0]
          
        elif any(np.array_equal(self.agent_pos,self.pit_pos[i]) for i in range(len(self.pit_pos))):
          self.reward += -20
          self.agent_pos = [0,0]
        
        elif (self.agent_pos == self.ladder_pos).all():
          self.reward += 2
          self.agent_pos = [2,3]
        
        self.timestep += 1
        done = (True if self.timestep >= self.max_timesteps else False) or done
        info = {'agent_pose':calculate_state(self.agent_pos)}
        
        return calculate_state(self.agent_pos), self.reward, done, info
        
    def render(self,mode="human",plot=False):
      fig, ax  = plt.subplots(figsize=(10,10))
      ax.set_xlim(0,5)
      ax.set_ylim(0,5)
      
      def plot_image(plot_pos):
        plot_agent,plot_snake,plot_pit,plot_ladder,plot_goal = \
        False,False,False,False,False

        if np.array_equal(self.agent_pos, plot_pos):
          plot_agent=True
        if any(np.array_equal(self.snake_pos[i], plot_pos) for i in range(len(self.snake_pos))):
          plot_snake=True
        if any(np.array_equal(self.pit_pos[i], plot_pos) for i in range(len(self.pit_pos))):
          plot_pit=True
        if np.array_equal(self.ladder_pos, plot_pos):
          plot_ladder=True
        if np.array_equal(self.goal_pos, plot_pos):
          plot_goal=True

        if plot_agent and \
                all(not item for item in [plot_snake,plot_pit,plot_ladder,plot_goal]):
                agent= AnnotationBbox(OffsetImage(plt.imread('jb.jpg'), zoom=0.40),
                  np.add(plot_pos, [0.5,0.5]), frameon=False)
                ax.add_artist(agent)

        #Image for snake
        elif plot_snake and all(not item for item in 
                    [plot_agent, plot_pit, plot_ladder,plot_goal]):
          snake= AnnotationBbox(OffsetImage(plt.imread("sanke.jpg"), zoom=0.07), 
                                np.add(plot_pos, [0.5, 0.5]), frameon= False)
          ax.add_artist(snake)

        #Image for pit
        elif plot_pit and all(not item for item in [plot_agent, plot_snake, plot_ladder,plot_goal]):
          pit= AnnotationBbox(OffsetImage(plt.imread('PentaPit5.jpg'), zoom=0.23), 
                              np.add(plot_pos, [0.5, 0.5]), frameon= False)
          ax.add_artist(pit)

        #Image for potal
        elif plot_ladder and all(not item for item in 
                    [plot_agent, plot_snake, plot_pit,plot_goal]):

          ladder = AnnotationBbox(OffsetImage(plt.imread('ladder.png'), zoom=0.4),
                    np.add(plot_pos, [0.5, 0.5]), frameon= False)
          ax.add_artist(ladder)

          #image for goal
        elif plot_goal and all(not item for item in 
                  [plot_agent, plot_snake, plot_pit, plot_ladder]):
          goal = AnnotationBbox(OffsetImage(plt.imread('villan.jpg'), zoom=0.1),
                  np.add(plot_pos, [0.5, 0.5]), frameon= False)
          ax.add_artist(goal)

      coordinates_state_mapping_2 = {}

      for j in range(self.environment_height * self.environment_width):
        coordinates_state_mapping_2[j] = np.asarray(
            [j % self.environment_width, int(np.floor(j / self.environment_width))])

      for position in coordinates_state_mapping_2:
        plot_image(coordinates_state_mapping_2[position])

      plt.xticks([0, 1, 2, 3, 4])
      plt.yticks([0, 1, 2, 3, 4])
      plt.grid()
      plot_image(self.agent_pos)
      plt.show()

