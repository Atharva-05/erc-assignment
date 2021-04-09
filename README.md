# ERC-assignment
Objective: To balance a ball on a beam using Open AI gym’s ballbeam-gym environment<br>
The system uses a PID controller with the distance of the ball from the center of the beam as error

# Files
ball-beam.py : PID Controller code<br>
ball-beam-balanced.jpg : Screenshot of balanced ball<br>
ep-01.jpg/ ep-02.jpg/ ep-03.jpg : Screenshots of graphs for each episode <br><br>

# Images
Balanced ball on a beam:<br>
<img src="https://github.com/Atharva-05/erc-assignment/blob/main/ball-beam-balanced.jpg" width="500" height="300">
<br><br>Episode 01:<br>
<img src="https://github.com/Atharva-05/erc-assignment/blob/main/ep-01.jpg" width="450" height="275">
<br><br>Episode 02:<br>
<img src="https://github.com/Atharva-05/erc-assignment/blob/main/ep-02.jpg" width="450" height="275">
<br><br>Episode 03:<br>
<img src="https://github.com/Atharva-05/erc-assignment/blob/main/ep-03.jpg" width="450" height="275">

# Code

```ruby

import gym
import ballbeam_gym
import numpy as np
from matplotlib import pyplot as plt

#Values of P I D constants
kp = 2.75
ki = 0.001
kd = 2.25

max_timesteps = 100

kwargs = {'timestep':0.05, 
        'setpoint':0.0,
        'beam_length':1.0, 
        'max_angle':0.5, 
        'init_velocity':0.5}

#Initialize the environment with the given parameters
env = gym.make('BallBeamSetpoint-v0', **kwargs)

for ep in range(3):
    
    # Initialize empty arrays to store data points that will be plotted after termination of each episode
    ballPos = []
    beamAngle = [] 

    previous_error = 0
    integral = 0
    derivative = 0

    observation = env.reset()

    for t in range(1, max_timesteps):
        
        env.render()

        # Error is the distance of the ball from the setpoint
        error = env.bb.x - env.setpoint
        
        # PID Controller
        integral = integral + error
        derivative = error - previous_error
        previous_error = error
        
        action = (kp * error) + (ki * integral) + (kd * derivative)


        # tanh function is used as a squishing function on the control value obtained from the PID Controller
        # Increases stability and smoothness of the system
        action = np.tanh(action) * 1.5

        # Pass the action to the env.step() function. Returns the state of the system in the observation variable.
        # Returns True to done variable if conditions for episode termination are met
        observtaion, reward, done, info = env.step(action)
        
        # Store the datapoints in the arrays for plotting graphs
        ballPos.append(env.bb.x)
        beamAngle.append(env.bb.theta)

        if done:
            print("Episode {} finished in {} timesteps".format(ep + 1, t+1))
            # Plot the graphs of Position of the ball and Angle of the beam against timesteps
            env.reset()
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(ballPos, label = 'Position of Ball')
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(beamAngle, label = 'Angle of Beam')
            plt.legend()
            plt.show()
            break

    env.close()
        

```
