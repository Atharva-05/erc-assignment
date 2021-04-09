import gym
import ballbeam_gym
import numpy as np
from matplotlib import pyplot as plt
import time

#  Values of P I D constants
kp = 2.75
ki = 0.001
kd = 2.25

max_timesteps = 100

kwargs = {'timestep':0.05, 
        'setpoint':0.0,
        'beam_length':1.0, 
        'max_angle':0.5, 
        'init_velocity':0.5}

# Initialize the environment with the given parameters
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
        
        # PID Controller logic
        integral = integral + error
        derivative = error - previous_error
        previous_error = error
        
        action = (kp * error) + (ki * integral) + (kd * derivative)

        # tanh function is used as a squishing function on the control value obtained from the PID Controller
        # The range of tanh is [-1,1] which is suitable for setting the beam angle
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
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(ballPos, label = 'Position of Ball')
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(beamAngle, label = 'Angle of Beam')
            plt.legend()
            plt.show()
            env.reset()
            break

    env.close()
        
