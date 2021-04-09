import gym
import ballbeam_gym
import numpy as np
from matplotlib import pyplot as plt

kp = 2.75
ki = 0.001
kd = 2.25
max_timesteps = 100

kwargs = {'timestep':0.05, 
        'setpoint':0.0,
        'beam_length':1.0, 
        'max_angle':0.5, 
        'init_velocity':0.5}

env = gym.make('BallBeamSetpoint-v0', **kwargs)

for ep in range(3):

    ballPos = []
    beamAngle = [] 

    previous_error = 0
    integral = 0
    derivative = 0

    observation = env.reset()

    for t in range(1, max_timesteps):
        
        env.render()

        error = env.bb.x - env.setpoint
        
        integral = integral + error
        derivative = error - previous_error
        previous_error = error
        
        action = kp * error + ki * integral + kd * derivative
        action = np.tanh(action) * 1.5

        observtaion, reward, done, info = env.step(action)
        
        ballPos.append(env.bb.x)
        beamAngle.append(env.bb.theta)

        if done:
            print("Episode {} finished in {} timesteps".format(ep + 1, t+1))
            plt.subplot(2, 1, 1)
            plt.plot(ballPos, label = 'Position of Ball')
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(beamAngle, label = 'Angle of Beam')
            plt.legend()
            plt.show()
            break
    
    env.reset()
        
