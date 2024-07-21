import numpy as np

def revise_state(s):
    # Calculate the distance moved forward
    distance_moved = s[0] - s[8]
    
    # Calculate the velocity of the cheetah
    velocity = np.sqrt(s[8]**2 + s[9]**2)
    
    # Calculate the angle of the front tip
    angle_front_tip = s[1]
    
    # Calculate the angular velocity of the front tip
    angular_velocity_front_tip = s[6]
    
    # Calculate the angular velocity of the second rotor
    angular_velocity_second_rotor = s[7]
    
    # Calculate the reward based on the distance moved and the velocity
    reward = distance_moved + 0.1 * velocity
    
    # Create the updated state
    updated_s = np.concatenate((s, np.array([distance_moved, velocity, angle_front_tip, angular_velocity_front_tip, angular_velocity_second_rotor, reward])))
    
    return updated_s

def intrinsic_reward(updated_s):
    # Calculate the intrinsic reward based on the updated state
    intrinsic_reward = updated_s[17] + 0.1 * updated_s[18] + 0.01 * updated_s[19]
    
    return intrinsic_reward

