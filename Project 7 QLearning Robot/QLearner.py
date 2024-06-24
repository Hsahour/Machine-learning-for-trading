""""""                                        
"""                                        
Template for implementing QLearner  (c) 2015 Tucker Balch                                        
                                        
Copyright 2018, Georgia Institute of Technology (Georgia Tech)                                        
Atlanta, Georgia 30332                                        
All Rights Reserved                                        
                                        
Template code for CS 4646/7646                                        
                                        
Georgia Tech asserts copyright ownership of this template and all derivative                                        
works, including solutions to the projects assigned in this course. Students                                        
and other users of this template code are advised not to share it with others                                        
or to make it available on publicly viewable websites including repositories                                        
such as github and gitlab.  This copyright statement should not be removed                                        
or edited.                                        
                                        
We do grant permission to share solutions privately with non-students such                                        
as potential employers. However, sharing with other current or future                                        
students of CS 7646 is prohibited and subject to being investigated as a                                        
GT honor code violation.                                        
                                        
-----do not edit anything above this line---                                        
                                        
Student Name: Hossein Sahour                                      
GT User ID: hsahour3                                        
GT ID: 903941641                                  
"""                                        
                                        
import numpy as np
import random as rand

class QLearner(object):
    def __init__(self, num_states=100, num_actions=4, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False):
        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.s = 0  
        self.a = 0  
        self.q_table = np.zeros((num_states, num_actions))
        self.model = {} if dyna > 0 else None
        self.experiences = []

    def querysetstate(self, s):
        self.s = s
        action = self.select_action(s)
        self.a = action
        return action

    def query(self, s_prime, r):
        self.q_table[self.s, self.a] = (1 - self.alpha) * self.q_table[self.s, self.a] + \
            self.alpha * (r + self.gamma * np.max(self.q_table[s_prime, :]))
        if self.dyna > 0:
            self.experiences.append((self.s, self.a, s_prime, r))
            self.simulate_dyna()
        self.s = s_prime
        action = self.select_action(s_prime)
        self.a = action
        self.rar *= self.radr
        return action

    def select_action(self, s):
        if rand.random() < self.rar:
            return rand.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_table[s, :])

    def simulate_dyna(self):
        for _ in range(self.dyna):
            idx = rand.randint(0, len(self.experiences) - 1)
            s, a, s_prime, r = self.experiences[idx]
            self.q_table[s, a] += self.alpha * (r + self.gamma * np.max(self.q_table[s_prime, :]) - self.q_table[s, a])

    def author(self):
        return 'hsahour3' 

if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")

