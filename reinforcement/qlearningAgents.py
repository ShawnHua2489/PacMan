# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"

        self.seenqvals = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #print ('seenqvals ', self.seenqvals[(state, action)])

        if (state,action) not in self.seenqvals:
            return 0
        #print ('getQValue seenqvals ', self.seenqvals[(state, action)])
        return self.seenqvals[(state, action)]




    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        if not self.getLegalActions(state):
            return 0

        #print ('Q value for each action ', [self.getQValue(state, action)for action in self.getLegalActions(state)])
        #print ('legal actions ', self.getLegalActions(state))
        #print ('all entries in seeqvals ', self.seenqvals)
        #print ('max Q value ', max(self.getQValue(state, action)for action in self.getLegalActions(state)))

        return max(self.getQValue(state, action)for action in self.getLegalActions(state))



    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        if not self.getLegalActions(state): #terminal state
            return None
        else:
            a = util.Counter()
            for action in self.getLegalActions(state):
                a[action] = self.getQValue(state, action)
            #maxval = (max(a.values()))
            #print ([key for key in a.keys if key.value == maxval])
            #return max(a, key=a.get)
        """

        "*** YOUR CODE HERE ***"
        #print ('used computeActionFromQValues')

        actions = self.getLegalActions(state)

        if not actions:
            return None

        best_score, best_actions = None, []

        for action in actions:
            score = self.getQValue(state, action)

            if best_score is None or score > best_score:
                best_actions = [action]
                best_score = score
            elif score == best_score:
                best_actions.append(action)

        return random.choice(best_actions)



    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        #print ('used getAction')
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if not legalActions:
            action = None
        elif util.flipCoin(self.epsilon):
            #print ('flip coin ', util.flipCoin(self.epsilon))
            #explore
            action = random.choice(legalActions)
        else:
            #exploit
            action = self.getPolicy(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf

        if not self.getLegalActions(state): #terminal state
            return None
        else:
            a = util.Counter()
            for nextaction in self.getLegalActions(nextState):
                a[nextaction] = self.getQValue(nextState, nextaction)

        maxQ = max(a, key=a.get)
        sample = reward + self.discount * maxQ
        """
        "*** YOUR CODE HERE ***"

        newqval = (1-self.alpha) * self.getQValue(state, action) + \
            self.alpha * (reward + self.discount * self.getValue(nextState))

        self.seenqvals[(state, action)] = newqval
        #print (self.seenqvals)


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        return sum(self.weights[feature] * value for feature, value in features.items())


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        newqval = (1-self.alpha) * self.getQValue(state, action) + \
                  self.alpha * (reward + self.discount * self.getValue(nextState))

        self.seenqvals[(state, action)] = newqval


        features = self.featExtractor.getFeatures(state, action)

        difference = (reward + self.discount * self.getValue(nextState)) - \
                     self.getQValue(state, action)

        for feature in self.weights:
            self.weights[feature] += self.alpha * difference * features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
