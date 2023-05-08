# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            newStateVals = util.Counter()
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    newStateVals[state] = max([self.getQValue(state,action) for action in self.mdp.getPossibleActions(state)])
            self.values = newStateVals.copy()




    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.


        """
        "*** YOUR CODE HERE ***"
        #newState = self.mdp.getTransitionStatesAndProbs(state, action)[0][0]
        #prob = self.mdp.getTransitionStatesAndProbs(state, action)[0][1]
        vstar = 0


        #    newnewState = self.mdp.getTransitionStatesAndProbs(newState, action)[0][0]
        #    newnewprob = self.mdp.getTransitionStatesAndProbs(newState, action)[0][1]
        #    print (newnewState)
        #    vstar += newnewprob * self.getValue(newnewState)
        #print(vstar)
        #reward = self.mdp.getReward(state, action, newState)
        #print ("probs: ", prob)
        #print ("trans states: ", self.mdp.getTransitionStatesAndProbs(state, action))

        #Qvalue = prob * (reward + self.discount * self.getValue(newState))
        #print ("Q Value: ", reward)




        Qvalue = 0
        for newState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            vstar = self.getValue(newState)
            reward = self.mdp.getReward(state, action, newState)
            Qvalue += prob * (reward + self.discount * vstar)
        #print (Qvalue)
        return Qvalue


    #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.



        """
        "*** YOUR CODE HERE ***"

        if not self.mdp.getPossibleActions(state):
            return None
        else:
            a = util.Counter()
            for action in self.mdp.getPossibleActions(state):
                a[action] = self.computeQValueFromValues(state, action)

            return max(a, key=a.get)



    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"


        states = self.mdp.getStates()

        for i in range(self.iterations):
            index = i % len(states)
            indstate = states[index]
            if not self.mdp.isTerminal(indstate) and self.mdp.getPossibleActions(indstate):
                self.values[indstate] = max([self.getQValue(indstate,action) for action in self.mdp.getPossibleActions(indstate)])



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"


        predecessors = {}
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if not self.mdp.isTerminal(nextState):
                        if nextState in predecessors:
                            predecessors[nextState].add(state)
                        else:
                            predecessors[nextState] = set([state])


        priorityquene = util.PriorityQueue()
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                svalue = self.getValue(s)
                #print (svalue)
                #print ('all Q ', [self.getQValue(s, action) for action in self.mdp.getPossibleActions(s)])
                highestQ = max([self.getQValue(s, action) for action in self.mdp.getPossibleActions(s)])
                diff = abs(highestQ - svalue)
                #print ('diff ', diff)
                #print ('highest Q ', highestQ)

                priorityquene.push(s, -diff)

        for _ in range(self.iterations):
            if priorityquene.isEmpty():
                return #no return
            s = priorityquene.pop()
            #print (s)
            if not self.mdp.isTerminal(s):
                self.values[s] = max([self.getQValue(s, action) for action in self.mdp.getPossibleActions(s)])
            #print ('Q ', highestQ)
            for p in predecessors:
                pvalue = self.getValue(p)
                highestQ2 = max([self.getQValue(p, action) for action in self.mdp.getPossibleActions(p)])
                diff = abs(highestQ2 - pvalue)
                if diff > self.theta:
                    priorityquene.update(p, -diff)

