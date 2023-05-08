# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent, Actions


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"


        foodleft = sum(int(j) for i in newFood for j in i)


        if foodleft > 0:
            foodDistance = [manhattanDistance(newPos, (x, y)) for x, row in enumerate(newFood)
                            for y, food in enumerate(row) if food]

            minFoodDistance = min(foodDistance)

        else:
            minFoodDistance = 0

        if newGhostStates:
            ghostDistance = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
            minGhostDistance = min(ghostDistance)

            if minGhostDistance == 0:
                minGhostDistance = -2000
            else:
                minGhostDistance = -6 / minGhostDistance

        else:
            minGhostDistance = 0

        return -2 * minFoodDistance + minGhostDistance - 40 * foodleft



        """
        if newScaredTimes != 0:
            score += 100/manhattanDistance(newGhostStates, newPos) #calculate distance
            score += 10/manhattanDistance(newFood, newPos)

        elif newScaredTimes == 0:
            score += 100/manhattanDistance(newGhostStates, newPos) #calculate distance
            score += 10/manhattanDistance(newFood, newPos)

        return score
        #return successorGameState.getScore()
        """
def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"




        def helper(state, depth, indexagent):
            if indexagent == state.getNumAgents():
                if depth == self.depth:
                    return self.evaluationFunction(state)
                else:
                    return helper(state, depth + 1, 0)
            else:
                actionlist = state.getLegalActions(indexagent)

                if len(actionlist) == 0:
                    return self.evaluationFunction(state)

                next_states = (
                    helper(state.generateSuccessor(indexagent, action), depth, indexagent + 1)
                    for action in actionlist
                )

                return (max if indexagent == 0 else min)(next_states)

        return max(
            gameState.getLegalActions(0), key = lambda x: helper(gameState.generateSuccessor(0, x), 1, 1)
        )

            #if state.isWin() or state.isLose():
            #    return

            #if depth == 0:
            #    return

            #for action in (gameState.getLegalActions(indexagent)):
            #    curState = gameState.generateSuccessor(indexagent, action)


            #    return helper(curState, depth+1, indexagent)


        #return (helper(gameState, 1, i) for i in range(gameState.getNumAgents()))

        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBeta(state, depth, alpha, beta, agent):
            isMax = agent == 0
            nextDepth = depth-1 if isMax else depth
            if nextDepth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            nextAgent = (agent + 1) % state.getNumAgents()
            bestVal = -999999 if isMax else 999999
            bestAction = None
            bestOf = max if isMax else min

            for action in state.getLegalActions(agent):
                successorState = state.generateSuccessor(agent, action)
                valOfAction, _ = alphaBeta(
                    successorState, nextDepth, alpha, beta, nextAgent)
                if bestOf(bestVal, valOfAction) == valOfAction:
                    bestVal, bestAction = valOfAction, action

                if isMax:
                    if bestVal > beta:
                        return bestVal, bestAction
                    alpha = max(alpha, bestVal)
                else:
                    if bestVal < alpha:
                        return bestVal, bestAction
                    beta = min(beta, bestVal)

            return bestVal, bestAction

        _, action = alphaBeta(gameState, self.depth+1, -999999, 999999, self.index)
        return action


        #util.raiseNotDefined()

def mean(lst):
    result = list(lst)

    return sum(result)/len(result)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def helper(state, depth, indexagent):
            if indexagent == state.getNumAgents():
                if depth == self.depth:
                    return self.evaluationFunction(state)
                else:
                    return helper(state, depth + 1, 0)
            else:
                actionlist = state.getLegalActions(indexagent)

                if len(actionlist) == 0:
                    return self.evaluationFunction(state)

                next_states = (
                    helper(state.generateSuccessor(indexagent, action), depth, indexagent + 1)
                    for action in actionlist
                )

                return (max if indexagent == 0 else mean)(next_states)

        return max(
            gameState.getLegalActions(0), key = lambda x: helper(gameState.generateSuccessor(0, x), 1, 1)
        )






def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    The function mainly takes into account the following variables: the distance to the
    closest food, the Manhattan distances to the closest ghost, capsule, and scared ghost,
    and the numbers of food and capsules left. It is an upgraded version of the eval
    function for Q1.
    """
    "*** YOUR CODE HERE ***"



# Abbreviation
better = betterEvaluationFunction
