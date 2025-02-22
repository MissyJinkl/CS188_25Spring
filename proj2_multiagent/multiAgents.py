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

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        close_ghost_penalty = 0
        min_ghost_dis = 100000
        for ghost in newGhostStates:
            if ghost.getPosition() == newPos and ghost.scaredTimer == 0:
                close_ghost_penalty = -1000
            elif manhattanDistance(ghost.getPosition(), newPos)<=2 and ghost.scaredTimer == 0:
                close_ghost_penalty = -200
            else: 
                ghost_dis = manhattanDistance(ghost.getPosition(), newPos)
                if min_ghost_dis > ghost_dis:
                    min_ghost_dis = ghost_dis
        #print("min ghost dis is", min_ghost_dis)
        
        food_list = currentGameState.getFood().asList()
        min_food_dis = 100000
        if food_list:
            for food in food_list:
                food_dis = manhattanDistance(food, newPos)
                if food_dis == 0:
                    food_dis = 0.5
                if min_food_dis > food_dis:
                    min_food_dis = food_dis
            # print(min_food_dis)
            return 20.0/min_food_dis - 1.0/min_ghost_dis + close_ghost_penalty
        else:
            return 20 - 1.0/min_ghost_dis + close_ghost_penalty
        

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        return self.minimax(gameState, 0, 0)[1]
        
    def minimax(self, gamestate: GameState, depth, agentIndex):
        if depth == self.depth or gamestate.isWin() or gamestate.isLose():
            return self.evaluationFunction(gamestate), None
        if agentIndex == 0:
            return self.max_value(gamestate, depth, 0)
        else:
            return self.min_value(gamestate, depth, agentIndex)
        
    def max_value(self, gamestate: GameState, depth, agent_index):
        best_score = -float('inf')
        LegalActions = gamestate.getLegalActions(0)
        score = 0
        action = None
        for legal_action in LegalActions:
            next_state = gamestate.generateSuccessor(0, legal_action)
            score = self.minimax(next_state, depth, 1)[0]
            if score > best_score:
                best_score = score
                action = legal_action
        return best_score, action
    
    def min_value(self, gamestate: GameState, depth, agent_index):
        min_score = float('inf')
        LegalActions = gamestate.getLegalActions(agent_index)
        score = 0
        action = None
        for legal_action in LegalActions:
            next_state = gamestate.generateSuccessor(agent_index, legal_action)
            if agent_index == gamestate.getNumAgents()-1:
                score = self.minimax(next_state, depth+1, 0)[0]
            else:
                score = self.minimax(next_state, depth, agent_index+1)[0]
            if score < min_score:
                min_score = score
                action = legal_action
        return min_score, action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, 0, 0, -float('inf'), float('inf'))[1]

    def minimax(self, gamestate, depth, agentIndex, alpha, beta):
        if depth == self.depth or gamestate.isWin() or gamestate.isLose():
            return self.evaluationFunction(gamestate), None
        if agentIndex == 0:
            return self.max_value(gamestate, depth, 0, alpha, beta)
        else:
            return self.min_value(gamestate, depth, agentIndex, alpha, beta)
        
    def max_value(self, gamestate, depth, agentIndex, alpha, beta):
        best_score = -float('inf')
        LegalActions = gamestate.getLegalActions(0)
        score = 0
        action = None
        for legal_action in LegalActions:
            next_state = gamestate.generateSuccessor(0, legal_action)
            score = self.minimax(next_state, depth, 1, alpha, beta)[0]
            if score > best_score:
                best_score = score
                action = legal_action
            alpha = max(alpha, best_score)
            if alpha > beta:
                break
        return best_score, action
    
    def min_value(self, gamestate, depth, agentIndex, alpha, beta):
        min_score = float('inf')
        LegalActions = gamestate.getLegalActions(agentIndex)
        score = 0
        action = None
        for legal_action in LegalActions:
            next_state = gamestate.generateSuccessor(agentIndex, legal_action)
            if agentIndex == gamestate.getNumAgents()-1:
                score = self.minimax(next_state, depth+1, 0, alpha, beta)[0]
            else:
                score = self.minimax(next_state, depth, agentIndex+1, alpha, beta)[0]
            if score < min_score:
                min_score = score
                action = legal_action
            beta = min(beta, min_score)
            if alpha > beta:
                break
        return min_score, action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState, 0, 0)[1]
        
    def expectimax(self, gamestate, depth, agent_index):
        if depth == self.depth or gamestate.isWin() or gamestate.isLose():
            return self.evaluationFunction(gamestate), None
        if agent_index == 0:
            return self.max_value(gamestate, depth, 1)
        else:
            return self.ghost_value(gamestate, depth, agent_index)
        
    def max_value(self, gamestate, depth, agent_index):
        score = 0
        best_score = -float('inf')
        LegalActions = gamestate.getLegalActions(0)
        action = None
        for legal_action in LegalActions:
            next_state = gamestate.generateSuccessor(0, legal_action)
            score = self.expectimax(next_state, depth, 1)[0]
            if score > best_score:
                best_score = score
                action = legal_action
        return best_score, action
    
    def ghost_value(self, gamestate, depth, agent_index):
        total_score = 0.0
        LegalActions = gamestate.getLegalActions(agent_index)
        # if not LegalActions:
        #     return self.evaluationFunction(gamestate), None
        for legal_action in LegalActions:
            next_state = gamestate.generateSuccessor(agent_index, legal_action)
            if agent_index == gamestate.getNumAgents() - 1:
                total_score += self.expectimax(next_state, depth+1, 0)[0]
            else:
                total_score += self.expectimax(next_state, depth, agent_index+1)[0]
        action_num = len(LegalActions)
        return total_score/action_num, None

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Other than trivial evaluation function, 
    1. I consider the distance to the closest ghost and the scared time of the ghost. 
    If the ghost is close and scared, I will try to eat it. If the ghost is close and not scared, I will try to avoid it.
    2. The distance to the closest food and the number of food left. Try to eat the closest food.
    3. The distance to the closest capsule, and the number of capsules left. It's similar to the food.
    4. If the game is win, return inf. If the game is lose, return -inf.
    """
    
    "*** YOUR CODE HERE ***"
    pacman_pos = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    ghost_states = currentGameState.getGhostStates()
    capsules_pos = currentGameState.getCapsules()

    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return -float('inf')
    
    evaluation_score = currentGameState.getScore()

    close_ghost_penalty = 0
    min_ghost_dis = float('inf')
    for ghost in ghost_states:
        if manhattanDistance(ghost.getPosition(), pacman_pos)<=1 and ghost.scaredTimer == 0:
            close_ghost_penalty = -20000
        elif manhattanDistance(ghost.getPosition(), pacman_pos)<=2 and ghost.scaredTimer > 4:
            close_ghost_penalty = 1000    
        ghost_dis = manhattanDistance(ghost.getPosition(), pacman_pos)
        if min_ghost_dis > ghost_dis:
            min_ghost_dis = ghost_dis
    ghost_score = close_ghost_penalty + min_ghost_dis * 0.8
    #print("ghost score is", ghost_score)
        
    min_food_dis = 1000000
    #few_food_bonus = 0
    for food in food_list:
        food_dis = manhattanDistance(food, pacman_pos)
        if min_food_dis > food_dis:
            min_food_dis = food_dis
    food_score = 30.0/min_food_dis - 15*len(food_list)
    #print("food score is", food_score)

            
    if capsules_pos:
        min_capsule_dis = 100000
        for capsule in capsules_pos:
            capsule_dis = manhattanDistance(capsule, pacman_pos)
            if min_capsule_dis > capsule_dis:
                min_capsule_dis = capsule_dis
        return evaluation_score + ghost_score + food_score + 10.0/min_capsule_dis - 25*len(capsules_pos)
    else:
        return evaluation_score + ghost_score + food_score
    

# Abbreviation
better = betterEvaluationFunction
