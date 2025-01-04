from go_search_problem import GoProblem, GoState, Action
from adversarial_search_problem import GameState
from heuristic_go_problems import *
import random
from abc import ABC, abstractmethod
import numpy as np
import time
from game_runner import run_many
import pickle
import torch
from torch import nn
import os 
import json


MAXIMIZER = 0
MIMIZER = 1

class GameAgent():
    # Interface for Game agents
    @abstractmethod
    def get_move(self, game_state: GameState, time_limit: float) -> Action:
        # Given a state and time limit, return an action
        pass


class RandomAgent(GameAgent):
    # An Agent that makes random moves

    def __init__(self):
        self.search_problem = GoProblem()
        self.move_num = 0

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        get random move for a given state
        """
        # Code for finding length of games
        # if game_state.internal_state.move_number()==0 or game_state.internal_state.move_number()==1:
        #     print("game lasted ",self.move_num," moves")
        #     self.move_num = 0
        # else:
        #     self.move_num = game_state.internal_state.move_number()
        actions = self.search_problem.get_available_actions(game_state)
        return random.choice(actions)

    def __str__(self):
        return "RandomAgent"


class GreedyAgent(GameAgent):
    def __init__(self, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.search_problem = search_problem
        self.move_num = 0

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        get move of agent for given game state.
        Greedy agent looks one step ahead with the provided heuristic and chooses the best available action
        (Greedy agent does not consider remaining time)

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        """
    
        # Create new GoSearchProblem with provided heuristic
        search_problem = self.search_problem

        # Player 0 is maximizing
        if game_state.player_to_move() == MAXIMIZER:
            best_value = -float('inf')
        else:
            best_value = float('inf')
        best_action = None

        # Get Available actions
        actions = game_state.legal_actions()

        # Compare heuristic of every reachable next state
        good_actions = []
        for action in actions:
            new_state = search_problem.transition(state = game_state, action = action)
            value = search_problem.heuristic(new_state, new_state.player_to_move())
            if game_state.player_to_move() == MAXIMIZER:
                if value == best_value:
                    good_actions.append(action)
                if value > best_value:
                    good_actions = []
                    good_actions.append(action)
                    best_value = value

            else:
                if value == best_value:
                    good_actions.append(action)

                if value < best_value:
                    good_actions = []
                    good_actions.append(action)
                    best_value = value

        # Return best available action
        return random.choice(good_actions)

    def __str__(self):
        """
        Description of agent (Greedy + heuristic/search problem used)
        """
        return "GreedyAgent + " + str(self.search_problem)


class MinimaxAgent(GameAgent):
    def __init__(self, depth=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.depth = depth
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using minimax algorithm


        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # implement get_move method of MinimaxAgent
        best_action = None
        curr_depth = self.depth
        #if it is the maximizers turn, move for the maximzer
        if(game_state.player_to_move() == 0):
            x,y = self.max_val(game_state,curr_depth)
        #if it is the minimzers turn, move for the minimzer
        else:
            print(game_state)
            print("this is min value" + str(self.min_val(game_state,curr_depth)))
            x,y = self.min_val(game_state,curr_depth)
            print(x)
            print(y)
        best_action = x
        return best_action

    def __str__(self):
        return f"MinimaxAgent w/ depth {self.depth} + " + str(self.search_problem)


    def max_val(self,curr_state: GameState,  current_depth=float('inf')):
        best_action = None
        #if the current state is terminal, use the reward value
        if(self.search_problem.is_terminal_state(curr_state)):
            return best_action, self.search_problem.evaluate_terminal(curr_state) * float('inf')
        #if the current depth is zero, use the heuristic value
        if(current_depth==0):
            return best_action, self.search_problem.heuristic(curr_state,curr_state.player_to_move())
        z = -float('inf')
        #search all available actions
        for curr_action in self.search_problem.get_available_actions(curr_state):
            #find the substate that results from the action made
            sub_state = self.search_problem.transition(curr_state,curr_action)
            #recurse and find the values at the base depth
            _,value = self.min_val(sub_state,current_depth-1)
            #if the returned value is greater then the max value, reset the best action to that action
            if(value>=z):
                z = value
                best_action = curr_action
        return best_action,z


    def min_val(self,curr_state: GameState,  current_depth):
        best_action = None
        #if the current state is terminal, use the reward value
        if(self.search_problem.is_terminal_state(curr_state)):
            return best_action, self.search_problem.evaluate_terminal(curr_state) * float('inf')
        #if the current depth is zero, use the heuristic value
        if(current_depth==0):
            return best_action,self.search_problem.heuristic(curr_state,curr_state.player_to_move())
        z = float('inf')
        #search all available actions
        for curr_action in self.search_problem.get_available_actions(curr_state):
            #find the substate that results from the action made
            sub_state = self.search_problem.transition(curr_state,curr_action)
            #recurse and find the values at the base depth
            _,value = self.max_val(sub_state,current_depth-1)
            #if the returned value is lesser then the max value, reset the best action to that action
            if(value<=z):
                z = value
                best_action = curr_action
        return best_action,z



class AlphaBetaAgent(GameAgent):
    def __init__(self, depth=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.depth = depth
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using alpha-beta algorithm
        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # implement get_move algorithm of AlphaBeta Agent
        best_action = None
        cutoff_depth = self.depth
        #if it is the maximizers turn, move for the maximzer with the max_val_ab function
        if(game_state.player_to_move() == 0):
            x,y = self.max_val_ab(game_state,-float('inf'), float('inf'),cutoff_depth)
        #if it is the minimzers turn, move for the minimzer with the min_val_ab function
        else:
            x,y = self.min_val_ab(game_state,-float('inf'), float('inf'), cutoff_depth)
        best_action = x
        return best_action

    def max_val_ab(self, curr_state: GameState, alpha: int, beta: int,  cutoff_depth=float('inf')):
        best_action = curr_state.size**2
        #if the current state is terminal, use the reward value
        if(self.search_problem.is_terminal_state(curr_state)):
            return best_action ,self.search_problem.evaluate_terminal(curr_state)*float('inf')
        #if the current depth is zero, use the heuristic value 
        if(cutoff_depth==0):
            return best_action,  self.search_problem.heuristic(curr_state,curr_state.player_to_move())

        max_value = -float('inf')
        #search all available actions
        for curr_action in self.search_problem.get_available_actions(curr_state):
            #find the substate that results from the action made
            sub_state = self.search_problem.transition(curr_state,curr_action)
            #Recurse and find the values at the base depth
            _,value = self.min_val_ab(sub_state,alpha,beta, cutoff_depth-1)

            #find the max value and keep track of the best action associated with that value
            if(value>=max_value):
                max_value = value
                best_action = curr_action
                alpha = value

            #if the max value is greater then beta, just return the best action now, this is what makes ab pruning ab pruning
            if(max_value > beta):
                return best_action ,max_value
        return best_action ,max_value


    def min_val_ab(self, curr_state: GameState, alpha: int, beta: int, cutoff_depth=float('inf')):
        best_action = curr_state.size**2

        #if the current state is terminal, use the reward value
        if(self.search_problem.is_terminal_state(curr_state)):
            return best_action ,self.search_problem.evaluate_terminal(curr_state)*float('inf')
        #if the current depth is zero, use the heuristic value 
        if(cutoff_depth==0):
            return best_action,  self.search_problem.heuristic(curr_state,curr_state.player_to_move())
        min_value = float('inf')
        #search all available actions
        for curr_action in self.search_problem.get_available_actions(curr_state):
            #find the substate that results from the action made
            sub_state = self.search_problem.transition(curr_state,curr_action)
            #Recurse and find the values at the base depth
            _,value = self.max_val_ab(sub_state,alpha,beta, cutoff_depth-1)
        #find the max value and keep track of the best action associated with that value
            #find the max value and keep track of the best action associated with that value
            if(value<=min_value):
                min_value = value
                best_action = curr_action
                beta = value

            #if the max value is greater then beta, just return the best action now, this is what makes ab pruning ab pruning
            if(min_value < alpha):
                return best_action ,min_value
        return best_action ,min_value

    def __str__(self):
        return f"AlphaBeta w/ depth {self.depth} + " + str(self.search_problem)


class IterativeDeepeningAgent(GameAgent):
    def __init__(self, cutoff_time=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.cutoff_time = cutoff_time
        self.search_problem = search_problem
        self.moves = 0

    def get_move(self, game_state:GoState, time_limit):
        """
        Get move of agent for given game state using iterative deepening algorithm (+ alpha-beta).
        Iterative deepening is a search algorithm that repeatedly searches for a solution to a problem,
        increasing the depth of the search with each iteration.
        The advantage of iterative deepening is that you can stop the search based on the time limit, rather than depth.
        The recommended approach is to modify your implementation of Alpha-beta to stop when the time limit is reached
        and run IDS on that modified version.
        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """

        best_action = None

        cutoff_depth =1
        start_time = time.time()
        the_time = min(time_limit,self.cutoff_time)
        #if it is the maximizers turn, move for the maximzer with the max_val_ab function
        while(time.time()-start_time < the_time*.95):
            if(game_state.player_to_move() == 0):
                x,y = self.max_val_ab(game_state,-float('inf'), float('inf'),cutoff_depth,start_time, the_time)
            #if it is the minimzers turn, move for the minimzer with the min_val_ab function
            else:
                x,y = self.min_val_ab(game_state,-float('inf'), float('inf'), cutoff_depth,start_time, the_time)
            if(x != None):
                best_action = x


            cutoff_depth+= 1

        #print("cutoff depth is " , cutoff_depth) # print cutoff depth that was reached
        return best_action


    def max_val_ab(self, curr_state: GameState, alpha: int, beta: int,  cutoff_depth=float('inf'),start_time=0.0,time_limit=0.0):
        best_action = None
        #if the current state is terminal, use the reward value
        if(curr_state.is_terminal_state()):
            return best_action ,self.search_problem.evaluate_terminal(curr_state)*float('inf')
        #if the current depth is zero, use the heuristic value
        if(cutoff_depth==0):
            return best_action,  self.search_problem.heuristic(curr_state,curr_state.player_to_move())
        if(time.time()-start_time >= time_limit):
            return best_action, self.search_problem.heuristic(curr_state,curr_state.player_to_move())

        max_value = -float('inf')
        #search all available actions
        for curr_action in curr_state.legal_actions():
            if(time.time()-start_time >= time_limit):
                return best_action, max_value
            #find the substate that results from the action made
            if(curr_action!= None):
                sub_state = self.search_problem.transition(curr_state,curr_action)
            else:
                sub_state = curr_state
            #Recurse and find the values at the base depth
            _,value = self.min_val_ab(sub_state,alpha,beta, cutoff_depth-1,start_time,time_limit)

            #find the max value and keep track of the best action associated with that value
            if(value>=max_value):
                max_value = value
                best_action = curr_action
                alpha = value

            #if the max value is greater then beta, just return the best action now, this is what makes ab pruning ab pruning
            if(max_value > beta):
                return best_action ,max_value
        return best_action ,max_value

    def min_val_ab(self, curr_state: GameState, alpha: int, beta: int, cutoff_depth=float('inf'),start_time=0.0,time_limit=0.0):
        best_action = None 

        #if the current state is terminal, use the reward value
        if(curr_state.is_terminal_state()):
            return best_action ,self.search_problem.evaluate_terminal(curr_state)*float('inf')
        #if the current depth is zero, use the heuristic value
        if(cutoff_depth==0):
            return best_action,  self.search_problem.heuristic(curr_state,curr_state.player_to_move())
        if(time.time()-start_time >= time_limit):
            return best_action, self.search_problem.heuristic(curr_state,curr_state.player_to_move())
        min_value = float('inf')
        #search all available actions
        for curr_action in curr_state.legal_actions():
            if(time.time()-start_time >= time_limit):
                return best_action, min_value
            #find the substate that results from the action made

            if(curr_action!= None):
                sub_state = self.search_problem.transition(curr_state,curr_action)
            else:
                sub_state = curr_state
            #Recurse and find the values at the base depth
            _,value = self.max_val_ab(sub_state,alpha,beta, cutoff_depth-1,start_time,time_limit)
            #find the max value and keep track of the best action associated with that value
            #find the max value and keep track of the best action associated with that value
            if(value<=min_value):
                min_value = value
                best_action = curr_action
                beta = value

            #if the max value is greater then beta, just return the best action now, this is what makes ab pruning ab pruning
            if(min_value < alpha):
                return best_action ,min_value

        return best_action ,min_value


    def __str__(self):
        return f"IterativeDeepneing + " + str(self.search_problem)



class MCTSNode:
    def __init__(self, state, parent=None, children=None, action=None):
        # GameState for Node
        self.state = state

        # Parent (MCTSNode)
        self.parent = parent

        # Children List of MCTSNodes
        if children is None:
            children = []
        self.children = children

        # Number of times this node has been visited in tree search
        self.visits = 0

        # Value of node (number of times simulations from children results in black win)
        self.value = 0

        # Action that led to this node
        self.action = action

    def __hash__(self):
        """
        Hash function for MCTSNode is hash of state
        """
        return hash(self.state)


class MCTSAgent(GameAgent):
    def __init__(self, c=np.sqrt(2),game_size = 5):
        """
        Args:
            c (float): exploration constant of UCT algorithm
        """
        super().__init__()
        self.c = c

        self.root = None
        self.agent_2 = create_value_agent_from_model()
        self.agent = RandomAgent()
        
        # open opening book file if it exists, else make a new opening book
        self.path = "opening_book"+str(game_size)+".json"
        if os.path.exists(self.path):
            self.opening_book = self.read_opening_book()
        else:
            self.opening_book = {}
        self.build_opening_book = False # gets modified by build_opening_book() in game_runner.py
        self.opening_book_depth = 1 

        # Initialize Search problem
        self.search_problem = GoProblem()

        self.move_num = 0

    def read_opening_book(self):
        """
        Reads the opening book json file and sets self.opening_book
        """
        print("reading opening book")
        with open(self.path, 'r') as file:
            self.opening_book = json.load(file)
        return self.opening_book

    def save_opening_book(self,depth:int):
        """
        Uses bfs to traverse the mcts tree and add states/actions to the opening book until depth reaches depth.
        Only adds states/actions to the opening book dict if the player to move in the state is the same as the
        agent player to move
        """
        # first check if needs to update existing opening book file
        state_to_actions = self.opening_book

        player = self.root.state.player_to_move()
        q = [self.root]
        while len(q) > 0:
            curr_node = q.pop()
            # save and return when reached depth
            if curr_node.state.internal_state.move_number() > depth:
                print("saving opening book")
                self.opening_book = state_to_actions
                # Convert dictionary to JSON string
                json_data = json.dumps(state_to_actions)

                # Write JSON string to a file
                with open(self.path, "w") as f:
                    f.write(json_data) 
                return

            # add state action to dict
            max_visits = 0
            best_child = None
            for child in curr_node.children:
                q.append(child)
                if child.visits > max_visits:
                    best_child = child 
                    max_visits = child.visits
            if player == curr_node.state.player_to_move():
                if best_child is not None:
                    state_to_actions[str(curr_node.state.get_board())] = best_child.action # board array is not hashable so have to convert to str
    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using MCTS algorithm

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # # code for printing how long games last
        # if game_state.internal_state.move_number()==0 or game_state.internal_state.move_number()==1:
        #     print("game lasted ",self.move_num," moves")
        #     self.move_num = 0
        # else:
        #     self.move_num = game_state.internal_state.move_number()


        # if state is in opening book, get action
        if str(game_state.get_board) in self.opening_book.keys():
            return self.opening_book[str(game_state.get_board)]
        #tracker variable
        changed = False
        #this if statement checks if the board is within one of the 1st two moves of the game 
        #check number of peices != 0 or 1
        if(len(game_state.legal_actions()) != len(game_state.get_board()[0])**2 
            and len(game_state.legal_actions()) - 1 != len(game_state.get_board()[0])**2 ) and self.root is not None and game_state.is_terminal_state() is False:
                #we then find the child node of the root that corresponds to the current game state
                for child in self.root.children:
                    if np.array_equal( game_state.get_board(),child.state.get_board()):
                        #once this child is found we set it to the root, cut off the unnecessary parts of the tree 
                        # (which saves time during back prop) and set changed to true
                        self.root = child
                        self.root.parent = None
                        changed = True
                #if somehow we have not cut the tree and are past the first move in the game, then we will just make a new tree
                #this should theoretically never happen, but we figured it was good to have to make sure our
                #bot would not just love if it didn't know what it was doing
                if(changed is False):
                    self.root = MCTSNode(game_state)
        else:
            #otherwise we make a new MCTSNode
            self.root = MCTSNode(game_state)

        start_time = time.time()
        node = self.root
        while(time.time()-start_time < time_limit*.9-0.5):

                #select
                leaf = self.select(node)
                #expand
                child= self.expand(leaf)
                #simulate
                result = self.simulate(child)
                #backprop
                self.back_prop(result,child)      
        # save opening book if done with building tree
        if self.build_opening_book:
            self.save_opening_book(self.opening_book_depth)
            return
        
        most_visits = 0
        best_action = None
        best_child = None
        #This for loop chooses the best child(the one with the most visits) and makes the corresponding action
        for child in node.children:
            if child.visits > most_visits and child.action in node.state.legal_actions():
                most_visits = child.visits
                best_action = child.action
                best_child = child

        self.root = best_child
        return best_action

    def select(self,node):
        curr_node = node
        #traverse tree until we reach a terminal state
        while not curr_node.state.is_terminal_state():
            #initialize necessary variables
            best_value = -float('inf')
            big_n = curr_node.visits
            best_node = None
            #if there are no legal actions or no children, return this node
            if len(curr_node.state.legal_actions()) == 0 or len(curr_node.children) == 0:
                return curr_node
            #if the number of legal actions is greater the the number of children retturn this node
            if len(curr_node.state.legal_actions()) > len(curr_node.children):
                return curr_node
            #loop through the children 
            for child in curr_node.children:
                if child.visits == 0:
                    value = float('inf')  # Encourage exploration of unvisited nodes
                else:
                    value = (child.value / child.visits) + self.c * np.sqrt(np.log(big_n) / child.visits) #calculates UCT score
                if value > best_value:
                    best_value = value
                    best_node = child
            curr_node = best_node
        #return the best node
        return curr_node



    def expand(self,leaf):
        #if the leaf is a terminal state, just return
        if leaf.state.is_terminal_state():
            return leaf
        state = leaf.state

        actions = state.legal_actions()
        children = leaf.children
        already_explored = []
        #make a list of the actions that have already been explored
        for child in children:
            already_explored.append(child.action)
        #from this get a list of the availible actions
        availible_actions = list(set(actions)-set(already_explored))
        action = random.choice(availible_actions)

        #make a new node for these actions
        child_state = self.search_problem.transition(state,action )
        child_node = MCTSNode(child_state,parent = leaf,action=action)
        leaf.children.append(child_node)
        #return the exanded child
        return child_node

    def simulate(self,child):
        state = child.state
        #simulate random movees until the game ends
        while not state.is_terminal_state():
            action = self.agent.get_move(state,0.01)
            state = self.search_problem.transition(state,action)
        result = self.search_problem.evaluate_terminal(state)
        #return the results
        return result


    def back_prop(self,result, child):

        curr_node = child
        #while children have parents 
        while curr_node is not None:
            #increment visits
            curr_node.visits = curr_node.visits + 1
            #increment values depending on the result and the player
            if result == -1 and  curr_node.state.player_to_move() == 0:
                curr_node.value = curr_node.value +1
            elif result ==1  and curr_node.state.player_to_move() == 1:
                curr_node.value = curr_node.value +1
            #send the information back up the tree
            curr_node = curr_node.parent
    def __str__(self):
        return "MCTS"

class MCTSOriginalAgent(GameAgent):
    def __init__(self, c=np.sqrt(2)):
        """
        Args:
            c (float): exploration constant of UCT algorithm
        """
        super().__init__()
        self.c = c


        self.search_problem = GoProblem()


    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using MCTS algorithm

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """

        start_time = time.time()
        node = MCTSNode(game_state)
        while(time.time()-start_time < time_limit*.9-0.5):
            
                leaf = self.select(node)
        
                child= self.expand(leaf)
        
                result = self.simulate(child)
        
                self.back_prop(result,child)
                # i +=1
        most_visits = 0
        best_action = None

        for child in node.children:
            if child.visits > most_visits and child.action in node.state.legal_actions():
                most_visits = child.visits
                best_action = child.action
        if(best_action is None):
            return game_state.size**2
        return best_action



    def select(self,node):
        #selects a node that has the best policy, the one that should be explored next
        #while we have not selected a node and have not reached a leaf, keep choosing a better node
        #continue choosing nodes that have the highest uct score
        #for each action in all the possible actions at a certain state
        #calculate the UCT score - N is per layer
        curr_node = node
        #gotta check time limit in here
        while not curr_node.state.is_terminal_state():

            best_value = -float('inf')
            big_n = curr_node.visits
            best_node = None
            if len(curr_node.state.legal_actions()) == 0 or len(curr_node.children) == 0:
                return curr_node
            if len(curr_node.state.legal_actions()) > len(curr_node.children):
                return curr_node

            for child in curr_node.children:
                if child.visits == 0:
                    value = float('inf')  # Encourage exploration of unvisited nodes
                else:
                    value = (child.value / child.visits) + self.c * np.sqrt(np.log(big_n) / child.visits)
                if value > best_value:
                    best_value = value
                    best_node = child
            curr_node = best_node
        return curr_node



    def expand(self,leaf):
        #take the child of the current state and expand their children so that you can simulate them later
        if leaf.state.is_terminal_state():
            return leaf
        state = leaf.state

        actions = state.legal_actions()
        children = leaf.children
        already_explored = []
        for child in children:
            already_explored.append(child.action)

        availible_actions = list(set(actions)-set(already_explored))
        action = random.choice(availible_actions)
        if not state.is_terminal_state():
            child_state = self.search_problem.transition(state,action )
            child_node = MCTSNode(child_state,parent = leaf,action=action)
            leaf.children.append(child_node)
            return child_node

        #make the new MCTS nodes for the children and add them to the current leafs children
    def simulate(self,child):
        #simulate it completely randomly until terminal state, which will return whether you will run or not
        #rollout policy is completely random
        state = child.state
        while not state.is_terminal_state():
            actions = state.legal_actions()
            random_action = random.choice(actions)
            state = self.search_problem.transition(state,random_action)
        result = self.search_problem.evaluate_terminal(state)
        #return results
        return result


    def back_prop(self,result, child):
        curr_node = child

        while curr_node is not None:

            curr_node.visits = curr_node.visits + 1

            if result == -1 and  curr_node.state.player_to_move() == 0:
                curr_node.value = curr_node.value +1

            elif result ==1  and curr_node.state.player_to_move() == 1:
                curr_node.value = curr_node.value +1

            curr_node = curr_node.parent


    def __str__(self):
        return "MCTS Original Agent"


class FinalAgent(GameAgent):
    def __init__(self,game_size:int):
        super().__init__()
        #make a new MCTS agent
        self.mcts = MCTSAgent(game_size=game_size)
        #Make an ITDSAgent that uses the value function from the neural net we made in part 2 of the project
    
        
        self.moves = 0
        #initialize the cutoff based off the current game size
        if game_size == 5:
            self.mcts_cutoff = 44
            heuristic_search_problem = GoProblemLearnedHeuristic(create_value_agent_from_model())
            self.ids = IterativeDeepeningAgent(1,heuristic_search_problem)
        else:
            self.mcts_cutoff = 155
            self.ids = IterativeDeepeningAgent(1)
    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        is_starting_state = len(game_state.legal_actions())==game_state.size**2+1 or len(game_state.legal_actions())==(game_state.size**2)
        if is_starting_state:
            self.moves = 0
        else:
            self.moves+=1
        #if we think the game is close to ending use IDS
        if self.moves>=self.mcts_cutoff:
            return self.ids.get_move(game_state,time_limit)
        #IF we think the game is not close to ending use MCTS
        else:
            return self.mcts.get_move(game_state,time_limit)
    def __str__(self):
        return "FINAL AGENT"

def get_final_agent_5x5():
    #make a new final agent
    return FinalAgent(game_size=5)
def get_final_agent_9x9():
    #make a new final agent
    return FinalAgent(game_size=9)


def load_model(path: str, model):
    """
    Load model from file

    Note: you still need to provide a model (with the same architecture as the saved model))

    Input:
        path: path to load model from
        model: Pytorch model to load
    Output:
        model: Pytorch model loaded from file
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()

        
        output_size = 1
        
        self.in_layer = nn.Linear(input_size,100)

        self.layer_1 = nn.Linear(100,90)

        self.layer_2 = nn.Linear(90,70)

        self.layer_3 = nn.Linear(70,50)

        self.out_layer = nn.Linear(50,1)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Run forward pass of network

        Input:
            x: input to network
        Output:
            output of network
        """
        a = self.in_layer(x)
        b = self.tanh(a)

        c = self.layer_1(b)
        d = self.relu(c)

        e = self.layer_2(d)
        f = self.relu(e)

        g = self.layer_3(f)
        h = self.relu(g)

        i = self.out_layer(h)
        j = self.sigmoid(i)
        return (j)

class GoProblemLearnedHeuristic(GoProblem):
    """"This is the go problem learned heuristic"""
    def __init__(self, model=None, state=None):
        super().__init__(state=state)
        self.model = model

    def __call__(self, model=None):
        """
        Use the model to compute a heuristic value for a given state.
        """
        return self

    def encoding(self, state):
        """
        Get encoding of state (convert state to features)
        Note, this may call get_features() from Task 1. 

        Input:
            state: GoState to encode into a fixed size list of features
        Output:
            features: list of features
        """
        #get encoding of state (convert state to features)
        features = get_features(state)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        return features_tensor

    def heuristic(self, state, player_index):
        """
        Return heuristic (value) of current state

        Input:
            state: GoState to encode into a fixed size list of features
            player_index: index of player to evaluate heuristic for
        Output:
            value: heuristic (value) of current state
        """
        value = 0
        features = self.encoding(state)
        value = self.model.forward(features)
        # Note, your agent may perform better if you force it not to pass
        # (i.e., don't select action #25 on a 5x5 board unless necessary)
        return value

    def __str__(self) -> str:
        return "Learned Heuristic"


def create_value_agent_from_model():
    """
    Create agent object from saved model. This (or other methods like this) will be how your agents will be created in gradescope and in the final tournament.
    """

    model_path = "value_model.pt"
    # Update number of features for your own encoding size
    feature_size = 80
    model = load_model(model_path, ValueNetwork(feature_size))
    heuristic_search_problem = GoProblemLearnedHeuristic(model)

    # Try with other heuristic agents (IDS/AB/Minimax)
    learned_agent = GreedyAgent(heuristic_search_problem)

    return learned_agent

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, board_size=5):
        super(PolicyNetwork, self).__init__()




        #  Add more layers, non-linear functions, etc.
        #self.linear = nn.Linear(input_size, output_size)
        self.in_layer = nn.Linear(input_size,128)

        self.layer_1 = nn.Linear(128,50)

        self.layer_2 = nn.Linear(50,32)

        self.layer_3 = nn.Linear(32,26)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Run forward pass of network

        Input:
            x: input to network
        Output:
            output of network
        """
        #Update as more layers are added

        a = self.in_layer(x)
        b = self.relu(a)

        c = self.layer_1(b)
        d = self.relu(c)

        e = self.layer_2(d)
        f = self.relu(e)

        g = self.layer_3(f)

        return g

def get_features(game_state: GoState):
    """
    Map a game state to a list of features.

    Some useful functions from game_state include:
        game_state.size: size of the board
        get_pieces_coordinates(player_index): get coordinates of all pieces of a player (0 or 1)
        get_pieces_array(player_index): get a 2D array of pieces of a player (0 or 1)

        get_board(): get a 2D array of the board with 4 channels (player 0, player 1, empty, and player to move). 4 channels means the array will be of size 4 x n x n

        Descriptions of these methods can be found in the GoState

    Input:
        game_state: GoState to encode into a fixed size list of features
    Output:
        features: list of features
    """
    board_size = game_state.size

    #Encode game_state into a list of features
    features = []
    #Get the current state
    #get board
    #
    board = game_state.get_board()

    player_0 = list(board[0].flatten())
    player_1 = list(board[1].flatten())
    empty = list(board[2].flatten())

    player_to_move = (board[3][0][0])

    #could consider adding extra features
    features = player_0+ player_1 + empty
    features.append(player_to_move)

    player_0_count = sum(player_0)
    features.append(player_0_count)
    player_1_count = sum(player_1)
    features.append(player_1_count)
    empty_count = sum(empty)
    features.append(empty_count)

    if player_to_move == 0:
        count_diff = player_0_count - player_1_count
    else:
        count_diff = player_1_count - player_0_count
    features.append(count_diff)

    return features
class PolicyAgent(GameAgent):
    def __init__(self, search_problem, model_path, board_size=5):
        super().__init__()
        self.search_problem = search_problem
        self.model = load_model(model_path, PolicyNetwork(80,5))
        self.board_size = board_size

    def encoding(self, state):
        #get encoding of state (convert state to features)
        features = get_features(state)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        return features_tensor

    def get_move(self, game_state, time_limit=1):
        """
        Get best action for current state using self.model

        Input:
            game_state: current state of the game
            time_limit: time limit for search (This won't be used in this agent)
        Output:
            action: best action to take
        """

        # Select LEGAL Best Action predicted by model
        # The top prediction of your model may not be a legal move!
        features = self.encoding(game_state)
        actions = self.model(features)
        actions = torch.argsort(actions,descending=True)
        #check to see if the action is a legal action at the current stateac

        for action in actions:
            if(action.item() in self.search_problem.get_available_actions(game_state)):
                if(action.item() != self.board_size**2 ):
                    return action.item()

        return game_state.size**2
        # Note, you may want to force your policy not to pass their turn unless necessary
        #assert action in self.search_problem.get_available_actions(game_state)


    def __str__(self) -> str:
        return "Policy Agent"

def create_policy_agent_from_model():
    """
    Create agent object from saved model. This (or other methods like this) will be how your agents will be created in gradescope and in the final tournament.    
    """

    model_path = "policy_model.pt"
    agent = PolicyAgent(GoProblem(size=5), model_path)
    return agent


def main():
    agent1 = MCTSAgent()
    agent2 = GreedyAgent()
    # Play 10 games
    from game_runner import run_many
    run_many(agent1, agent2, 10)


if __name__ == "__main__":
    main()
