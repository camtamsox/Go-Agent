1. Introduction

In the pursuit of making the best AI go player we changed the following things from part 2. We upgraded MCTS so that it contains relevant information from past trees it has made, as well as optimized the time it uses. We found the optimal first move and first response move for our agent. We made a combined agent that uses MCTS for the early and mid game , and when the game is close to ending it uses IDS with the learned value function heuristic.

2. Changes to MCTS

In the original MCTS implementation, the agent would create an entirely new tree for every move. However, we realized that there is information that MCTS can keep from the previous move. More specifically, the tree node corresponding to the new state contains valuable information from the simulations done previously. So, we decided to set the root of the MCTS tree to the node corresponding to the current state in the previous MCTS tree.

In addition to saving the tree, we also optimized how long each move would take. We originally had set our time limit to 1, but we were concerned that with the longer games on the 9x9 board, our agents would run out of time. We settled on a time limiter of time_limit\*.9 - 0.5 which should give ample time for an agent to make moves for the whole game even on a 9x9.

We also attempted different rollout strategies. These included epsilon greedy with both the given heuristic and the value function heuristic, as well as AlphaBeta and IDS. After testing all these options we concluded that random was still the best rollout policy as the MCTS agent with random rollout was able to win the most against the MCTS agents with other rollout policies .

Finally we used MCTS to find optimal starting moves, see more in the Opening Book section of the README.

3. Combining MCTS and IDS

Alpha-beta is guaranteed to find the best moves available when it can look through the entire game tree. So, if given enough time, running IDS will always find the most optimal move. However, this is too computationally expensive when there are many possible game states that can be reached. Only towards the end of a go game can IDS look at all possible game states. This is why we decided to only use IDS at the very end of the game. For the rest of the game, we used our modified MCTS, as this was our best performing bot for this stage.

To find when we needed to switch from MCTS to IDS, we needed to find how close a given state was to being at the end of the game. We decided to have fixed move values for switching. To find the optimal move values, we first simulated many 5x5 and 9x9 games (improved MCTS vs original MCTS) which showed that games last around 48 moves for 5x5 and 160 moves for 9x9. Next, we simulated IDS games against random to find how many moves it could lookahead. For a 5x5, IDS could lookahead around 8 moves while on a 9x9, it could only lookahead around 5 moves. With these numbers, we could find an initial guess for the move value to switch from MCTS to IDS. From there, we manually tuned the values and found move 44 to be optimal for 5x5 games and move 155 to be optimal for 9x9 (when our agent played against original MCTS). We hypothesize that the optimal moves are higher than our original guesses because if IDS can’t look through every move, it will have to rely on its bad heuristic which would be significantly worse than using normal MCTS. It is better to ensure that IDS can fully look through the rest of a game than use IDS prematurely.

It is important to note that our IDS uses the learned value heuristic for 5x5, and the given heuristic for a 9x9. This is because we did not have enough time or the dataset to train a new value function for a 9x9 grid.

4. Opening book

For MCTS, the opening moves are the hardest because it has the highest branching factor. However, there are few possible game states during the opening moves which means that the agent can memorize (precompute) the optimal moves for some depth into the search tree. That way, it can save its computation time for later in the game.

When implementing the opening book, we made a new method in game_runner called build_opening_book and added file opening/saving functionality in MCTS agent. When build_opening_book is called, it calls get move once for MCTS if MCTS is the first player to move and it calls get move once for every possible move if MCTS is the second player to move. In the MCTS agent class, the agent adds on to an existing opening book file if it exists or makes a new one. When get_move is called and MCTS is building the book, it first builds the simulation tree and then saves the book. The game stops being simulated after that. When get_move is called and MCTS is not building a book, it checks if the game state is in the opening book dictionary and returns the corresponding action if it is, otherwise it proceeds as normal.

For the 5x5, we were able to create an opening book of the first 3 moves. When creating this opening book, we let MCTS build a tree for 60 seconds on each of the 27 possible starting board states and selected the best actions. For the 9x9, we created an opening book of the first 2 moves and ran MCTS for 16 seconds on each of the 83 possible starting positions.

5. About the Agents

MCTSAgent: This is the improvedMCTS, with our all our optimizations

MCTSOriginalAgent: This is the original implementation of MCTS from part 2. We used this agent to test against our new implementation.

FinalAgent: This agent makes ove based off of MCTS and IDS (with the learned value function). If we think the game is ending, that is the number of moves made in a game is past a certain point, we make a move with IDS, otherwise this agent makes a move with MCTS.

No changes were made to Greedy, IDS, AlphaBeta, Minimax, and Random, and these agents were used in the testing of our agents.

To see the results of our model, please see the pdf labeld "Results"
