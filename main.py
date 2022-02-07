from abc import ABC, abstractmethod
from dice_game import DiceGame
import numpy as np
import math
import time


class DiceGameAgent(ABC):
    def __init__(self, game):
        self.game = game

    @abstractmethod
    def play(self, state):
        pass


class AlwaysHoldAgent(DiceGameAgent):
    def play(self, state):
        return (0, 1, 2)


class PerfectionistAgent(DiceGameAgent):
    def play(self, state):
        if state == (1, 1, 1) or state == (1, 1, 6):
            return (0, 1, 2)
        else:
            return ()


def play_game_with_agent(agent, game, verbose = False):
    state = game.reset()

    if (verbose): print(f"Testing agent: \n\t{type(agent).__name__}")
    if (verbose): print(f"Starting dice: \n\t{state}\n")

    game_over = False
    actions = 0
    while not game_over:
        action = agent.play(state)
        actions += 1

        if (verbose): print(f"Action {actions}: \t{action}")
        _, state, game_over = game.roll(action)
        if (verbose and not game_over): print(f"Dice: \t\t{state}")

    if (verbose): print(f"\nFinal dice: {state}, score: {game.score}")

    return game.score


def main():
    # random seed makes the results deterministic
    # change the number to see different results
    # or delete the line to make it change each time it is run
    np.random.seed(1)

    game = DiceGame()

    agent1 = AlwaysHoldAgent(game)
    play_game_with_agent(agent1, game, verbose=True)

    print("\n")

    agent2 = PerfectionistAgent(game)
    play_game_with_agent(agent2, game, verbose=True)


    agent3 = MyAgent(game)
    play_game_with_agent(agent2, game, verbose=True)

SKIP_TESTS = True


def tests(num=10, gamma=None):
    import time

    total_score = 0
    total_time = 0
    n = num

    np.random.seed()

    print("Testing basic rules.")
    print()

    game = DiceGame()
    #game = DiceGame(dice=5, sides=6)

    start_time = time.process_time()
    test_agent = MyAgent(game, False, gamma)
    total_time += time.process_time() - start_time

    for i in range(n):
        start_time = time.process_time()
        score = play_game_with_agent(test_agent, game)
        total_time += time.process_time() - start_time

        print(f"Game {i} score: {score}")
        total_score += score

    print()
    print(f"Average score: {total_score / n}")
    print(f"Total time: {total_time:.4f} seconds")
    return total_score



def test_distribution(s = 1, n=10000):
    import matplotlib.pyplot as plt
    scores = np.zeros(shape=101)
    gammas = np.zeros(shape=101)
    for i in range(101):
        gamma = i / 100
        total_score = tests(n, gamma)
        scores[i] = total_score/n
        gammas[i] = gamma

    #n, bins, patches = plt.hist(scores, 50,)
    #std = np.std(scores)
    #avg = np.average(scores)
    #med = np.median(scores)
    plt.scatter(gammas, scores)


    #print(f"mean = {avg}")
    #print(f'Standard Deviation = {std}')
    #print(f"median = {med}")
    plt.xlabel('Discount Rate value')
    plt.ylabel('Average Score ')
    plt.title('Avarage Score Vs Discount rate')
    plt.grid(True)
    plt.show()

    for idx, gamma in enumerate(gammas):
        print(f"gamma = {gamma}, score = {scores[idx]}")




class MyAgent(DiceGameAgent):
    def __init__(self, game: DiceGame, verbose=False, gamma=1):
        use_value_iter = True
        sync_updates = True
        self.verbose = verbose
        states = game.states
        idx = [i for i in range(len(states))]
        self.states_idx_map = dict(zip(states, idx))
        self.discount_rate = gamma
        actions = game.actions
        hold_action = [actions[-1] for _ in range(len(states))]
        self.policy = dict(zip(states, hold_action))  # the initial policy assign the hold action to each state
        if use_value_iter:
            t_start = time.time()
            U, num_iter = self.value_iteration(game, self.discount_rate)
            self.policy = self.get_optimal_policy(game, U, self.discount_rate)
            t_elapsed = time.time() - t_start
            if self.verbose:
                print(f"Value Iterations converged in {num_iter} iterations and {t_elapsed} seconds")
        else:
            t_start = time.time()
            if sync_updates:
                self.policy, num_iter = self.policy_iteration(game, self.discount_rate)
            else:
                self.policy, num_iter = self.policy_iteration_async(game, self.discount_rate)
            t_elapsed = time.time() - t_start
            if self.verbose:
                print(f"Policy Iterations converged in {num_iter} iterations and {t_elapsed} seconds")
        super().__init__(game)


    def value_iteration(self, game: DiceGame, discount_rate):
    # Value Iteration Algorithm to solve the system of Bellman euqations
        err = math.inf
        delta = 10**-2 # maximum absolute error
        iter_max = 10**4
        u_value = self.policy_evaluation(game, discount_rate, self.policy)
        u_value_p = u_value.copy()
        iter_num = 0
        while err > delta:
            iter_num += 1
            if self.verbose and iter_num % 1 == 0:
                print(f"{iter_num}  iteration of Value Iteration. Maximum error = {err}")
            if iter_num > iter_max:
                if self.verbose:
                    print("Maximum number of iterations exceeded")
                return u_value, iter_num
            for state_idx, state in enumerate(game.states):
                Q_values = self.get_q_values(game, state, game.actions, u_value_p, discount_rate)
                u_value[state_idx] = max(Q_values)
            err = max(abs(u_value-u_value_p))
            u_value_p = u_value.copy()
        return u_value, iter_num


    def get_q_values(self, game, state, actions, u_value_p, discount_rate):
    # Returns the Q values for every action given the state S
        Q = np.zeros(shape=(len(actions)))
        for action_idx, action in enumerate(actions):
            new_states, game_over, reward, probabilities = game.get_next_states(action, state)  # state, game_over, reward, probabilities
            if game_over:
                Q[action_idx] += probabilities * reward
            else:
                u_val_temp = np.array([u_value_p[self.states_idx_map[new_state]] for new_state in new_states])
                Q[action_idx] += sum(np.dot(probabilities, (reward + discount_rate * u_val_temp)))
        return Q

        #     Q[action_idx] = sum()
        #     for new_state_idx, probability in enumerate(probabilities):
        #         if game_over:  # if the action was to hold all dice then the game is over and there is no next state.
        #             Q[action_idx] += probability * reward
        #         else:
        #             new_state_idx = self.states_idx_map[new_states[new_state_idx]]
        #             Q[action_idx] += probability * (reward + discount_rate * u_value_p[new_state_idx])
        # return Q


    def policy_iteration(self, game, discount_rate):
    # Policy Iteration algorithm with synchronous updated of utility values
        actions = game.actions
        states = game.states
        hold_action = [actions[-1] for _ in range(len(states))]
        pi = dict(zip(states, hold_action))  # the initial policy assign the hold action to each state
        pi = self.policy.copy()
        changed = True
        iter_num = 0
        while changed:
            iter_num += 1
            changed = False
            u_values = self.policy_evaluation(game, discount_rate, pi)
            for state_idx, state in enumerate(states):
                policy_action = pi[state]
                q_value = self.get_q_values(game, state, game.actions, u_values, discount_rate)
                idx_best_action, = np.where(q_value == np.amax(q_value))
                best_action = game.actions[idx_best_action[0]]
                q_value_ba = max(q_value)
                q_value_policy = self.get_q_values(game, state, [policy_action], u_values, discount_rate)
                if q_value_ba > q_value_policy:
                    if self.verbose:
                        print(f"iteration number {iter_num}, state {state}, action {pi[state]} swapped with {best_action}")
                    pi[state] = best_action
                    changed = True
        return pi, iter_num


    def policy_iteration_async(self, game, discount_rate):
    # Policy iteration algorithm with asyncronous updates of the utility values
        actions = game.actions
        states = game.states
        hold_action = [actions[-1] for _ in range(len(states))]
        pi = dict(zip(states, hold_action)) # the initial policy assign the hold action to each state
        changed = True
        iter_num = 0
        u_values = self.policy_evaluation(game, discount_rate, pi)
        while changed:
            iter_num += 1
            changed = False
            for state_idx, state in enumerate(states):
                policy_action = pi[state]
                q_value = self.get_q_values(game, state, game.actions, u_values, discount_rate)
                idx_best_action, = np.where(q_value == np.amax(q_value))
                best_action = game.actions[idx_best_action[0]]
                q_value_ba = max(q_value)
                q_value_policy = self.get_q_values(game, state, [policy_action], u_values, discount_rate)
                if q_value_ba > q_value_policy:
                    if self.verbose:
                        print(f"iteration number {iter_num}, state {state}, action {pi[state]} swapped with {best_action}")
                    pi[state] = best_action
                    changed = True
                    u_values[state_idx] = self.get_q_values(game, state, [pi[state]], u_values, discount_rate)  # asincronous update
        return pi, iter_num


    def policy_evaluation(self, game, discount_rate, pi):
        from numpy.linalg import inv
        ns = len(game.states)
        B = np.zeros(shape=(ns, ns))
        A = np.zeros(shape=(ns, 1))
        I = np.eye(ns)
        for state_idx, state in enumerate(game.states):
            action = pi[state]
            new_states, game_over, reward, probabilities = game.get_next_states(action, state)
            for idx, state_prob in enumerate(probabilities):
                A[state_idx][0] += reward * state_prob
                if not game_over:
                    new_state_idx = self.states_idx_map[new_states[idx]]
                    B[state_idx][new_state_idx] = discount_rate * state_prob
        # system of linear question of the form: U = A + BxU --> (I-B)U = A --> U = inv(I-B) x A
        U = np.matmul(inv(I - B), A)
        return U


    def get_optimal_policy(self, game, u_value, discount_rate):
    # Get the optimal policy pi* from the vectors of Utility values
        policy = dict()
        for state in game.states:
            Q_s = self.get_q_values(game, state, game.actions, u_value, discount_rate)
            idx, = np.where(Q_s == np.amax(Q_s))
            policy[state] = game.actions[idx[0]]
        return policy


    def play(self, state):
    # Given a state retrieve the action specified by the policy
        action = self.policy[state]
        return action



if __name__ == "__main__":
    tests(num=10, gamma=1)
    #test_distribution()