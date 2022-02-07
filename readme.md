## Introduction
This readme file aims to give an overview on the steps and decisions taken to complete the three-dice assignment, more specifically, the following points will be touched:
- Choice of algorithm to calculate the optimal policy
- Choice of parameters
- Performance comparison between different algorithms and parameter values

The scope of the assignment is to produce an Agent which maximize the average score when playing the dice game as per rules dictated by the game itself. The problem is solved by calculating the optimal deterministic policy which maps each state to an action. <br>
In order for the agent to be able to play following different games rules (unknown to the Agent), the Agent implementation does not include specific game rules, it exploits the information the game class provides instead.


## Algorithms to find the Optimal Policy
Two algorithms were implemented to find the optimal policy which maximizes the expected score, Value Iteration and Policy Iteration. <br>
Their implementation reflects the sudo-code shown in [1] figure 17.4 pp 653 (Value Iteration) and figure 17.7 pp 657 (Policy Iteration).<br>
After empirical experiments Policy Iteration was chosen as the preferred method. The reasons and experimental data are reported in the sections below (**Results Comparison**).

### Value Iteration
This is the first algorithm being implemented. As the same name suggests, Value-Iteration starts with a random value of the vector of utility-values and progressively updates it until each single value converges to the actual value.
Convergence in Value-Iteration is reached when the **max norm** of the difference between the vector of utility values before and after the update is less than a certain threshold denoted as *max_err*.
The value of the threshold (*max_err*) affects both the final policy and the time the algorithm takes to converge.
Setting a relatively high value or *max_err* speeds up the algorithm at the expenses of a possible suboptimal policy.
Policies and execution times are reported in Table 2 for different values of *max_err*. Table 3 shows each policy performance in terms of average score and standard deviation.  

### Policy Iteration
Policy Iteration is the second algorihtm being implemented. It, on the other hand, stars with a given initial policy denoted with $\pi_0$. The algorithm then evaluates the the Value-function for each state for the given policy. This step is called policy evaluation. <br>
The second step of the algorihtm is called policy updated. The initial policy is updated if the maximum value of the Q-function for a given  state is greater than the previously calculated Utility-Value for the same state. The action for that state is updated within the policy with the action that produces the maximum value of the Q-function. 
The Q-function is calculated for each state, if the policy has not been updated then the algorithm has converged, otherwise the same two steps, policy evalution and policy updated are repeted until convegency is reached.

It is worth noticing that compared to Value iteration where the utility values needed to be calculated iteratively, in Policy Iteration the value-functions for each state, given a policy $\pi$, can be written as a system of linear equations and do not need any iterative solver to be computed.
Recalling the Value-Function for a sate S:


$ U_s =  \sum_a P_{\pi}(a|s) * \sum_{s’}(P_{\pi} (s'|a,s) * (R(s,a,s') + \gamma * U(s')) $


Since we are dealing with deterministic policies the first summation can be removed because for each state the agent will execute the action determined by the policy itself with probability 1

$ U_s =  \sum_{s’}( P_{\pi} (s'|a,s) * (R(s,a,s') + \gamma * U(s'))) $

Let's now imagine that after executing action a from state s we could end up in three different states $s'$, $s''$ and $s'''$ with probabilities $P1 = P_{\pi} (s'|a,s)$, $P2 = P_{\pi} (s''|a,s)$ and $P3 = P_{\pi} (s'''|a,s)$ where: 

$ P1 + P2 + P3  = 1 $

We can then expand the equation above as:

$ U_s =  P_{\pi} (s'|a,s) * R(s,a,s') + P_{\pi} (s''|a,s) * R(s,a,s'') + P_{\pi} (s'''|a,s) * R(s,a,s''') + \gamma\left(( P_{\pi} (s'|a,s)* U(s') + P_{\pi} (s''|a,s)* U(s'') + P_{\pi} (s'''|a,s)* U(s''')\right) $

Which can be rewritten as: 

$ U = A + \gamma * B * U $

Where:
- U is the vector of utility values for each state s
- $\gamma$ is the discount rate
- B' is the matrix containing the conditional probability of ending in a state s' which multiplies the utility value of the end state
- A is a vector or size ns x 1 of terms independent from the vector of utility values
The system of linear equation can be easily solved by using linear algebra.

$U = A + B' * U $


$ U = (I-B')^{-1} * A $

### Choices of parameters

The final value of the Value-Function depends on the reward, probability distribution and discount rate. While reward and probability distribution are given for each action and state, the discount rate gamma can be tuned to influence the final policy. <br> 
The discount rate $\gamma$ sets the balance between  short term rewards and long term rewards. $\gamma$ can take values between 0 and 1 included. Lower values of $\gamma$ make the optimal policy be more conservative and more likely to hold all three dice rather then re-roll to possibly obtain a better scores. Different values have been tested and it has been found that setting $\gamma = 1$ gives te best avarage score (see **Effect of Gamma on the optimal policy** section for more detailed considerations).  The empirical results  are shown in the **Result Comparison** section below. <br>
<br>
With gamma increasing the time required for both Value iteration and Policy iteration to converge increases. This does not constitute a major concern as for the standard game rules (3 dice 6 faces) both algorithms coverage well within the time limit of 30 seconds.<br> More information about the time taken by the two algorithms to converge to a solution are given in the **Result Comparison** section.


## Compatibility with different game rules
The algorithm in fully compatible with extended game rules, as it completely relies on the game class to establish the number of possible states, actions, rewards and probability distributions of ending in a certain state s1 given a state s0 and an action a0.


## Data structure
Once the optimal policy is calculated by means of the Policy Iteration algorithm, the policy itself is stored into a dictionary which uses states as keys and action as values.
This allows to quickly retrieve the best action a (determined by the policy) given a state s. 


## Results Comparison
This section reports the results of the experiments to establish both the optimal algorithm selection  to find the optimal policy and the value of parameters like gamma and *err_max* (maximum absolute error allowed in value Iteration between two consecutive updates)

It is also worth  mentionining that all results and graphs shown below are obtained for standard game rules (6 dice and 6 faces each dice)
The 


### Effect of Gamma on the optimal policy
As mentioned above, $\gamma$ sets the weight given to long term rewards. This means that with gamma = 0 the agent's policy won't even try to re-roll the dice as the Agent would be unaware of a possible better score. This makes it equivalent to the "always-hold" agent. Gamma = 1 can be only applied to finite enviroment horizon. For the dice game, having negative rewards for every dice re-roll makes the environment horizon finite, it is therefore safe using a value of gamma equal to 1. 
The graph below shows how the average score is affected when the discount rate increases from 0 to 1. For this experiment the solver used to find the optimal policy for a given value of gamma is Policy Iteration. Also the average score is calculated on a sample size or 10000 games. 


|   $\gamma$  | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1 | 
| --- |-----|-----|-----|-----|-----|-----|-----|-----|-----|---| ---|
|  **average score**   | 10.52    | 10.44    |   10.51  |  10.48   | 10.63    | 10.75    |  10.92   | 11.24    |   11.73  |  12.32 | 13.35  | 


| ![](avgScore_vs_gamma.png)|
| :----: |
| Fig 1: Effect of discount rate gamma on the average score (using Policy Iteration) |



It is clear that the best average score is obtained for gamma = 1

### Value Iteration vs Policy Iteration
This section provides a comparison on the performances and policy provided by Value Iteration and Policy Iteration algorithm. For all algorithm the value of the discount rate gamma was set to 1
The table below displays the whole policy obtained by the two approaches end different values of the maximum errors for value Irritation.

Table t1.0 shows how value iteration is generally slower to converge with respect to Policy iteration.
It also shows how the final policy is affected by the value of the maximum error allowed. 
For value iteration reducing the maximum error results in a more optimal policy at the expenses of the time taken to reach convergence.
It is worth highlighting that the optimal policy for value iteration is reached for a maximum error less than 1e-4.


On the other hand, it is interesting to notice how policy iteration produces the optimal policy while still being the fastest to coverage

| **STATE** | **ACTION** | **ACTION** | **ACTION** | **ACTION** |
| :-------: | :-------: |  :------: |  :----: |  :----: |
| **--**  | **Value Iteration <br> (err = 1e-0)** | **Value Iteration <br> (err = 1e-2)** | **Value Iteration<br> (err = 1e-4)** | **Policy Iteration** |
| (1, 1, 1) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) |
| (1, 1, 2) | (0, 1)    | (0, 1)    | (0, 1)    | (0, 1)    |
| (1, 1, 3) | (0, 1)    | (0, 1)    | (0, 1)    | (0, 1)    |
| (1, 1, 4) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) |
| (1, 1, 5) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) |
| (1, 1, 6) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) |
| (1, 2, 2) | (1, 2)    | (1, 2)    | (1, 2)    | (1, 2)    |
| (1, 2, 3) | (0,)      | (0,)      | (0,)      | (0,)      |
| (1, 2, 4) | (0,)      | (0,)      | (0,)      | (0,)      |
| (1, 2, 5) | (0,)      | (0,)      | (0,)      | (0,)      |
| (1, 2, 6) | (0,)      | (0,)      | (0,)      | (0,)      |
| (1, 3, 3) | (0,)      | (0,)      | (0,)      | (0,)      |
| (1, 3, 4) | (0,)      | (0,)      | (0,)      | (0,)      |
| (1, 3, 5) | (0,)      | (0,)      | (0,)      | (0,)      |
| (1, 3, 6) | (0,)      | (0,)      | (0,)      | (0,)      |
| (1, 4, 4) | (0,)      | (0,)      | (0,)      | (0,)      |
| (1, 4, 5) | (0,)      | (0,)      | (0,)      | (0,)      |
| (1, 4, 6) | (0,)      | (0,)      | (0,)      | (0,)      |
| (1, 5, 5) | (0,)      | (0,)      | (0,)      | (0,)      |
| (1, 5, 6) | (0,)      | (0,)      | (0,)      | (0,)      |
| (1, 6, 6) | (0,)      | (0,)      | (0,)      | (0,)      |
| (2, 2, 2) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) |
| (2, 2, 3) | (0, 1)    | (0, 1)    | (0, 1)    | (0, 1)    |
| (2, 2, 4) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) |
| (2, 2, 5) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) |
| (2, 2, 6) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) |
| (2, 3, 3) | (0,)      | (0,)      | (0,)      | (0,)      |
| (2, 3, 4) | (0,)      | (0,)      | (0,)      | (0,)      |
| (2, 3, 5) | (0,)      | (0,)      | (0,)      | (0,)      |
| (2, 3, 6) | (0,)      | (0,)      | (0,)      | (0,)      |
| (2, 4, 4) | (0,)      | (0,)      | (0,)      | (0,)      |
| (2, 4, 5) | (0,)      | (0,)      | (0,)      | (0,)      |
| (2, 4, 6) | **(0, 1, 2)** | **(0,)**      | **(0,)**      | **(0,)**      |
| (2, 5, 5) | (0,)      | (0,)      | (0,)      | (0,)      |
| (2, 5, 6) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) |
| (2, 6, 6) | (0,)      | (0,)      | (0,)      | (0,)      |
| (3, 3, 3) | **(0, 1, 2)** | **()**        | **()**        | **()**        |
| (3, 3, 4) | **(0, 1, 2)** | **()**        | **()**        | **()**        |
| (3, 3, 5) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) |
| (3, 3, 6) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) |
| (3, 4, 4) | ()        | ()        | ()        | ()        |
| (3, 4, 5) | **(0, 1, 2)** | **()**        | **()**        | **()**        |
| (3, 4, 6) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) |
| (3, 5, 5) | ()        | ()        | ()        | ()        |
| (3, 5, 6) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) |
| (3, 6, 6) | **(0, 1)**    | **()**        | **()**        | **()**        |
| (4, 4, 4) | ()         | ()        | ()        | ()        |
| (4, 4, 5) | ()        | ()        | ()        | ()
| (4, 4, 6) | **(0, 1, 2)** | **()**        | **()**        | **()**        |
| (4, 5, 5) | ()        | ()        | ()        | ()        |
| (4, 5, 6) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) | (0, 1, 2) |
| (4, 6, 6) | **(0, 1)**    | **()**        | **()**        | **()**        |
| (5, 5, 5) | ()        | ()        | ()        | ()        |
| **(5, 5, 6)** | **(0, 2)**    | **(0, 2)**    | **()**        | **()**        |
| **(5, 6, 6)** | **(0, 1)**    | **(0, 1)**    | **()**        | **()**        |
| (6, 6, 6)  | ()        | ()        | ()        | ()        |
| **TIME TO CONVERGE (s)** | **0.85** | **2.9** | **4.6** | **1.2**  |

Table 2: Comparison between Value-Iteration and Policy-Iteration, policies and time convergence 


The table below (Table 3) report the policies performances obtained with policy iteration and value iteration for different values of the maximum absolute error. As Value Iteration err = 1
The specific of the experiments reported in Table 1.1 are:  number of samples = 200, sample size = 5000


|               | **Value Iteration (err = 1e-0)**| **Value Iteration (err = 1e-2)** | **Policy Iteration** |
|---------------|------------------------------|------------------|------------------|
| **Mean**          |      13.2996            |         13.345                   |     13.348             |
| **Std Deviation** |      0.035              |         0.037                     |     0.035             |

Table 3: Performance comparison between policies obtained with different algorithms

| ![](Val_Iter_err_1.png) |
| :----: |
| Fig 2: Avarage score distribution for policy generated by  Value Iteration ( max_err = 1) |


| ![](Val_Iter_err_001.png) |
| :----: |
| Fig 3: Avarage score distribution for policy generated by Value Iteration ( max_err = 0.01) |


| ![](Pol_ite.png) |
| :----: |
| Fig 4: Avarage score distribution for policy generated by Policy Iteration |



## References

[1] Russell, S, & Norvig, P 2016, Artificial Intelligence: a Modern Approach, EBook, Global Edition : A Modern Approach, Pearson Education, Limited, Harlow. Available from: ProQuest Ebook Central. [30 September 2021].
# MDP_dice_game
