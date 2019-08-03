---
layout: post
title: "⛏  MineRL Competition [Part 3]: Behavioral Cloning"
---

In this post, we study a basic approach to integrate one the main components of this competition: the demonstrations dataset. By providing this dataset, we can reduce the sample complexity of RL algorithms, which is essential given the speed and complexity of the simulated environment. We will apply a technique, called Behavioral Cloning, which is more related to supervised learning than RL. We will also try to enhance a cloned policy by applying A2C again.

As previously discussed, we want to avoid the temporal dependence of the task, which adds complexity to the problem. Since we want to iteratively build a more complex solution (and not going all-in from the start), in this post we focus on another task: the *MineRLTreechop* task. This specific task is practically Markovian even if the state is just one frame: once the agent sees a tree, it should move towards it and chop it down. Obviously, by assuming this setting, the agent could be stuck more frequently. We will discuss how to model this time dependence in future posts.

# Dataset
As we did for the *NavigateDense* task, we start by analyzing the dataset to better understand its characteristics. We start again by analyzing a batch of states and actions from the dataset.

![Sequence of observations](/images/minerl/treechop_pov.svg){:class="img-responsive" .center-image}
*Observations*

![Cumulated return ](/images/minerl/treechop_return.svg){:class="img-responsive" .center-image}
*Cumulated return*

We can notice the structure of the reward in the task by the return plot: each time a block of wood is chopped, the agent gets +1 reward, until it reaches 64 chopped blocks.
In this task we are missing the compass observation, since any tree is equally good and we do not have a goal position. This means that the agent is learning just from visual observations.

As we did for the *NavigateDense* task, we should look at how actions are distributed and correlated.
![Action means](/images/minerl/treechop_action_means.svg){:class="img-responsive" .center-image}

As expected, the most used action in this case is the attack action, used to effectively destroy blocks of wood and leaves. We also look at correlations, which show a very similar correlation between actions. This means that the dependence structure of action could be task-independent, which would be a very nice feature to build solutions for the more complex tasks. We will try to verify this hypothesis further on.

![Action heatmap](/images/minerl/treechop_action_heatmap.svg){:class="img-responsive" .center-image}

# Behavioral Cloning
We can now build a basic behavioral cloning setup, by leveraging the policy we have defined in the previous post. This way, we are basically trying to recover the average policy that generated the set of demonstrations.
While we do this, we also want to recover the value function (since A2C needs a critic), which can be directly estimated from the demonstrations.
We try to learn them together, by minimizing the MSE of the value error and maximizing the log probability of the performed action.

# Behavioral Cloning Results
To test the performance of the cloned behavior during training, in order to understand when to stop, we applied the following scheme:
- Hold-out of 20% of samples as a validation set
- Each 5 epochs run the policy on the environment many times and record the average score

Since we have also integrated the value function, we can set different levels of gamma. As expected, since we are constrained (for now) in a "reactive" setting, the lower gamma was better (see the plot below).
![Test performances](/images/minerl/run_bc.svg){:class="img-responsive" .center-image}
*Average return during training*

Using a gamma of *0.9*, we also show train and test losses. We can notice how the peak in the episode returns roughly corresponds to the moment in which the test performance increases.

![Test performances](/images/minerl/bc_traintest.svg){:class="img-responsive" .center-image}
*Train and test losses*

As always, we can look at a full episode to see how it performs! From the cumulated reward plot we can notice that the agent got stuck at some point, later recovering and completing the episode.

![Test performances](/images/minerl/treechop_replay_deterministic.gif){:class="img-responsive" .center-image}
*Replay of the episode*
![Test performances](/images/minerl/bc_reward.svg){:class="img-responsive" .center-image}
*Cumulated reward*

<!--
# Improving the cloned policy
As we all know, behavioral cloning suffers from a major issue: if the agent gets in a state that was never reached by the demonstration, it can get unstable and have "destructive" behaviors. This issue is alleviated by the size and diversity of the dataset. By further training the policy using A2C, we want to verify if we can get to an optimal behavior which is stable also in "unseen" states. Other possible solutions are to interleave these 2 phases in a loop, going back and forth between behavioral cloning and RL training.

# Conclusions
Conclusions here.
-->
