---
layout: post
title: "⛏  MineRL competition [Part 1]: Domain analysis and first simple solutions"
---

[MineRL](https://www.aicrowd.com/challenges/neurips-2019-minerl-competition) is a recently started NeurIPS19 competition, with the goal of sample-efficient DRL algorithms in a very complex and hierarchical problem. I always had a passion for minecraft, since it is a quite complete game, which (in humans) stimulate strategy, exploration and building creativity.
MineRL was created starting from [Malmo](https://github.com/microsoft/malmo), in which other competitions where hosted in the past (e.g. [this one](https://www.microsoft.com/en-us/research/academic-program/collaborative-ai-challenge/)). This time, I also want to thank the organizers for the excellent job of building a simple package ready to use (Malmo, instead, was quite complex to setup and to use).

In the recent years, we have seen a lot of impressing DRL achievements (citing the common ones, [Alphastar](https://deepmind.com/blog/alphastar-mastering-real-time-strategy-game-starcraft-ii/) and [OpenAI Five](https://openai.com/five/)), but a lot of them share the common issue of sample complexity: to reach a satisfactory result, a huge amount of sampled experiences need to be gathered. This competition has the exact goal of understanding this issue better: can we design sample-efficient algorithms which learn to act in complex environments with the aid of a dataset of human demonstrations? For this reason, I think the main topics to be covered in this competition are:
- Inverse RL to exploit the dataset
- Transfer between tasks (each "higher-level" task contains part of the lower ones)
- Hierarchical controller: choosing sequence of actions (options)
- Model based: using the demonstrations to learn a model of the environment

These are the first things that come to my mind, but we will see what happens next!
In this post, we will approach the problem in one of its simpler configurations, first analyzing the corresponding dataset, and then applying a simple DQN to the environment.

# Domain Analysis
In this particular part, we will only use the environment *MineRLNavigateDense-v0*, which is the easiest one. It requires to navigate to a given block (diamond), which is placed on the surface at a distance of 64 blocks. It is *dense*, i.e. the reward is shaped to point exactly to the block. We will use this property to use low values of gamma, since an almost greedy controller is good in this setting of dense reward.
Also the dataset has the same task division, so we will focus on the same one.

When we deal with offline datasets, it is always better to explore them a little bit, both to understand the problem better but also to cross-check that everything works fine.
The basic action to do on the dataset is to call *data.seq_iter*, which creates a generator of sequences belonging to the same episode, return observations, actions, rewards and termination flags. We can see an example of images returned by the dataset in the following image. The observation also contains a compass angle, pointing to a position in the proximity of the goal. In the following plot, we also show the reward per timestep inside the same episode, which has a value a bit lower than 1.0 (which is the reward of moving exactly one block towards the goal).

![Sequence of observations](/images/minerl/obs_sequence.png){:class="img-responsive" .center-image}
![Sequence of compass and reward](/images/minerl/compass_sequence.png){:class="img-responsive" .center-image}
If you want to read more on the environment or the dataset, go to [MineRL documentation](http://minerl.io/docs/#).

Since we are interested in applying DQN, which has a constraint on discrete actions, we need to look at the action space in detail. In MineRL, actions are described as a dictionary of possible *sub-actions*, which are controllable independently. These sub-actions form a macro action that is performed by the agent at each timestep. Among these sub-actions, only one is continuous: the one that controls the camera movement, expressed as a tuple of pitch and yaw movement (see below picture), from *-180* to *+180* degrees per tick.

![Camera actions](/images/minerl/camera.jpg){:class="img-responsive" .center-image}
*Moving a camera in space*

Since our simple approach uses DQN, we need to understand how to discretize these continuous actions. Instead of relying on "expert-guess", we can look directly at the data we have.

![Action distribution](/images/minerl/camera_dist.png){:class="img-responsive" .center-image}
*Camera distribution, notice the logarithmic scale.*

Given this distribution, we chose to approximate the camera actions in each axis using

$$ \{-20.0, -2.0, 0.0, +2.0, +20.0\} $$

as possible values.

CONTINUES...

<!--
- Analysis:
  - Reward from dataset misaligned with env
  - Actions frequency and correlation
    - Action composing? Automatic discovery of clusters?
- Flat world, no stochasticity:
  - Solution with DQN, description of code
  - Repeated actions?
  - Training curves
  - GIF of policy rollout
  - Commentary
- Flat world, stochasticity + POV:
  - Solution with DQN, description of code
  - Training curves
  - GIF of trained policy
  - Commentary
- Default world
  - Solution with base DQN ?
  - Pretraining on dataset
-->
