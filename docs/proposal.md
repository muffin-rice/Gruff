---
layout: default
title:  Proposal
---

## Introduction

Intro to bridge: 

Definition of [trick](https://www.britannica.com/topic/trick)

Bridge is a non-deterministic card game involving imperfect information and teamwork. A short summary of the game loop (divided into its 2 phases) is as follows: 

2 pairs of players first participate in an "auction" where they propose in turn (through "bids") how many tricks they will win, trying to find good bids through the limited information they have. After the bid is created, a "contract" is formed – the winning pair is trying to fulfill their bid, hence the naming "contract". 

After the contract is created, 13 tricks are played and the number of tricks won by each pair are counted. Depending on the contract and the differences of tricks won vs. the contract, points are given out. This process is repeated until enough points have been won. 

In our agent, we focus on the bidding phase; for simplicity, we will not consider features of the game such as vulnerability. 

A more in-depth look at bidding is as follows: 

“The bidding phase of Contract Bridge is like Mini-Bridge but with a much larger state space (each player now holds a hand with 13 cards from 4 suits). Unlike Mini-Bridge, a player has both her teammate and competitors, making it more than a full-collaborative IG. Therefore, multiple trade-offs needs to be considered. Human handcrafted conventions to signal private hands, called bidding systems. For example, opening bid 2♥ used to signal a very strong hand with hearts historically, but now signals a weak hand with long hearts. Its current usage blocks opponents from getting their best contract, which happens more frequently than its previous usage (to build a strong heart contract).” [Tian et al., 2020](https://arxiv.org/pdf/2008.06495.pdf)

## Summary 

We will be using DeepMind's [OpenSpiel](https://github.com/deepmind/open_spiel) as a starter for the environment. Our agent should be able to output a bid based on previous bids and knowledge of its current hand. 

Our goal is to be able to test and implement algorithms that are able to perform well in imperfect information, zero-sum, and collaborative games, of which Bridge is a unique game that fills all of these criteria. 

## Algorithms 

We plan on using algorithms such as joint policy search along with a deep learning agent, both of which were previously explored in RL and Bridge RL literature.

## Evaluation 

Quantitavely, we can use metrics that measure the difference in bidding ability such as International Matching Points per Board (IMPs/b) to compare our agent with other Bridge AI agents of varying strength. There are also other metrics and methods outlined in previous literature that we may use. For example, we can use a Bridge agent from a paper in 2016 as a baseline, where an average IMPs/b of +0.1 suggests a significant improvement. Because Bridge is played on a single deck of 52 cards, we can generate a large set of hands from some permutation of those cards. The [JPS paper](https://arxiv.org/pdf/2008.06495.pdf#section.7) describes a training set of 2.5 million hands, and an evaluation set with 50k hands. We will utilize the same hands to maximize our chances for success.

Our agent should also perform well by certain qualitative metrics; one particular sanity-check case would be ensuring that the agent can leverage basic heuristics (not bidding high on weak hands, not throwing away high cards on a surely lost trick). The agent's actions can be verified with the OpenSpiel game tree [visualizer](https://github.com/michalsustr/spielviz). We expect exceeding SoTA Bridge AI to be extremely difficult, but performing much better than very basic agents and meeting SoTA agents should be realistic. 