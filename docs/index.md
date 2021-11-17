---
layout: default
title:  Home
---

## What is Bridge? 
Bridge is a non-deterministic card game involving imperfect information and teamwork. A short summary of the game loop (divided into its 2 phases) is as follows: 

2 pairs of players first participate in an "auction" where they propose in turn (through "bids") how many tricks they will win, trying to find good bids through the limited information they have. After the bid is created, a "contract" is formed – the winning pair is trying to fulfill their bid, hence the naming "contract". 

After the contract is created, 13 tricks are played and the number of tricks won by each pair are counted. Depending on the contract and the differences of tricks won vs. the contract, points are given out. This process is repeated until enough points have been won. 

In our agent, we focus on the bidding phase; for simplicity, we will not consider features of the game such as vulnerability. 

A more in-depth look at bidding is as follows: 

“The bidding phase of Contract Bridge is like Mini-Bridge but with a much larger state space (each player now holds a hand with 13 cards from 4 suits). Unlike Mini-Bridge, a player has both her teammate and competitors, making it more than a full-collaborative IG. Therefore, multiple trade-offs needs to be considered. Human handcrafted conventions to signal private hands, called bidding systems. For example, opening bid 2♥ used to signal a very strong hand with hearts historically, but now signals a weak hand with long hearts. Its current usage blocks opponents from getting their best contract, which happens more frequently than its previous usage (to build a strong heart contract).” [Tian et al., 2020](https://arxiv.org/pdf/2008.06495.pdf)

## What is Gruff? 
Gruff is the name of the troll under the bridge in the folktale *The Three Billy Goats Gruff*. 

Here, it is a Neural Network that outputs a bid given the game state, which includes their current hand and all of the previous bids. 

[Source code](https://github.com/muffin-rice/Gruff)

Reports:

- [Proposal](proposal.html)
- [Status](status.html)
- [Final](final.html)

