# I Built an ML Model That Predicts Pokemon Battles Before a Single Move Is Made

I've been playing Pokemon for 27 years. Started with a hand-me-down Game Boy and Pokemon Blue, no idea what I was doing, just picking the cool-looking ones. Most of us did.

Turns out that instinct — *which six Pokemon you bring* — matters way more than most people think. Not just in-game, but on the competitive ladder too. The team you pick before turn one constrains everything that happens after. It's the biggest decision you make and you make it blind.

So I built a model that tries to quantify it.

**FutureSightML** is an open-source ML system that predicts win probability from team composition alone. No moves played. No rating peeked at. Just: here are 12 Pokemon, who wins?

## What it actually does

You give it two teams (or one team against the metagame). It tells you the probability Team A beats Team B. That's it. Simple question, hard problem.

Under the hood it's a neural transformer encoder with a pairwise matchup matrix branch, ensembled with an XGBoost model running on 618+ hand-crafted features. The neural side learns embeddings for every species, move, item, and ability. The XGBoost side knows about type coverage, stat distributions, threat matrices, and dozens of competitive mechanics details that matter (Intimidate punishing physical attackers, Heavy-Duty Boots dodging hazards, Regenerator + pivot move sustainability loops).

The ensemble weighs them per-format — some formats lean heavier on the neural model, some on XGBoost.

## The numbers

Here's what it achieves on held-out test data, predicting purely from team composition with ratings equalized:

| Format | AUC |
|---|---|
| Gen 9 OU | 0.759 |
| Gen 9 Ubers | 0.814 |
| Gen 3 OU | 0.854 |
| Gen 1 OU | 0.810 |

For context: Dota 2 draft prediction — a comparable "predict the winner from the lineup" problem — typically hits 0.66-0.71 AUC. Hearthstone deck prediction lands around 0.65-0.68. Our worst format still beats those benchmarks.

We support **129 formats across Generations 1 through 9**. OU, UU, RU, NU, Ubers, VGC, Doubles OU, Little Cup, Monotype, National Dex — if people play it on Showdown, we probably have a model for it.

## Why this is harder than it sounds

Pokemon team prediction has a unique problem that Dota/League draft prediction doesn't: the design space is enormous and sparse. In Dota, you pick from ~120 heroes with fixed kits. In Pokemon, you pick 6 from 1000+ species, each with 4 moves chosen from a pool of hundreds, an item, an ability, EVs, a nature, and (in Gen 9) a Tera type. The combinatorial space is effectively infinite.

Also — and this took us embarrassingly long to figure out — the game mechanics change dramatically across generations. In Gen 1-3, whether a move is physical or special isn't a property of the move. It's a property of the move's *type*. Psychic is always special. Ghost is always physical. Gen 4 changed this. If you don't account for that, your Gen 1 damage calculations are just... wrong. Your features are wrong. Your model is learning from garbage.

So we built generation-aware everything. Type charts, stat formulas, move categories, learnsets — all correct for the generation you're predicting in. Gen 1 gets its unified Special stat. Gen 2-5 get the Steel-resists-Dark/Ghost chart. Every generation's quirks are respected.

## What you can do with it

The project includes:
- A **desktop app** with a retro-styled GUI (think Pokemon Stadium menus) — pick a format, paste or generate a team, see your predicted win rate
- A **genetic algorithm team builder** that evolves teams to maximize predicted win rate against the metagame
- A **battle simulator** with Monte Carlo estimation
- A **FastAPI backend** with full API docs

It's all local. Nothing leaves your machine. Your teams, your data, your predictions — all on your hardware.

## Trained on millions of real battles

The models train on replay data from Pokemon Showdown — real games between real players. We parse the full battle logs to extract both teams, both players' ratings, and the outcome. The current training set uses up to 50,000 battles per format, with data sourced from both the Showdown replay API and a 30.5-million-replay bulk dataset.

Every metric we report is honest: held-out test set, equalized ratings (so the model can't just learn "higher-rated player wins"), and proper train/test splits applied before data augmentation.

## Open source

The whole thing is MIT licensed and on GitHub. Clone it, train your own models, tear it apart, build on it. The README has everything you need to get started.

I built this because I wanted to know — actually *know*, with data — whether my team was good. Not "my friend says Garchomp is broken" good. Quantifiably, measurably good. Twenty-seven years of loving this game and I finally have a number for it.

**GitHub:** github.com/HotHams/FutureSightML
