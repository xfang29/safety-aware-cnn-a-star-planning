# Safety-Aware CNN Cost Map for A*-Based Path Planning

## Status

Proposal stage / ongoing project.

## Overview

This project proposes a hybrid learning-and-search framework for autonomous-driving path planning. The goal is to train a CNN or lightweight U-Net model to predict a safety-aware cost map from simplified bird's-eye-view driving scenes, and then use the learned cost representation to guide a standard A* planner.

The neural network does not directly replace the planner. Instead, it provides a learned cost map or path-probability map that helps A* generate safer and more driving-like paths.

## Planned Method

- Generate synthetic driving-like grid maps with road boundaries, lane-center information, static obstacles, start points, and goal points.
- Generate expert paths using safety-aware A* or Dijkstra search.
- Train a CNN/U-Net model to predict a path-probability map or residual cost map.
- Combine the learned map with rule-based safety costs.
- Evaluate against standard A*, Dijkstra, and rule-based safety-aware A*.

## Planned Evaluation Metrics

- Success rate
- Collision rate
- Path length ratio
- Planning time
- Expanded nodes
- Minimum obstacle clearance
- Lane deviation
- Path smoothness

## Documents

- `docs/Research_Project_Topic_Proposal.pdf`: project proposal and planned research direction.

## Notes

This repository currently contains the project proposal. Code, experiments, and results will be added as the project progresses.
