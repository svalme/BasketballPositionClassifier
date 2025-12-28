# BasketballPositionClassifier

This project builds a machine learning system to classify basketball player positions using multi-season WNBA and NBA performance data. Rather than relying on rigid position labels, the model learns positional roles directly from statistical profiles.

The system uses a hierarchical classification approach:
- Stage 1: A coarse classifier predicts whether a player is a Guard, Forward, or Center.
- Stage 2: Once the coarse role is determined, the model routes the player to a specialized classifier trained only on that role.

For example:
    - If Stage 1 predicts Guard, the model then classifies among:
        * Guard
        * Guard-Forward
    - If Stage 1 predicts Forward, the model chooses between:
        * Forward
        * Forward-Guard
        * Forward-Center
    - If Stage 1 predicts Center, the model chooses between:
        * Center
        * Center-Forward

Each fine classifier only sees examples from its own group, which greatly reduces ambiguity. This design reflects the reality of modern basketball, where positional boundaries are fluid and role-based.

## Key Features
- Multi-season WNBA + NBA dataset, 2020-2024, using the nba_api python package
- Cleaned and normalized position labels
- Feature engineering using per-game and per-36 stats
- Hierarchical (two-stage) classification
- Train/test split by season
- Interpretable models using logistic regression

## Player Positions (Coarse)
| Label  | Description |
| ------------- | ------------- |
| G  | Guard — primary ball-handlers and perimeter players  |
| F  | Forward — wings and versatile scorers/defenders  | 
| C  | Center — interior players focused on rebounding and rim protection |

## Player Positions (Fine Grained)
These provide more detail and capture hybrid roles commonly seen in modern basketball. Many players operate across multiple roles depending on lineup and system.

| Label  | Description |
| ------------- | ------------- |
| Guard | Traditional point or shooting guard |
| Guard-Forward | Combo guard / wing scorer |
| Forward | Standard wing or power forward |
| Forward-Guard | Wing with strong ball-handling responsibilities |
| Center | Traditional interior big |
| Center-Forward | Big with perimeter or stretch ability |
| Forward-Center | Hybrid big-forward role |
