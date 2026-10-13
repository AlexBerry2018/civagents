# Roles of the AI government

## Ruler
Sets the strategy every 10 turns and adjusts tax/science rates. Does not command units directly.

## Economist
Commands Settlers, Workers and cities: where to found cities, how to improve land, what cities build.

## Military
Commands Explorers and combat units: exploration, huts, defense of cities, reacting to threats.

## Science
Chooses research and research goals.

## Executor
Deterministic code, not a language model: validates every action against the list of allowed actions and sends it to the game. Invalid actions never reach the game.
