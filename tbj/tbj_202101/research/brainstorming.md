Fastball Command

- can't measure intent
- assume highest value location in 'neighbourhood', and so distance to location is size of miss

Shortstop Defense

- model expected outs on play
- observed variance from expectation would be incremental above average for play
- sum differences from expectation for each player

Choice: SS Defense

Plan:
- Preprocessing Pipeline; add features around SS distance to intersection, and time related to the play (i.e. time to intersection point). Consider not just using orthogonal vector to intersection, possibly optimize time of play (minimize time to intersecion using assumption of player speed). Maybe also momentum to base feature
- Consider line drives separately (hang time + launch speed?)

- Build model(s) for expected rate of out on play

- Sum players OAA


Limitations:
- Will be ignoring double plays. Could possibly be done with a model of the second out on doubleplays, and restrict to plays where SS had opportunity (i.e. 3-6, 4-6, 5-6-3, etc.) BUT in this dataset there are no out counts so it's impossible to know if a turn was unsuccessful due to slow turn, or if there were already 2 outs. 