# Tsumego
A tsumego solver using deep Q-learning. https://en.wikipedia.org/wiki/Tsumego

## Example game after playing 264000 training games
![img1](https://user-images.githubusercontent.com/30019468/121734949-bfe22800-caf5-11eb-9b12-b706c566f7dc.png)
The three unmarked white stones are the initial stones. Black wins if he captures all white stones. Black 1 is a natural move, trying to destroy white's eye. White tries to expand his eyespace by playing white 2. Black 3 looks like a bad move, but white ignores it and creates an eye with white 4. This lets black play black 5 to save black 1. Black 3 now looks like a well-placed stone.

![img2](https://user-images.githubusercontent.com/30019468/121735532-8e1d9100-caf6-11eb-9108-4b75827e4249.png)
All moves in this diagram are bad lol. Black 2 looks like black might be trying to threaten to falsify white's eye. White 3 is a terrible move, filling white's own eye.

![img3](https://user-images.githubusercontent.com/30019468/121735750-dccb2b00-caf6-11eb-9ad8-c648dfdb4217.png)
Black 1 looks kind of reasonable. White tries to make eyespace in the center with white 2, but black 3 is the vital point, removing any eyespace there. White captures the two black stones with white 4, good.

![img4](https://user-images.githubusercontent.com/30019468/121736032-4a775700-caf7-11eb-8b95-f002ae25f44b.png)
Though it looks like white has two eyes, black 1 is a great move known as a throw-in, the only way to kill white in this position. White is now completely dead. The game continued until all white stones were captured and black won.
