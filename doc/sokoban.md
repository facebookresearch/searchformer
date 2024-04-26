# Playing Sokoban

We are also including a small example that can be used to solve a Sokoban puzzle as a game.
This implementation uses PyGame and the repository [https://github.com/morenod/sokoban.git](https://github.com/morenod/sokoban.git).

First, this dependency must be initialized by running from the repositories root directory:

```
git submodule update --init --recursive
```

Then, the level contained in the included file `static/sokoban/example-7722.txt` can be played by running from the repositories root directory:

```
python -m searchformer.sokoban play --level=static/sokoban/example-7722.txt
```
