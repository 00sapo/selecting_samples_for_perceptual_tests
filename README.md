# Selecting samples for listening tests

In this repo you find useful code for selecting samples for designing listening tests.

In 'contardo' there is a state-of-art method for solving the *p-dispersion* problem.

In the root directory there is my code for a non mathematically-proven
clustering approach to the same problem (much more efficient but much less
effective). Note that my approach seems to be better here, but I don't really
understand why: it seems that the Contardo's code is not suitable for this case.

To see the examples, install Gurobi, then install all dependencies for the julia files,
finally run `julia -i ours.jl`.

In future, there should be another approach which does *p-dispersion* + clustering.

Federico Simonetta
[https://federicosimonetta.eu.org](https://federicosimonetta.eu.org)
