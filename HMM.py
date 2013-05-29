#4 states
#1. Studying for final "hmm", "so boring"
#2. Plaing video game: "Penta-Kill!", "Let's go play video games!", "OMG!"
#3. Final lower GPA: "OMG!", "hmm"
#4. Final boosts GPA: "OMG!", "Let's go play video games!"

#probabilities for transition
#                 study video lower  boost
A = matrix(RR, 4, [0.5,     0,  0.3, 0.2,
                   0.5,  0.28,  0.3, 0.02
                  0.05,  0.70, 0.20, 0.05
                  0.20,  0.02, 0.40, 0.38)]

#5 phrases
emission_symbols = ['hmm', 'so boring', 'Penta-Kill!', 'Let's go play video games!', 'OMG!']

#probabilites for uttering for each state
#                     'hmm'   'so boring' 'Penta-kill!', 'Let's go play video games!', 'OMG!'
B = matrix(RR, 4, 5, [0.20,         0.80,             0,                            0,      0,
                         0,           0,            0.02,                         0.50,    0.48,
                      0.20,           0,              0,                            0,    0.80,
                         0,           0,              0,                          0.30,   0.70])

#initial probabilities
initial = [0.3,0.1,0.3,0.3]

#creating model
model = hmm.DiscreteHiddenMarkovModel(A,B,initial,emission_symbols)

#these are commands u did in lecture. I just tried to keep in here just for examples.
set_random_seed(0); model.sample(10)
set_random_seed(1); model.sample(10)
set_random_seed(1);
a,b = model.generate_sequence(20)
a, list(b)
