import numpy as np, random
random.seed(42);states=[(i,j) for i in range(3) for j in range(3)];Q=np.zeros((3,3,2));alpha,gamma,epsilon=.1,.9,.2
def step(state,action):
 n,e=state
 if action==0: nxt=(max(0,n-1),min(2,e+random.choice([0,1])))
 else: nxt=(min(2,n+random.choice([0,1])),max(0,e-1))
 return nxt,-sum(nxt)
for _ in range(5000):
 state=random.choice(states);a=random.randrange(2) if random.random()<epsilon else int(np.argmax(Q[state]));nxt,r=step(state,a);Q[state[0],state[1],a]+=alpha*(r+gamma*np.max(Q[nxt])-Q[state[0],state[1],a])
print('Q-table:');print(np.round(Q,2))
for st in states:print('Traffic state',st,'best action','North green' if np.argmax(Q[st])==0 else 'East green')
