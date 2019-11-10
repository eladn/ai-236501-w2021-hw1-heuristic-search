import numpy as np
from matplotlib import pyplot as plt

X = np.array([400, 450, 900, 390, 550])

# TODO: Write the code as explained in the instructions
# raise NotImplemented()  # TODO: remove!

T = np.array(np.linspace(0.01, 5, 100))

P = np.zeros((len(T), len(X)))

for i,x in enumerate(X):
    P[:, i] = (x / min(X)) ** -(1/T)

row_sums = P.sum(axis=1)
P = P / row_sums[:, np.newaxis]

print(P)

for i in range(len(X)):
    plt.plot(T, P[:, i], label=str(X[i]))

plt.xlabel("T")
plt.ylabel("P")
plt.title("Probability as a function of the temperature")
plt.legend()
plt.grid()
plt.show()
exit()
