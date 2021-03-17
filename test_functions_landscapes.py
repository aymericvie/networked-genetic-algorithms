import matplotlib 
matplotlib.rcParams['pdf.fonttype'] = 42 
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# =============================================================================
# # Rastrigin plot
# =============================================================================
fig = plt.figure()
ax = plt.axes(projection='3d')

def rastrigin(x, y):
    return 20 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y)

x = np.linspace(-5.12, 5.12, 100)
y = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x, y)
Z = rastrigin(X, Y)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='jet', edgecolor='none')
ax.set_title('Rastrigin function (2 dimensions)');
plt.savefig("rastrigin_landscape.eps", format="eps")

# =============================================================================
# # Sphere plot
# =============================================================================
fig = plt.figure()
ax = plt.axes(projection='3d')

def sphere(x, y):
    return x**2 + y**2

x = np.linspace(-5.12, 5.12, 100)
y = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x, y)
Z = sphere(X, Y)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='jet', edgecolor='none')
ax.set_title('Sphere function (2 dimensions)');
plt.savefig("sphere_landscape.eps", format="eps")

# =============================================================================
# # Ackley plot
# =============================================================================
fig = plt.figure()
ax = plt.axes(projection='3d')

def ackley(x, y):
    return -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2))) - np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + 20 + np.exp(1)

x = np.linspace(-32.768, 32.768, 100)
y = np.linspace(-32.768, 32.768, 100)
X, Y = np.meshgrid(x, y)
Z = ackley(X, Y)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='jet', edgecolor='none')
ax.set_title('Ackley function (2 dimensions)');
plt.savefig("ackley_landscape.eps", format="eps")