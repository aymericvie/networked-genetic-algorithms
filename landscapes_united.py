import numpy as np
import matplotlib 
matplotlib.rcParams['pdf.fonttype'] = 42 
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

def rastrigin(x, y):
    return 20 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y)
def sphere(x, y):
    return x**2 + y**2
def ackley(x, y):
    return -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2))) - np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + 20 + np.exp(1)


xR = np.linspace(-5.12, 5.12, 100)
yR = np.linspace(-5.12, 5.12, 100)
XR, YR = np.meshgrid(xR, yR)
ZR = rastrigin(XR, YR)

xS = np.linspace(-5.12, 5.12, 100)
yS = np.linspace(-5.12, 5.12, 100)
XS, YS = np.meshgrid(xS, yS)
ZS = sphere(XS, YS)

xA = np.linspace(-32.768, 32.768, 100)
yA = np.linspace(-32.768, 32.768, 100)
XA, YA = np.meshgrid(xA, yA)
ZA = ackley(XA, YA)

fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(131, projection='3d')
surf = ax.plot_surface(XR, YR, ZR, rstride=1, cstride=1, cmap='jet',
                       edgecolor='none')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_title('Rastrigin function (2d)')
fig.colorbar(surf, shrink=0.2, aspect=5)

ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(XS, YS, ZS, rstride=1, cstride=1, cmap='jet',
                       edgecolor='none')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$y$')
ax2.set_title('Sphere function (2d)')
fig.colorbar(surf2, shrink=0.2, aspect=5)

ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(XA, YA, ZA, rstride=1, cstride=1, cmap='jet',
                       edgecolor='none')
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$y$')
ax3.set_title('Ackley function (2d)')
fig.colorbar(surf3, shrink=0.2, aspect=5)
fig.tight_layout() 

#plt.savefig("functions_landscape.eps", format="eps")
plt.savefig("functions_landscape.png", format="png",dpi=600)
plt.show()
