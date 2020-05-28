import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from time import time
from celluloid import Camera


def barycentre(particules):
    n = particules.shape[0]
    bary = np.zeros((1, n))
    for i in range(n):
        bary = bary+particules[:, i]
    return bary/n


def PSO(f, dim, nb_particules, phi1, phi2, Nmax, eps, w_min, w_max, lower_bound, upper_bound):
    # initialisation

    b = min(np.abs(lower_bound), np.abs(upper_bound))
    particules = (np.random.rand(dim, nb_particules)*2*b-b)
    histo_particules = np.zeros((dim, nb_particules, Nmax))

    histo_particules[:, :, 0] = particules
    # print(particules)
    pb = np.copy(particules)
    gb = pb[:, 0]
    for i in range(nb_particules):
        if f(gb) > f(pb[:, i]):
            gb = pb[:, i]
    vitesse = np.random.rand(dim, nb_particules)
    
    w=w_max
    dw=(w_max-w_min)/Nmax

    n = 1
    deplacement_bary = 1000
    while n < Nmax and deplacement_bary > eps:
        bary = barycentre(particules)
        for p in range(nb_particules):
            U1 = np.random.rand()
            U2 = np.random.rand()
            vitesse[:, p] = w*vitesse[:, p] + phi1*U1*(pb[:, p]-particules[:, p])+phi2*U2*(
                gb-particules[:, p])  # rajouter le parametre w
            particules[:, p] = particules[:, p]+vitesse[:, p]

            # On gere les conditions aux bords
            mask1 = particules < lower_bound
            mask2 = particules > upper_bound
            particules = particules * \
                (~np.logical_or(mask1, mask2)) + \
                lower_bound*mask1+upper_bound*mask2

            histo_particules[:, :, n] = particules
            

            # Mise a jourÃ‚Â des pb et du gb
            if f(pb[:, p]) > f(particules[:, p]):
                pb[:, p] = particules[:, p]
                if f(gb) > f(pb[:, p]):
                    gb = pb[:, p]
        n += 1
        deplacement_bary = LA.norm(bary-barycentre(particules))
        w=w-dw

    return gb, n, histo_particules[:, :, :n]

def f(x):
    return -(x[1]+47)*np.sin(np.sqrt(np.abs(x[0]/2+x[1]+47)))-x[0]*np.sin(np.sqrt(np.abs(x[0]-(x[1]+47))))



X = np.arange(-512, 512, 0.25)
Y = np.arange(-512, 512, 0.25)
X, Y = np.meshgrid(X, Y)
Z = -(Y+47)*np.sin(np.sqrt(np.abs(X/2+Y+47))) - \
    X*np.sin(np.sqrt(np.abs(X-(Y+47))))


Nmax = 1000
eps = 1e-2
dim = 2
nb_part = 100
phi1 = 1
phi2 = 1
w_min=0.4
w_max=0.9

# temps_total = 0
# n_total = 0
# pres_total = 0

#Calcul de moyenne
# for i in range(100):
start = time()

res, n, histo_particules = PSO(
         f, dim, nb_part, phi1, phi2, Nmax, eps, w_min, w_max, -512, 512)

temps_exec = time()-start
#     temps_total+=temps_exec
#     n_total+=n
#     pres_total+=f(res)+959.7507

# print("temps_moyen:",temps_total/100,"s")
# print("# étapes moyen:",n_total/100)
# print("précision moyenne:",pres_total/100)

fig = plt.figure()
ax = fig.gca()
camera = Camera(fig)
for i in range(n):
    bary = barycentre(histo_particules[:,:,i])
    plt.scatter(histo_particules[0, :, i],
                histo_particules[1, :, i], color="red", s=5, marker="x")
    plt.scatter(bary[0][0],bary[0][1],color="m",s=40,marker="x")
    
    camera.snap()
surf = ax.contour(X, Y, Z)


print(res)
print(f(res))
print("Nombre d'etapes: ", n)
print("Temps d'execution : ", temps_exec)

anim = camera.animate(blit=True)
anim.save('scatter.gif')

