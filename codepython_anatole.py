
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


##  Fit data spectro (pour corriger les valeurs de mesure et être le plus précis possible)
def linear_func(x, a, b):
    return a * x + b
# Données expérimentales et théoriques
theo = np.array([546.1, 577, 579.1])
exp = np.array([546.4, 577.7, 579.9])

# Ajustement linéaire
params, _ = curve_fit(linear_func, theo, exp)


# Charger un fichier texte en tableau numpy
datav_0 = np.loadtxt("v''=0.txt")
# Application de la 
datav_0 = linear_func(datav_0, *params)
# trie d'ordre
datav_0.sort() 
# Conversion en nb d'onde cm-1
nbonde0=1/datav_0*1e7


datav_1 = np.loadtxt("v''=1.txt")
datav_1 = linear_func(datav_1, *params)
datav_1.sort() 
datav_1_1=datav_1[-5:]
nbonde1=1/datav_1_1*1e7


datav_2 = np.loadtxt("v''=2.txt")
datav_2 = linear_func(datav_2, *params)
datav_2.sort()

nbonde2=1/datav_2*10**(7)


#################

# Définition des indexation de v v' v" 
vv0 = np.linspace(24,38,15)[::-1]
vv1 = np.linspace(16,20,5)[::-1]
vv2 = np.linspace(9,16,8)[::-1]


# Calcul des delta sigma
ds0 = nbonde0[:-1] - nbonde0[1:]
ds1 = nbonde1[:-1] - nbonde1[1:]
ds2 = nbonde2[:-1] - nbonde2[1:]

def fctmod(v,we,xe):
    return ( we - 2 * we * xe * (v + 1) )

# concatenation des vv0 vv1 vv2
vv = np.array(list(vv0[:-1])+list(vv1[:-1])+list(vv2[:-1]))

# concatenation des concatenation des ecarts ds0 ds1 ds2
ds=np.array(list(ds0)+list(ds1)+list(ds2))

#fit sur la fonction 
popt0, _ = curve_fit(fctmod, vv, ds)

# Tracé des résultats
plt.figure(figsize=(10, 6),dpi =100)


plt.scatter(vv0[:-1], ds0, label="Données v''=0", color='red')
plt.scatter(vv1[:-1], ds1, label="Données v''=1", color='green')
plt.scatter(vv2[:-1], ds2, label="Données v''=2", color='blue')
plt.plot(np.linspace(0, 65, 100), fctmod(np.linspace(0, 65, 100), *popt0), linestyle="--", color='red')
for i in range(62):
    plt.fill([i,i,i+1,i+1],[0,fctmod(i, *popt0),fctmod(i, *popt0),0],color = 'green',alpha = 0.2)


plt.xlabel("Valeurs de v''")
plt.ylabel("Ds (écarts)")
plt.title("Ajustement des données avec la fonction choisie")
plt.xlim([0,65])
plt.ylim([0,140])
plt.legend()
plt.grid()
plt.show()

# Utiliser les paramètres trouvés
we, xe = popt0
print(" ")
print("we : ", np.round(we,6))
print("xe : ", np.round(xe,6))
print("we * xe : ", np.round(we*xe,6))

integral_value = np.sum(fctmod(np.linspace(0, 62, 63), *popt0))

## ancienne methode
# popt0, popt0[0]*popt0[1] [1.39345023e+02 1.58966486e-02] 2.215118871945482
# Racine (v où f(v) = 0) : 61.90634103751014
# Valeur de l'intégrale de 0 à 61.90634103751014 : 4244.605311801013
# De : 4313.72404368164
# ancienne methode
print(" ")
print("intégrale de v = 0 à 61.9 de delta sigma  = we - 2 * we * xe * (v + 1) =>  somme des delta sigma de 0 à environs 62")
print("delta_sigma = ", integral_value)

print(" ")
De=1/2*we-1/4*we*xe+integral_value
print("De = we/2 - 1/4 * we*xe + delta_sigma")
print('De :', De)