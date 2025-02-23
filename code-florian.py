import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.integrate import quad

# Charger un fichier texte en tableau numpy
datav_0 = np.loadtxt("v''=0.txt") 
datav_0.sort() 
nbonde0=1/datav_0*1e7
datav_1 = np.loadtxt("v''=1.txt")
datav_1.sort() 
datav_1_1=datav_1[-5:]
nbonde1=1/datav_1_1*1e7
datav_2 = np.loadtxt("v''=2.txt") 
datav_2.sort()  
nbonde2=1/datav_2*10**(7)

vv0 = np.linspace(24,38,15)[::-1]
vv1 = np.linspace(16,20,5)[::-1]
vv2 = np.linspace(9,16,8)[::-1]




ds0=[]
ds1=[]
ds2=[]
for i in range(0,len(nbonde0)-1):
    ds0.append(list(nbonde0)[i]-list(nbonde0)[i+1])

for i in range(0,len(nbonde1)-1):
    ds1.append(list(nbonde1)[i]-list(nbonde1)[i+1])
    
for i in range(0,len(nbonde2)-1):
    ds2.append(list(nbonde2)[i]-list(nbonde2)[i+1])
    

def fctmod(v,we,xe):
    return (1*we-we*xe*(v+1))

vv = np.array(list(vv0[:-1])+list(vv1[:-1])+list(vv2[:-1]))
ds=np.array(list(ds0)+list(ds1)+list(ds2))
popt0, _ = curve_fit(fctmod, vv, ds)


# Générer des valeurs ajustées pour le tracé
x_fit0 = np.linspace(0, 65, 100)


y_fit0 = fctmod(x_fit0, *popt0)


# Tracé des résultats
plt.figure(figsize=(10, 6))

plt.scatter(vv0[:-1], ds0, label="Données v''=0", color='red')
plt.scatter(vv1[:-1], ds1, label="Données v''=1", color='green')
plt.scatter(vv2[:-1], ds2, label="Données v''=2", color='blue')
plt.plot(x_fit0, y_fit0, linestyle="--", color='red')



plt.xlabel("Valeurs de v''")
plt.ylabel("ds (écarts)")
plt.title("Ajustement des données avec la fonction choisie")
plt.xlim([0,65])
plt.ylim([0,140])
plt.legend()
plt.grid()
plt.show()

print(popt0, popt0[0]*popt0[1])
# Utiliser les paramètres trouvés
we, xe = popt0

# Trouver la racine f(v) = 0
root = fsolve(fctmod, x0=10, args=(we, xe))[0]  # x0 est une estimation initiale

# Calculer l'intégrale de 0 à root
integral_value, error = quad(fctmod, 0, root, args=(we, xe))

print(f"Racine (v où f(v) = 0) : {root}")
print(f"Valeur de l'intégrale de 0 à {root} : {integral_value}")

De=1/2*we-1/4*we*xe+integral_value

print(De)