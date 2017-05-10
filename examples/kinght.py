"""
Script to recreate the data in the table of [2] and Fig. 1 in [1]. 
"""

import matplotlib.pyplot as plt
import numpy as np

import wgm.core as wgm


def schiller():
    """Recreate the results in table 1 or 2 in [2]."""
    # Initialise wgm object (na and ns irrelevant as we will override the calculated m anyway)
    obj = wgm.WGM(na=1, ns=1, mode='TE')

    # Overide m_ratio to the value used in the tables
    obj.m_ratio = 1.363

    # Calculate and print the results
    m = 100  # Mode number
    nmax = 5  # Maximum order l to calculate up to
    print('Order l\tMode Number n={0:d}'.format(m))
    for n in range(1, nmax + 1):
        result = obj.calc_size_param(n=n, m=m)
        print('{}\t\t{}'.format(n, result))


def plot_beta_fiber():
    """Recreate just the fiber plot data in Fig.1 [1]"""
    rho_list = np.linspace(1.4, 2.8)
    beta_list = []
    for rho in rho_list:
        beta_list.append(wgm.beta_fiber(rho, lam0=1.55, N=1.44))

    fig, ax = plt.subplots()
    ax.plot(rho_list, beta_list)
    ax.set_xlabel('Fiber radius (um)')
    ax.set_ylabel('Propagation constant, beta (/um)')
    ax.set_ylim([5.4, 6])
    plt.show()


def knight():
    """Recreate Fig.1 in [1]"""

    #################################################
    # User Input
    #################################################
    # Wavelength of interest (um)
    lam0 = 1.55
    # Mode of WGM ('TE' or 'TM)
    mode = 'TM'
    # Define the material refractive index @ lam0
    n_sio2 = 1.44
    n_air = 1

    #################################################
    # Calculate the fiber
    #################################################
    rho_list = np.linspace(1.4, 2.5)  # Range of fiber radii (um)
    beta_list = []  # Initialise the list to store the beta results in
    for rho in rho_list:
        # Calculate the beta for the given rho at 1.55um and then store the result
        beta = wgm.beta_fiber(rho, lam0=lam0, N=n_sio2)
        beta_list.append(beta)

    fig, ax2 = plt.subplots()
    ax1 = ax2.twiny()
    ax1.plot(rho_list, beta_list, color='k')
    ax1.set_xlabel('Fiber radius (um)')
    ax1.set_ylabel('Propagation constant, beta (/um)')
    ax1.set_ylim([5.4, 6])

    #################################################
    # Calculate the sphere WGM
    #################################################
    obj = wgm.WGM(na=n_air, ns=n_sio2, mode=mode)  # Initialise the WGM object
    a_list = [50, 75, 100, 150, 200, 250]  # Radii of the spheres to calculate (um)
    nmax = 4  # Maximum order of n mode index to calculate up to
    for n in range(1, nmax + 1):
        beta_list = []
        for a in a_list:
            l = 2 * a * np.pi  # Approximately equal to the sphere radius as travelling close to the surface
            m = l  # Interested in modes at the equator

            x = obj.calc_size_param(n=n, m=m)
            beta = wgm.beta_wgm(lam0=lam0, l=l, x=x)
            beta_list.append(beta)
        ax2.plot(a_list, beta_list, '.', label=n)
    ax2.set_xlabel('Sphere radius (um)')
    ax2.set_ylabel('Propagation constant, beta (/um)')
    ax2.set_xlim([0, 300])
    ax2.legend(title='n')
    plt.show()


if __name__ == "__main__":
    # plot_beta_fiber()
    # test1()
    knight()
