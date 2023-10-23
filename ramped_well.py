import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


"""
Approximate Energy Eigenvalues
-----
E_1 ~ 0.5005675888061523
E_2 ~ 0.872
E_3 ~ 1.172

E_2 ~ 1.4799341735839842
E_3 ~ 2.2882755661010736
E_4 ~ 4.1494517059326155

ΔE = 
""" 

# CONSTANTS
a = 3 #[nm]
b = 0.5  #[eV / nm]
mc2 = 0.511e6 #[eV]
hc = 197.3 #[eV nm]
    
# Better to keep big array generations out of loops
dx = 0.01 #[nm]
x = np.arange(0, a + dx, dx)


def bisection(func:callable, E_l:float, E_u:float) -> tuple:
    """
    Finds roots of `generate_psi()` given two points on either side of a root.

    parameters
    -----------
    func : function
    Usually `generate_psi(E)`

    E_l : float
    Interval lower bound

    E_u : float
    Interval upper bound
    """
    threshold = 5e-7
    mid = (E_l + E_u) / 2 
    psi_lower = func(E_l)
    psi_upper = func(E_u)
    psi_mid = func(mid)

    if abs(E_u - E_l) <= threshold:
        print('Root Constrained')
        return mid, E_l, E_u
    elif psi_mid * psi_lower < 0:
        return bisection(func, E_l, mid)
    elif psi_mid * psi_upper < 0:
        return bisection(func, mid, E_u)
    else:
        print('Root Not Found')
        return None


def secantDescent(func:callable, E_0:float, E_1:float) -> float:
    """
    Finds root of `generate_psi()`, given an Energy using the secant descent method

    parameters
    ------------
    E : float
        Energy value for generating psi(x)[-1]

    returns
    -------
    E_c : float
        'Convergent' Energy eigenvalue
    """
    max_iter = 30   # Maximum number of iterations
    iter = 0
    threshold = 1e-13
    E_p = E_0
    E_c = E_1
    f_p = func(E_p)
    f_c = func(E_c)
    E_n = None
        
    while abs(E_p - E_c) >= threshold:
        if abs(E_p - E_c) <= threshold:
            print(f'Root Constrained After {iter} iterations')
            break
        E_n = (f_c * E_p - f_p * E_c) \
            / (f_c - f_p)
        f_n = func(E_n)
        # Tuple assignment update of variables
        E_p, E_c, f_p, f_c = E_c, E_n, f_c, f_n
        iter += 1
        # print(f"Energy: {E_n}")
    return E_n

def g(E:float, x:float) -> float:
    """
    Helper function for `generate_psi()`
    """
    return -((2 * mc2) / (hc)**2) * (E - b*x)


def generatePsiEndpoint(E_guess:float) -> float:
    """
   Generate a wave function solution, given an Energy value. 
    Returns the wave function value at the end of the array at psi(a)

    parameters
    -----------
    E_guess : float
    Initial guess of energy eigenvalue

    returns
    -------
    psi_n : float
        Value of psi(a) 'the wave function evaluated at the endpoint: a'
    """
    psi_p = 0       #psi(x-1)
    psi_c = 1e-6    #psi(x)

    # Numerical Integration (carrying only 3 values)
    for pos in x[2:]:
        psi_n = 2 * psi_c - psi_p + dx**2 * g(E_guess, pos) * psi_c
        psi_p = psi_c
        psi_c = psi_n
        # print(f'{pos:.3e}, {psi_n:.3e}')
    
    return psi_n


def generatePsiArray(E:float) -> np.ndarray:
    """Generate entire wave function given a likely Energy eigenvalue
    
    parameters
    ----------
    E : float
        Energy (possibly energy eigenvalue) of state psi

    returns
    -------
    array_like
        Array of psi(x)
    """
    psi_p = 0       #psi(x-1)
    psi_c = 1e-6    #psi(x)
    psi = np.zeros_like(x)
    psi[0], psi[1], = psi_p, psi_c

    # Numerical Integration (carrying only 3 values)
    for i, pos in enumerate(x[2:]):
        psi_n = 2 * psi_c - psi_p + dx**2 * g(E, pos) * psi_c
        psi_p = psi_c
        psi_c = psi_n
        psi[2+i] = psi_n

    return psi


def visualRootFinding(Energies:np.ndarray):
    """Interactively plot range of `psi(x)` for different Energy values
    to find eigenstates by iterative visual inspection.
    'Guess & Check'
    
    parameters
    ----------
    Energies : ndarray
        Range of possible Energy eigenvalues
    """
    mpl.interactive(True)
    mpl.use('TkAgg')
    scale = 2.5
    
    fig, ax = plt.subplots(1, figsize=(scale*8, scale*5))
    line_styles = ['-', '--', '-.', ':']  # Define different line styles
    markers = ['o', 's', 'D', 'v', '^', '.']  # Define different markers
    line_widths = [1, 2, 1.5, 2.5]  # Define different line widths
    
    for i, E in enumerate(Energies):
        psi = generatePsiArray(E)
        style = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]
        width = line_widths[i % len(line_widths)]
        label = f'{E:.3e}'
        ax.plot(x, psi, linestyle=style, marker=marker, markersize=4, lw=width, alpha=0.6, label=label)
    
    ax.axhline(y=0, xmin=0, xmax=3, color='r', linestyle='--', alpha=0.8)
    ax.axvline(x=0, color='k')
    ax.axvline(x=3, color='k')
    plt.title('Visual Root Finding of $\psi_i$ Eigenstates')
    plt.xlabel('x [nm]')
    plt.ylabel('$\psi(x) [m^{-1/2}]$')
    plt.ylim(-3.5e-5, 3.5e-5)
    fig.legend()
    plt.show()


def normalisePsi(psi:np.ndarray) -> np.ndarray:
    """Normalise |psi|^2 to find normalisation factors `A^2` and `A` 
    for the square modulus of psi and psi, respectively
    
    parameters
    ----------
    psi : ndarray
        Array of psi(x) values to be normalised 
    
    returns
    -------
    psi_normalised : ndarray
        Normalised array of psi(x) values
    """
    psiSquared = np.exp(psi)
    integralPsiSquared = dx * (np.sum(psiSquared[1: -1] + (psiSquared[0] + psiSquared[-1]) / 2)) #numerical approx. of integral
    A = 1 / np.sqrt(integralPsiSquared)

    return A * psi

def positionExpectation():
    #TODO
    ...


def momentumExpectation():
    #TODO but should be zero for the stationary states!
    ...
    

def plotGeneratePsiEndpoint():
    """Plots the function `generatePsiEndpoint` to visually inspect the distribution of roots along the E-axis"""
    scale=2
    E = np.linspace(0, 6, 1000)
    Y = generatePsiEndpoint(E)
    fig, ax = plt.subplots(1, figsize=(scale*8, scale*5))
    ax.plot(E, Y, 'r', label='$\psi(a)$ endpoints')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set(title='$\psi(a)$ endpoints Vs Energy', 
           xlabel='$\psi(x) [m^{-1/2}]$',
           ylabel='$\psi(a)_E$', 
           ylim=(-1e-2, 1e-2)
           )
    ax.legend()
    plt.show()


def plotPsi(psi:np.ndarray):
    """Plot psi(x) array
    
    parameters
    ----------
    array : ndarray
        Array of psi(x) values
    """
    # Shape of Potential
    l = 3
    X = np.linspace(0, a, 1000)
    V = np.where((X > 0) & (X < a), X * b, 1e-6) 
    
    fig, ax = plt.subplots(1, figsize=(8, 5))
    # ax.plot(X, V, 'k', alpha=0.6)
    ax.plot(x, psi, 'r', label=f'$\psi_1$, $E_1={0.50}$')
    ax.set(title='Ramped Infinite Square Well', 
           xlabel='x [nm]', 
           ylabel='Energy [eV]', 
        #    ylim=(0, 1e-3)
           )
    plt.show()


def main():

    # secantDescent(generatePsiEndpoint, 0.49, 0.499)
    for i, initials in enumerate([(0.49, 0.499), (1.17, 1.18), (0.873, 0.871)]):
        try:
            E = secantDescent(generatePsiEndpoint, *initials)
            print(f'Energy Eigenvalue E{i+1} = {E}')
        except TypeError:
            print(f'Eigenvalue: {(initials[0] + initials[1])/2} Not Found.')

    #plotPsi(
    #     normalisePsi(
    #         generatePsiArray(0.5005675888061523))
    #         )

    # visualRootFinding(np.linspace(1.17, 1.18, 30))
    #plotGeneratePsiEndpoint()
    


if __name__ == '__main__':
    main()
