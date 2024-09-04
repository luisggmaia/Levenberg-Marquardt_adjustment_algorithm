import numpy as np

def of(t, beta):
    """
    General function of an free oscillation.

    :param t: non-adjustable params.
    :type t numpy array.
    :param beta: array containing all the param of the function: ym (equilibrium position of the oscillation), dy (amplitude), omega (angular frequency), phi (inital phase of the oscillation).
    :type beta: numpy array.
    """

    ym, dy, omega, phi = beta
    return ym + dy*np.cos(omega*t + phi)

def grad_of(t, beta):
    """
    Gradient function of the General function of an free oscillation.

    :param t: non-adjustable params.
    :type t numpy array.
    :param beta: array containing all the param of the function.
    :type beta: numpy array.
    """

    ym, dy, omega, phi = beta
    c = np.cos(omega*t + phi)
    s = np.sin(omega*t + phi)
    return np.array([1, c, -t*dy*s, -dy*s])

def odf(t, beta):
    """
    General function of an underdamped oscillation.

    :param t: non-adjustable params.
    :type t numpy array.
    :param beta: array containing all the param of the function: ym (equilibrium position of the oscillation), dy (amplitude), gamma (gamma factor of exponential term), omega (angular frequency), phi (inital phase of the oscillation).
    :type beta: numpy array.
    """

    ym, dy, gamma, omegal, phi = beta
    return ym + dy*np.exp(-gamma*t)*np.sin(omegal*t + phi)

def grad_odf(t, beta):
    """
    Gradient funtion of the general function of an underdamped oscillation.

    :param t: non-adjustable params.
    :type t numpy array.
    :param beta: array containing all the param of the function.
    :type beta: numpy array.
    """

    ym, dy, gamma, omegal, phi = beta
    s = np.exp(-gamma*t)*np.sin(omegal*t + phi)
    c = np.exp(-gamma*t)*np.cos(omegal*t + phi)
    return np.array([1, s, -t*dy*s, t*dy*c, dy*c])

def fdof(t, beta, alpha):
    """
    General function of an driven dumped oscilation

    :param t: independent variable.
    :type t numpy array.
    :param beta: array containing all the adjustable param of the function.
    :type beta: numpy array.
    :param alpha: array containing non-adjustable params of the function.
    :type alpha: numpy array.
    """

    m, omega = alpha
    ym, gamma, F0 = beta
    g = np.sqrt(omega**2 - gamma**2)
    return ym + F0/(2*m*gamma)*(np.sin(omega*t)/omega - np.exp(-gamma*t)*np.sin(g*t)/g)

def grad_fdof(t, beta, alpha):
    """
    Gradient function of the general function of an driven dumped oscilation

    :param t: independent variable.
    :type t numpy array.
    :param beta: array containing all the adjustable param of the function.
    :type beta: numpy array.
    :param alpha: array containing non-adjustable params of the function.
    :type alpha: numpy array.
    """

    m, omega = alpha
    ym, gamma, F0 = beta
    q = F0/(2*m*gamma)
    g = np.sqrt(omega**2 - gamma**2)
    s = np.sin(omega*t)/omega
    r = np.sin(g*t)/g
    e = np.exp(-gamma*t)
    return np.array([1, q*(-1/gamma*s + ((1/gamma + gamma - gamma/g**2)*r + gamma/g**2*np.cos(g*t))*e), q/F0*(s - e*r)])