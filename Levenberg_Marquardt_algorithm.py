import numpy as np

class Levenberg_Marquardt:
    """
    Represents the Levenberg-Marquardt algorithm for adjust, from least-square method, an general function
    """

    def __init__(self, f, grad_f, t, y, beta, alpha = None, var = None, psi = 0.1, e = 0.0001, i = 100):
        """
        Define the class attributes, receiving the external params

        :param f: the function to be valued.
        :type f: function.
        :param grad_f: the gradient-function of f.
        :type grad_f: function.
        :param t: an list containing the abscissa inputs.
        :type t: list.
        :param y: list containing the ordinates to be adjusted.
        :type y: list.
        :param beta: 1D array whose elements are the f function params to be adjusted.
        :type beta: numpy array.
        :param alpha: non-adjustable parameters.
        :type alpha: numpy array.
        :param var: the ordinate corresponding variance.
        :type var: float or numpy array.
        :param psi: the damping factor of the algorithm.
        :type psi: float.
        :param e: square deviation precision to stop the algorithm iteration.
        :type e: float.
        :param i: maximum number of iterations.
        :type i: int.
        """

        self.f = f
        self.grad_f = grad_f
        self.t = np.array([t]).T
        self.y = np.array([y]).T
        self.alpha = alpha
        self.psi = psi
        self.chi_square = 0
        self.m = np.size(self.t)
        self.n = len(beta)
        self.e = e
        self.i = i
        self.beta = np.array(beta)

        if self.alpha is not None:
            self.alpha = np.array(self.alpha)

        if var is None:
            self.W = np.identity(self.m)
        elif isinstance(var, float) or (isinstance(var, np.ndarray) and var.ndim == 1 or var.ndim == 2):
            self.W = np.identity(self.m)*1/var
    
    def F(self, beta = None):
        """
        Array of f aplied to all t.

        :param beta: f parameter column vector.
        :type beta: 1D numpy array.
        """

        if beta is None:
            beta = self.beta
        if self.alpha is None:
            return np.array([[self.f(self.t[i, 0], beta) for i in range(self.m)]]).T
        return np.array([[self.f(self.t[i, 0], beta, self.alpha) for i in range(self.m)]]).T
    
    def J(self, beta = None):
        """
        Jacobian matrix of f aplied in beta.

        :param beta: f parameter column vector.
        :type beta: 1D numpy array.
        """
        
        if beta is None:
            beta = self.beta
        if self.alpha is None:
            return np.array([self.grad_f(self.t[i, 0], beta) for i in range(self.m)])
        return np.array([self.grad_f(self.t[i, 0], beta, self.alpha) for i in range(self.m)])
    
    def Vp(self, beta = None):
        """
        Calculates the parameter covariance matrix for a given beta.

        :param beta: f parameter column vector.
        :type beta: numpy array.
        """

        if beta is None:
            beta = self.beta
        J = self.J(beta)
        return np.linalg.inv(J.T @ self.W @ J)

    def minimization(self):
        """
        Minimization method of deviation squares.
        """
        
        delta = np.empty(self.n)
        previous_chi_square = 0
        i = 0
        while not self.criteria(i, previous_chi_square):
            previous_chi_square = self.chi_square
            F = self.F( )
            J = self.J( )
            v = J.T @ self.W @ J
            delta = np.linalg.inv(v + self.psi*np.diag(np.diag(v))) @ J.T @ self.W @ (self.y - F)
            chi = (self.y - F - J @ delta)
            self.chi_square = chi.T @ self.W @ chi
            self.beta = self.beta + delta.flatten( )
            i += 1
        return self.beta

    def criteria(self, i, previous_chi_square):
        """
        Stop criterion function.

        :param i: number of the iteration.
        :type i: int.
        :param previous_chi_square: the square deviation of previous iteration.
        :type previous_chi_square: float.
        """

        return self.e < self.chi_square - previous_chi_square < self.e or i > self.i