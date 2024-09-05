import numpy as np

class Levenberg_Marquardt:
    """
    Represents the Levenberg-Marquardt algorithm for adjust, from least-square method, an general function
    """

    def __init__(self, f, t, y, beta, grad_f = None, alpha = None, Var = None, psi = 0.1, e = 0.1, i = 100, h = np.sqrt(np.finfo(float).eps)):
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
        :param Var: the ordinate corresponding variance.
        :type Var: float or numpy array.
        :param psi: the damping factor of the algorithm.
        :type psi: float.
        :param e: square deviation precision to stop the algorithm iteration.
        :type e: float.
        :param i: maximum number of iterations.
        :type i: int.
        :param h: the step to calculate the numerical gradient.
        :type h: float.
        """

        self.f = f
        self.grad_f = grad_f if grad_f is not None else self.set_num_grad_g(f, h)
        self.t = np.array([t]).T
        self.y = np.array([y]).T
        self.alpha = alpha
        self.psi = psi
        self.chi_square = 0
        self.C = None
        self.m = np.size(self.t)
        self.n = len(beta)
        self.e = e
        self.i = i
        self.beta = np.array(beta)

        if self.alpha is not None:
            self.alpha = np.array(self.alpha)

        if Var is None:
            self.W = np.identity(self.m)
        elif isinstance(Var, float) or (isinstance(Var, np.ndarray) and Var.ndim == 1 or Var.ndim == 2):
            self.W = np.identity(self.m)*1/Var
    
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
    
    def set_C(self, beta = None):
        """
        Calculates the parameter covariance matrix for a given beta.

        :param beta: f parameter column vector.
        :type beta: numpy array.
        :return: the parameter covariance matrix.
        :rtype: numpy array.
        """

        if beta is None:
            beta = self.beta
        J = self.J(beta)
        self.C = np.linalg.inv(J.T @ self.W @ J)
        return self.C
    
    def minimization(self):
        """
        Minimization method of deviation squares.
        """
        
        beta = self.beta
        delta = np.empty(self.n)
        chi_square = self.chi_square
        previous_chi_square = 0
        i = 0
        while not self.criteria(i, previous_chi_square, chi_square):
            previous_chi_square = chi_square
            F = self.F( )
            J = self.J( )
            v = J.T @ self.W @ J
            delta = np.linalg.inv(v + self.psi*np.diag(np.diag(v))) @ J.T @ self.W @ (self.y - F)
            chi = (self.y - F - J @ delta)
            chi_square = chi.T @ self.W @ chi
            beta += delta.flatten( )
            i += 1
        return beta, chi_square
    
    def set_min_beta(self):
        """
        Set the beta value.

        :return: the beta value.
        :rtype: numpy array.
        """

        self.beta, self.chi_square = self.minimization( )

        return self.beta

    def criteria(self, i, previous_chi_square, chi_square):
        """
        Stop criterion function.

        :param i: number of the iteration.
        :type i: int.
        :param previous_chi_square: the square deviation of previous iteration.
        :type previous_chi_square: float.
        """

        return np.abs(chi_square - previous_chi_square) < self.e*chi_square or i > self.i

    def set_num_grad_g(self, f, h = np.sqrt(np.finfo(float).eps)):
        """
        Set the numerical gradient function.

        :param f: the function to be valued.
        :type f: function.
        :param h: the step to calculate the numerical gradient.
        :type h: float.
        :return: the numerical gradient function.
        :rtype: function.
        """

        grad_f = lambda t, beta, alpha = None: (f(t*(1 + h), beta, alpha) - f(t, beta, alpha))/(t*h)

        return grad_f