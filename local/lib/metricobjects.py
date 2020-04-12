import numpy as np
import tensorflow as tf
import itertools
import sympy
from progressbar import progressbar as pbar
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from einsteinpy.symbolic import MetricTensor, ChristoffelSymbols, RiemannCurvatureTensor, RicciTensor

class SyModel:
    def __call__(self, input_vectors):
        u = input_vectors
        assert u.shape[1] == len(self.input_vars), "incorrect number of dims, expected %d but got %d"%(len(self.input_vars), u.shape[1])
        
        r = [[output_var.subs(self.subs_vars(ui)) \
                for output_var in self.output_vars] \
                for ui in input_vectors]
        return np.array(r).astype(float)
    
    def subs_vars(self, input_vector):
        return {input_var:val for input_var,val in zip(self.input_vars, input_vector)}

    def apply(self, expression, input_vectors):
        r = [expression.subs(self.subs_vars(i)) for i in input_vectors]
        r = [sympy.lambdify([], i, "numpy")() for i in r]
        return np.r_[r]        
         
class SympyMetricObjects:
    
    def __init__(self, model):
        self.model = model

    def get_jacobian(self, input_vectors=None):

        x = self.model.output_vars
        u = self.model.input_vars

        J = sympy.Matrix([[xi.diff(ui) for ui in u] for xi in x])
        
        if input_vectors is not None:            
            J = self.model.apply(J, input_vectors)

        return J

    def get_metric_tensor(self, input_vectors=None):
        J = self.get_jacobian()
        me = (J.T*J)
        me.simplify()
        me = MetricTensor(me.as_immutable(), self.model.input_vars)
        
        if input_vectors is not None:            
            me = self.model.apply(me.tensor(), input_vectors)
        
        return me

    def get_inverse_metric_tensor(self, input_vectors=None):
        G = self.get_metric_tensor().tensor()
        G_ = MetricTensor(sympy.Matrix(G).inv().as_immutable(), self.model.input_vars)

        if input_vectors is not None:            
            G_ = self.model.apply(G_.tensor(), input_vectors)
        
        return G_


    def get_christoffel_symbols(self, input_vectors=None):
        """
        returns the christoffel symbols in the standard order:
            `cf[i,j,k]`= $\Gamma^i_{jk}$
        einsteinpy is used to obtain the tensor
        """
        cf = ChristoffelSymbols.from_metric(self.get_metric_tensor())

        if input_vectors is not None:            
            cf = self.model.apply(cf.tensor(), input_vectors)
        
        return cf

    def get_riemann_curvature_tensor_from_einsteinpy(self, input_vectors=None):
        """
        returns the riemman curvature tensor of second order in the standard order:
            `rc[i,j,k,m]`= $R^i_{jkm}$

        einsteinpy is used to obtain the tensor
        """
        rc = RiemannCurvatureTensor.from_christoffels(self.get_christoffel_symbols()).tensor()

        # reorder tensor in standard way
        rrc = sympy.MutableDenseNDimArray.zeros(*rc.shape)
        for i,j,k,m in itertools.product(*[range(len(self.model.input_vars))]*4):
            rrc[i,j,k,m] = rc[i,m,k,j]
        rc = rrc

        if input_vectors is not None:            
            rc = self.model.apply(rc.as_immutable(), input_vectors)

        return rc


    def get_riemann_curvature_tensor(self, input_vectors=None):
        """
        returns the riemman curvature tensor of second order in the standard order:
            `rc[i,j,k,m]`= $R^i_{jkm}$

        the tensor is built using the standard formulate by assembling its elements
        """
        
        sdc = self.get_dc()
        sc  = self.get_christoffel_symbols()

        nn = len(self.model.input_vars)
        rrc = sympy.MutableDenseNDimArray.zeros(*[nn]*4)
        for i,j,k,m in itertools.product(*[range(2)]*4):
            rr = sdc[k][i,j,m] - sdc[m][i,j,k] + sum([sc[i,z,k]*sc[z,j,m] for z in range(nn)]) - sum([sc[i,z,m]*sc[z,j,k] for z in range(nn)])
            rrc[i,j,k,m] = rr

        if input_vectors is not None:
            rrc = self.model.apply(rrc.as_immutable(), input_vectors)

        return rrc

    def get_dg(self, input_vectors=None):
        """
        returns the derivatives of the metric tensor wrt input variables, in the following order:
            `dg[k,i,j]`= $\partial g_{ik} / \partial u^k$
        """
        # derivatives of the metric tensor wrt input variables
        G  = self.get_metric_tensor().tensor()
        dg = [G.diff(i) for i in self.model.input_vars]

        if input_vectors is not None:
            dg = np.r_[[self.model.apply(i, input_vectors) for i in dg]].swapaxes(0,1)
        return dg          

    def get_dc(self, input_vectors=None):
        """
        returns the derivatives of the christoffel symbols wrt input variables, in the following order:
            `cf[m,i,j,k]`= $\partial \Gamma^i_{jk} / \partial u^m$
        """        
        cf = self.get_christoffel_symbols().tensor()
        dcf =  [cf.diff(i) for i in self.model.input_vars]
        
        if input_vectors is not None:
            dcf = np.r_[[self.model.apply(i, input_vectors) for i in dcf]].swapaxes(0,1)
            
        return dcf

    def get_ricci_tensor(self, input_vectors=None):
        sG = self.get_metric_tensor()
        rt = RicciTensor.from_metric(sG)
        if input_vectors is not None:
            rt = self.model.apply(rt.tensor(), input_vectors)

        return rt

    def get_ricci_scalar(self, input_vectors=None):
        sG_ = self.get_inverse_metric_tensor()
        srt = self.get_ricci_tensor()        
        rs = sum(sympy.Matrix(srt.tensor())*sympy.Matrix(sG_.tensor()))
        if input_vectors is not None:
            rs = self.model.apply(rs, input_vectors)

        return rs

class TensorFlowMetricObjects:
    
    def __init__(self, model):
        self.model = model

    @tf.function
    def get_jacobian(self, input_tensor):
        with tf.GradientTape(persistent=True) as gtape:
            tx = self.model(input_tensor)
        tJ = gtape.batch_jacobian(tx, input_tensor)
        return tJ

    @tf.function
    def get_metric_tensor(self, input_tensor):
        tJ = self.get_jacobian(input_tensor)
        tG = tf.stack([tf.matmul(tf.transpose(tJ[i]), tJ[i]) for i in range(len(tJ))])
        return tG

    @tf.function
    def get_inverse_metric_tensor(self, input_tensor):
        tG = self.get_metric_tensor(input_tensor)
        tG_ = tf.linalg.pinv(tG)
        return tG_

    @tf.function
    def get_hessian(self, input_tensor):
        with tf.GradientTape(persistent=True) as t1:
            _tJ = self.get_jacobian(input_tensor)
        tH = t1.batch_jacobian(_tJ, input_tensor)
        return tH
    
    @tf.function
    def get_dg(self, input_tensor):
        # compute derivatives of metric tensor wrt to input
        with tf.GradientTape(persistent=True) as gtape2:
            tG  = self.get_metric_tensor(input_tensor)
                
        tdg = gtape2.batch_jacobian(tG, input_tensor, experimental_use_pfor = False)
        tdg = tf.transpose(tdg, [0,3,2,1])
        return tdg

    @tf.function
    def get_dc(self, input_tensor):
        # compute derivatives of christoffel symbols wrt input
        nb_tries,max_tries = 0,2
        while nb_tries <= max_tries:
            nb_tries += 1
            try:
                with tf.GradientTape(persistent=True) as gtape2:
                    cf = self.get_christoffel_symbols(input_tensor)
                    
                tdC = gtape2.batch_jacobian(cf, input_tensor, experimental_use_pfor = False)
                tdC = tf.transpose(tdC, [0, 4, 1, 2, 3])   
                return tdC     
            except Exception as e:
                print("tf failed, retrying")
                if nb_tries == max_tries:
                    raise e


    @tf.function
    def get_christoffel_symbols(self, input_tensor):
        dG = self.get_dg(input_tensor)
        G_ = self.get_inverse_metric_tensor(input_tensor)
        cf = []

        for i in range(input_tensor.shape[0]):
            t =  tf.transpose(dG[i],[1,0,2])
            t += tf.transpose(dG[i],[1,2,0])
            t -= tf.transpose(dG[i],[0,1,2])

            cf.append(0.5*tf.tensordot(G_[i], t, 1))

        r = tf.stack(cf)
        return r  

    @tf.function
    def get_riemann_curvature_tensor(self, input_tensor):
        dc = self.get_dc(input_tensor)
        c  = self.get_christoffel_symbols(input_tensor)

        rrc = []
        nn = input_tensor.shape[0]
        ndims = c.shape[1]
        for n in range(nn):
            nrc, dci, ci = [], dc[n], c[n]
            for i,j,k,m in itertools.product(*[range(ndims)]*4):
                rr = dci[k][i,j,m] - dci[m][i,j,k] + sum([ci[i,z,k]*ci[z,j,m] for z in range(ndims)]) - sum([ci[i,z,m]*ci[z,j,k] for z in range(ndims)])
                nrc.append(rr)
            rrc.append(tf.reshape(tf.stack(nrc), shape=[ndims]*4))
        rrc = tf.stack(rrc)        

       
        return rrc

    @tf.function
    def get_ricci_tensor(self, input_tensor):
        rc = self.get_riemann_curvature_tensor(input_tensor)
        return tf.einsum('...ijim',rc)

    @tf.function
    def get_ricci_scalar(self, input_tensor):
        tG_ = self.get_inverse_metric_tensor(input_tensor)
        rt  = self.get_ricci_tensor(input_tensor)
        rs = tf.einsum('...ij,...ij->...',  tG_, rt)
        return rs

    def plot_curvature_for_2D_input_flat_manifold(self, x_range=[-1,1], y_range=[-1,1], n_points=1000, batch_size = 20, remove_percentile=3):
        """
        plots the curvature of the output manifold of transformed 2D data.
        the output manifold can have any dimension, but curvature will be plotted
        in the flat input 2D space.
        """
        assert self.model.input.shape[1]==2, "model must have 2D input only"
        from sklearn.preprocessing import MinMaxScaler
        x_range[0] -= np.abs(x_range[0])*.2
        x_range[1] += np.abs(x_range[1])*.2
        y_range[0] -= np.abs(y_range[0])*.2
        y_range[1] += np.abs(y_range[1])*.2
        z = []
        xy = []
        for i in pbar(range(n_points//batch_size)):
            data = (np.random.random(size=(batch_size, 2))-.5)*10
            data[:,0] = MinMaxScaler(feature_range=x_range).fit_transform(data[:,0].reshape(-1,1))[:,0]
            data[:,1] = MinMaxScaler(feature_range=y_range).fit_transform(data[:,1].reshape(-1,1))[:,0]
            tu = tf.Variable(data, dtype=np.float32)
            try:
                r = self.get_ricci_scalar(tu).numpy()
                z.append(r)
                xy.append(tu.numpy())
            except:
                # first call might fail in TF
                pass
        z = np.r_[z].flatten()

        # remove x% on top and bottom of distribution since outliers distort graphs
        zmin,zmax = np.percentile(z, [remove_percentile,100-remove_percentile])
        keep_idxs = (z>=zmin)&(z<=zmax)
        z = z[keep_idxs]

        xy = np.vstack(xy)[keep_idxs]
        x,y = xy[:,0], xy[:,1]


        fig = plt.figure(figsize=(13,3.5))
        
        ax1 = plt.subplot(121)
        ax1.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
        cntr1 = ax1.tricontourf(x, y, z, levels=14, cmap="RdBu_r")
        fig.colorbar(cntr1, ax=ax1)
        ax1.plot(x, y, 'ko', ms=3, alpha=.1)
        ax1.set_title("Ricci curvature in transformed data space")

        ax2 = plt.subplot(122)
        ax2.hist(z, bins=30);
        ax2.set_title("distribution of Ricci curvature")
        return ax1