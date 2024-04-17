import jax.numpy as np
import equinox as eqx
import jax
import time

class Func(eqx.Module):
    F: list
    G: list
    Demix: list
    Fourier: list 
    latent_size: int
    def __init__(self,channels=1,latent_size=2048,key=jax.random.PRNGKey(int(time.time()))):
        key1,key2,key3,key4,key5,key6 = jax.random.split(key,6)
        # Time evolution of dynamical system, driven by input mixed audio
        # TODO: Add random fourier modes of time as a channel?
        self.latent_size = latent_size
        @jax.jit
        def sintanh(x,N=1):
            return np.sin((-1)**N*(2*N+1)*np.pi/2*jax.nn.tanh(x))

        self.F = [
            eqx.nn.Linear(in_features=2*latent_size,out_features=latent_size,key=key),
            lambda x:sintanh(x,1),
            #jax.nn.relu,
            #np.sin,
            eqx.nn.Linear(in_features=latent_size,out_features=latent_size,key=key),
            jax.nn.tanh
        ]
        self.G = [
            eqx.nn.Linear(in_features=2*latent_size,out_features=latent_size,key=key),
            lambda x:sintanh(x,1),
            #jax.nn.relu,
            #np.sin,
            eqx.nn.Linear(in_features=latent_size,out_features=latent_size,key=key),
            jax.nn.sigmoid
        ]
        # From time evolving latent space, project down to the de-mixed individual audio channels
        self.Demix = [
            eqx.nn.Linear(in_features=latent_size,out_features=channels,key=key),
            jax.nn.tanh
        ]
        self.Fourier = [
            eqx.nn.Linear(in_features=1,out_features=latent_size,key=key),
            np.sin
        ]
        scale = 1.0
        w_f1 = jax.random.normal(key1,shape=(latent_size,2*latent_size)) * np.sqrt(2/(2*latent_size)) * scale
        w_f2 = jax.random.normal(key2,shape=(latent_size,latent_size)) * np.sqrt(2/(latent_size)) * scale
        w_g1 = jax.random.normal(key3,shape=(latent_size,2*latent_size)) * np.sqrt(2/(2*latent_size)) * scale
        w_g2 = jax.random.normal(key4,shape=(latent_size,latent_size)) * np.sqrt(2/(latent_size)) * scale
        w_d1 = jax.random.normal(key5,shape=(channels,latent_size)) * np.sqrt(2/(latent_size)) * scale
        w_fourier1 = jax.random.normal(key6,shape=(latent_size,1)) * np.sqrt(2) * scale

        b_demix = np.zeros((channels))
        b_latent = np.zeros((latent_size))
        b_time = np.zeros((1))


        w_where = lambda l: l.weight
        b_where = lambda l: l.bias
        
        self.F[0] = eqx.tree_at(w_where,self.F[0],w_f1)
        self.F[0] = eqx.tree_at(b_where,self.F[0],b_latent)
        self.F[2] = eqx.tree_at(w_where,self.F[2],w_f2)
        self.F[2] = eqx.tree_at(b_where,self.F[2],b_latent)
        
        self.G[0] = eqx.tree_at(w_where,self.G[0],w_g1)
        self.G[0] = eqx.tree_at(b_where,self.G[0],b_latent)
        self.G[2] = eqx.tree_at(w_where,self.G[2],w_g2)
        self.G[2] = eqx.tree_at(b_where,self.G[2],b_latent)

        self.Demix[0] = eqx.tree_at(w_where,self.Demix[0],w_d1)
        self.Demix[0] = eqx.tree_at(b_where,self.Demix[0],b_demix)

        self.Fourier[0] = eqx.tree_at(w_where,self.Fourier[0],w_fourier1)
        self.Fourier[0] = eqx.tree_at(b_where,self.Fourier[0],b_time)
    @eqx.filter_jit
    def __call__(self,t,y,args):

        """ Gradient update for 2nd order gated neural ODE

        Args:
            t f32: time
            y f32[latent_size]: latent space 
            args (_type_): Empty, just for format of diffrax.diffeqsolve
        Returns:
            dy/dt f32[latent_size]: derivative of y with respect to t at time t
        """
        t = t[np.newaxis]
        for L in self.Fourier:
            t = L(t)
        
        g = np.concatenate((y,t))
        f = np.concatenate((y,t))
        
        for L in self.F:
            f = L(f)
        for L in self.G:
            g = L(g)
        
        return g*(f-y)
    def _project_demix(self,y):
        # Project out of latent space back to audio channels
        for L in self.Demix:
            y = L(y)
        return y