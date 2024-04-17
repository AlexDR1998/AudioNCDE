import jax.numpy as np
import equinox as eqx
import jax
import time

class Func(eqx.Module):
    F_damp: list
    F_osc: list
    G_damp: list
    G_tot: list
    Demix: list
    latent_size: int
    #Fourier: list 
    def __init__(self,channels=2,latent_size=2048,key=jax.random.PRNGKey(int(time.time()))):
        
        key1,key2,key3,key4,key5,key6,key7,key8,key9,key10 = jax.random.split(key,10)
        # Time evolution of dynamical system, driven by input mixed audio
        self.latent_size = latent_size
        @jax.jit
        def sintanh(x,N=1):
            return np.sin((-1)**N*(2*N+1)*np.pi/2*jax.nn.tanh(x))

        self.F_damp = [
            eqx.nn.Linear(in_features=latent_size+2,out_features=latent_size,key=key1),
            #lambda x:sintanh(x,1),
            jax.nn.tanh,
            eqx.nn.Linear(in_features=latent_size,out_features=latent_size//2,key=key2),
            jax.nn.relu6 # maybe saturate this with different activation?
        ]
        self.F_osc = [
            eqx.nn.Linear(in_features=latent_size+2,out_features=latent_size,key=key3),
            jax.nn.tanh,
            #lambda x:sintanh(x,1),
            eqx.nn.Linear(in_features=latent_size,out_features=latent_size//2,key=key4),
            jax.nn.relu # maybe saturate this with different activation?
        ]
        self.G_damp = [
            eqx.nn.Linear(in_features=latent_size+2,out_features=latent_size,key=key5),
            jax.nn.relu,
            eqx.nn.Linear(in_features=latent_size,out_features=latent_size//2,key=key6),
            jax.nn.sigmoid
        ]

        self.G_tot = [
            eqx.nn.Linear(in_features=latent_size+2,out_features=latent_size,key=key7),
            jax.nn.relu,
            eqx.nn.Linear(in_features=latent_size,out_features=latent_size//2,key=key8),
            jax.nn.sigmoid
        ]



        # From time evolving latent space, project down to the de-mixed individual audio channels
        self.Demix = [
            eqx.nn.Linear(in_features=latent_size,out_features=channels,key=key9),
            jax.nn.tanh
        ]
        scale = 1.0
        w_fd1 = jax.random.normal(key1,shape=(latent_size,latent_size + 2)) * np.sqrt(2/(latent_size + 2)) * scale
        w_fd2 = jax.random.normal(key2,shape=(latent_size//2,latent_size)) * np.sqrt(2/(latent_size)) * scale
        w_fo1 = jax.random.normal(key3,shape=(latent_size,latent_size + 2)) * np.sqrt(2/(latent_size + 2)) * scale
        w_fo2 = jax.random.normal(key4,shape=(latent_size//2,latent_size)) * np.sqrt(2/(latent_size)) * scale
        
        
        w_gd1 = jax.random.normal(key5,shape=(latent_size,latent_size + 2)) * np.sqrt(2/(latent_size + 2)) * scale
        w_gd2 = jax.random.normal(key6,shape=(latent_size//2,latent_size)) * np.sqrt(2/(latent_size)) * scale
        w_gt1 = jax.random.normal(key7,shape=(latent_size,latent_size + 2)) * np.sqrt(2/(latent_size + 2)) * scale
        w_gt2 = jax.random.normal(key8,shape=(latent_size//2,latent_size)) * np.sqrt(2/(latent_size)) * scale
        
        w_d1 = jax.random.normal(key9,shape=(channels,latent_size)) * np.sqrt(2/(latent_size)) * scale
        

        b_demix = np.zeros((channels))
        b_latent = np.zeros((latent_size))
        b_latent_out = np.zeros((latent_size//2))
        


        w_where = lambda l: l.weight
        b_where = lambda l: l.bias
        
        self.F_damp[0] = eqx.tree_at(w_where,self.F_damp[0],w_fd1)
        self.F_damp[0] = eqx.tree_at(b_where,self.F_damp[0],b_latent)
        self.F_damp[2] = eqx.tree_at(w_where,self.F_damp[2],w_fd2)
        self.F_damp[2] = eqx.tree_at(b_where,self.F_damp[2],b_latent_out)
        
        self.F_osc[0] = eqx.tree_at(w_where,self.F_osc[0],w_fo1)
        self.F_osc[0] = eqx.tree_at(b_where,self.F_osc[0],b_latent)
        self.F_osc[2] = eqx.tree_at(w_where,self.F_osc[2],w_fo2)
        self.F_osc[2] = eqx.tree_at(b_where,self.F_osc[2],b_latent_out)



        self.G_damp[0] = eqx.tree_at(w_where,self.G_damp[0],w_gd1)
        self.G_damp[0] = eqx.tree_at(b_where,self.G_damp[0],b_latent)
        self.G_damp[2] = eqx.tree_at(w_where,self.G_damp[2],w_gd2)
        self.G_damp[2] = eqx.tree_at(b_where,self.G_damp[2],b_latent_out)



        self.G_tot[0] = eqx.tree_at(w_where,self.G_tot[0],w_gt1)
        self.G_tot[0] = eqx.tree_at(b_where,self.G_tot[0],b_latent)
        self.G_tot[2] = eqx.tree_at(w_where,self.G_tot[2],w_gt2)
        self.G_tot[2] = eqx.tree_at(b_where,self.G_tot[2],b_latent_out)

        self.Demix[0] = eqx.tree_at(w_where,self.Demix[0],w_d1)
        self.Demix[0] = eqx.tree_at(b_where,self.Demix[0],b_demix)

        
        
    @eqx.filter_jit
    def __call__(self,t,y,args):
        """ Gradient update for 2nd order gated neural ODE

        Split up y into position and velocity terms

        Args:
            t f32: time
            y f32[latent_size]: latent space encoding position ([:latent_size//2]) and velocity ([latent_size//2:])
            args diffrax.AbstractPath subclass: an interpolation of the input data to be used as a forcing term
                use args.evaluate(t) to return value of input at continuous t

        Returns:
            dy/dt f32[latent_size]: _description_
        """


        forcing_signal = args.evaluate(t)


        P = y[:self.latent_size//2]
        V = y[self.latent_size//2:]
        
        y_forced = np.concatenate((y,forcing_signal))
        damp = y_forced
        g_damp = y_forced
        osc = y_forced
        g_tot = y_forced
        
        for L in self.F_damp:
            damp = L(damp)
        for L in self.G_damp:
            g_damp = L(g_damp)
        for L in self.F_osc:
            osc = L(osc)
        for L in self.G_tot:
            g_tot = L(g_tot)        
        
        dP = V
        dV = -(V*damp*g_damp + P*osc)#*g_tot 

        #dP = jax.nn.tanh(V)
        #dV = -(V*damp + P*osc) 
        return np.concatenate((dP,dV))

        

    def _project_demix(self,y):
        # Project out of latent space back to audio channels
        for L in self.Demix:
            y = L(y)
        return y