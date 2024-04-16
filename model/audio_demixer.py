from model.abstract_model import AbstractModel
from model.update import Func
import jax.numpy as np
import equinox as eqx
import time
import diffrax as dfx
import jax

class AudioDemixer(AbstractModel):
    func: Func
    def __init__(self,channels,latent_size,key=jax.random.PRNGKey(int(time.time()))):
        self.func = Func(channels,latent_size,key)

    def __call__(self, ts, y0, input):
        coeffs = dfx.backward_hermite_coefficients(ts,input)
        control = dfx.CubicInterpolation(ts, coeffs)
        term = dfx.ControlTerm(self.func, control).to_ode()
        solution = dfx.diffeqsolve(term,
                                    dfx.Heun(),
                                    t0=ts[0],t1=ts[-1],
                                    dt0=1,
                                    y0=y0,
                                    max_steps=len(ts),
                                    #stepsize_controller=dfx.PIDController(rtol=1e-1, atol=1e-2),
                                    stepsize_controller=dfx.ConstantStepSize(),
                                    saveat=dfx.SaveAt(ts=ts))
        pfunc = jax.vmap(self.func._project_demix,in_axes=0,out_axes=0)

        return solution.ts,solution.ys,pfunc(solution.ys)
    
    	
    