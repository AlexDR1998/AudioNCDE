from model.audio_demixer import AudioDemixer
import jax.numpy as np
import equinox as eqx
import time
import diffrax as dfx
import jax
from tqdm import tqdm
import matplotlib.pyplot as plt
import optax
import musdb
import einops
import random
import tensorflow.summary as tfs

class Trainer(object):
    def __init__(self,latent_size,mus_path="../Data/musdb18",log_dir="logs/audemix",BATCHES=4,LENGTH=1.0):
        self.model=AudioDemixer(1,latent_size)
        self.latent_size = latent_size
        self.BATCHES = BATCHES
        self.LENGTH = LENGTH
        self.MUSDB = musdb.DB(root="../Data/musdb18/")
        self.LOG = tfs.create_file_writer(log_dir)

    def data_loader(self,key):
        """ Loads random self.BATCHES of audio with self.LENGTH in seconds.
        
        Args:
            key (jax.random.PRNGKey): JAX pseudo-random number key

        Returns:
            x f32[BATCHES,LENGTH*44100]: mixed audio from all tracks (vocal, guitar, drums etc.)
            y f32[BATCHES,LENGTH*44100]: only the vocal track
        """
        

        batch_indexes = jax.random.choice(key,len(self.MUSDB),shape=(self.BATCHES,))
        x = []
        y = []
        subkey = jax.random.fold_in(key,1)
        for b in range(self.BATCHES):
            subkey = jax.random.fold_in(subkey,b)
            track = self.MUSDB.tracks[batch_indexes[b]]
            track.chunk_duration = self.LENGTH
            pos =  jax.random.randint(key=subkey,shape=(1,),minval=0, maxval=track.duration - track.chunk_duration)
            track.chunk_start = float(pos[0])
            x.append(np.mean(track.audio,axis=1))
            y.append(np.mean(track.targets['vocals'].audio,axis=1))
        x = np.array(x)
        y = np.array(y)
        return x,y 
    
    def train(self,iters,FILENAME="models/audemix_1",key=jax.random.PRNGKey(int(time.time()))):
        
        @eqx.filter_jit # Wrap this function in JIT for speedup
        def makestep(model,audio,target,opt_state,key):
            
            @eqx.filter_value_and_grad # Compute gradients of this part
            def compute_loss(model,audio,target,key):
                v_model = jax.vmap(model,in_axes=(0,0,0),out_axes=(0,0,0)) # vmap model over batches
                N = audio.shape[1]
                ts = np.linspace(0,N,N)
                ts = einops.repeat(ts,"time -> batches time",batches=self.BATCHES)
                h0 = jax.random.normal(key,shape=(self.BATCHES,self.latent_size))
                _,_,prediction= v_model(ts,h0,audio)
                L = np.sqrt(np.mean((prediction[...,0]-target)**2))
                return L
            
            loss,grads = compute_loss(model,audio,target,key)
            updates,opt_state = self.OPTIMISER.update(grads, opt_state, model)
            model = eqx.apply_updates(model,updates)
            return model,opt_state,loss

        model = self.model
        schedule = optax.exponential_decay(1e-2, transition_steps=iters, decay_rate=0.99)
        self.OPTIMISER = optax.adam(schedule)
        model_diff,_ = model.partition()
        opt_state = self.OPTIMISER.init(model_diff)
        
        reload_every = 100 # Loading and chopping up the data is slow, only do every 100 epochs
        audio,target = self.data_loader(key)
        for i in tqdm(range(iters)):
            key = jax.random.fold_in(key,i)
            if i%reload_every==0:
                audio,target = self.data_loader(key)
            
            model,opt_state,loss = makestep(model,audio,target,opt_state,key)
            with self.LOG.as_default():
                tfs.scalar("Loss",loss,step=i)
    
        model.save(FILENAME,overwrite=True)
            
        
        #plt.plot(loss_log)
        #plt.show()
