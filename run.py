import sys

from trainer import Trainer
from model.audio_demixer_forced import AudioDemixer as AudioDemixer_forced
from model.update_first_order_forced import Func as Func_1_forced
from model.update_second_order_forced import Func as Func_2_forced
from model.audio_demixer_control import AudioDemixer as AudioDemixer_control
from model.update_first_order_control import Func as Func_1_control
from model.update_second_order_control import Func as Func_2_control



index=int(sys.argv[1])-1
label = 1
iters = 1000
LENGTH = 0.01
BATCHES = 4
latent_size = 128

if index==0:
    f = Func_1_control(1,latent_size=latent_size)
    model = AudioDemixer_control(f)
    T = Trainer(model,"../Data/musdb18",filename="audemix_1ord_control_"+str(label),BATCHES=BATCHES,LENGTH=LENGTH,MODE=0)
    T.train(iters)

if index==1:
    f = Func_2_control(1,latent_size=latent_size)
    model = AudioDemixer_control(f)
    T = Trainer(model,"../Data/musdb18",filename="audemix_2ord_control_"+str(label),BATCHES=BATCHES,LENGTH=LENGTH,MODE=0)
    T.train(iters)

if index==2:
    f = Func_1_forced(2,latent_size=latent_size)
    model = AudioDemixer_forced(f)
    T = Trainer(model,"../Data/musdb18",filename="audemix_1ord_forced_"+str(label),BATCHES=BATCHES,LENGTH=LENGTH,MODE=1)
    T.train(iters)


if index==3:
    f = Func_2_forced(2,latent_size=latent_size)
    model = AudioDemixer_forced(f)
    T = Trainer(model,"../Data/musdb18",filename="audemix_2ord_forced_"+str(label),BATCHES=BATCHES,LENGTH=LENGTH,MODE=1)
    T.train(iters)