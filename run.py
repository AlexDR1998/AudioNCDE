from trainer import Trainer

T = Trainer(128,"../Data/musdb18",BATCHES=128,LENGTH=1.0)
T.train(1000)