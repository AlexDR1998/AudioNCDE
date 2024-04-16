from trainer import Trainer

T = Trainer(128,"../Data/musdb18",BATCHES=32,LENGTH=0.01)
T.train(100)