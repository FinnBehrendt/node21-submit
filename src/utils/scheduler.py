import numpy as np 
# Taken from https://www.kaggle.com/socom20/effdet-v2
def cos_decay(start_val=1.0, end_val=1e-4, steps=100):
    return lambda x: ((1 - np.cos(x * np.pi / steps)) / 2) * (end_val - start_val) + start_val

def linear_warmup(start_val=1e-4, end_val=1.0, steps=5):
    return lambda x: x / steps * (end_val - start_val) + start_val  # linear

def scheduler_lambda(lr_frac=1e-4, warmup_epochs=5, cos_decay_epochs=60):
    if warmup_epochs > 0:
        lin = linear_warmup(start_val=lr_frac, end_val=1.0, steps=warmup_epochs)
        
    cos = cos_decay(start_val=1.0, end_val=lr_frac, steps=cos_decay_epochs)
    
    def f(x):
        if x < warmup_epochs:
            return lin(x)
        
        elif x <= (warmup_epochs + cos_decay_epochs):
            return cos(x - warmup_epochs)
        
        else:
            return lr_frac
        
    return f