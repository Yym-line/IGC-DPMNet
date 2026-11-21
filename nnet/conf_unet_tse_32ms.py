fs = 8000
chunk_len = 4  # (s)
chunk_size = chunk_len * fs 

nnet_conf = {
    "sr": 8000,                  
    "win_size": 256,              
    "fft_len": 256,               
    "enc_dim": 256,               
    "bottleneck_dim": 64,         
    "hidden_dim": 128,            
    "nhead": 4,                  
    "num_layer": 6,               
    "win_type": 'hann'            
}


# data configure:
train_dir = "/node/yym/node/CIE-mDPTNet_8/data/train/"
dev_dir = "/node/yym/node/CIE-mDPTNet_8/data/dev/"

train_data = {
    "mix_scp": train_dir + "mix_both.scp", 
    "ref_scp": train_dir + "ref.scp",
	"aux_scp": train_dir + "auxs1.scp",
    "sample_rate": fs,
}

dev_data = { 
    "mix_scp": dev_dir + "mix_both.scp", 
    "ref_scp": dev_dir + "ref.scp",
	"aux_scp": dev_dir + "auxs1.scp",
    "sample_rate": fs,
}

# trainer config
adam_kwargs = {
    "lr": 5e-4, 
    "weight_decay": 1e-5, 
}

trainer_conf = {
    "optimizer": "adam", 
    "optimizer_kwargs": adam_kwargs, 
    "min_lr": 1e-8, 
    "patience": 5,
    "factor": 0.5, 
    "logging_period": 200  
}
