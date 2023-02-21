from types import MappingProxyType

DEFAULT_CONFIG = MappingProxyType({
    "lr": 0.001,
    "batch_size": 2048,
    "microbatch": 64,
    "learning_steps": 320000,
    "log_interval": 20,
    "save_interval": 2000,
    "eval_interval": 1000,
    "ema_rate": "0.9999",
    "resume_checkpoint": "",
    "schedule_sampler": "lossaware",
    "diffusion_steps": 1000,                    # Changed
    "noise_schedule": "sqrt",
    "timestep_respacing": "",
    "vocab_size": 729,                          # Added
    "dataset": "ComMU",
    "data_dir": "datasets/ComMU-processed",
    "data_loader_workers": 2,                   # num_workers for DataLoader
    "corr_available": "mt,mn,rn,rr",            # Available corruptions - TODO: add 'at'
    "corr_max": 0,                              # Max number of corruptions
    "corr_p": 0.5,                              # Probability to choice each corruption
    # "corr_kwargs": "dict(p=0.5, count=3)",    # Keyword arguments for each corruption
    "seq_len": 2096,                            # Changed = TODO: 0
    "pretrained_embedding": "",                 # To use POZALabs' embedding, provide .pt name
    "hidden_t_dim": 128,                        # Transformer
    "hidden_dim": 128,                          # Transformer and Embedding
    "fnet_hidden_dim": 128,                     # FNet
    "fnet_intermediate_dim": 512,               # FNet
    "dropout": 0.1,
    "use_fp16": False,
    "fp16_scale_growth": 0.001,
    "seed": 102,
    "gradient_clipping": -1.0,
    "weight_decay": 0.0,
    "learn_sigma": False,
    "use_kl": False,
    "predict_xstart": True,
    "rescale_timesteps": True,
    "rescale_learned_sigmas": False,
    "sigma_small": False,
    "checkpoint_path": ".",
    "emb_scale_factor": 1.0,
    "num_fnet_layers": 6,                       # Added for FNet
    "use_attention" : False
})
