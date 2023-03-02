from types import MappingProxyType

DEFAULT_CONFIG = MappingProxyType({
    # # # General Settings # # #
    "lr": 0.0001,
    "batch_size": 2048,
    "microbatch": 64,
    "learning_steps": 320000,
    "log_interval": 20,
    "save_interval": 2000,
    "eval_interval": 1000,
    "ema_rate": "0.5,0.9,0.99,0.9999",
    "resume_checkpoint": "",
    # # # Scheduler and diffusion # # #
    "schedule_sampler": "lossaware",
    "diffusion_steps": 1000,                    # Changed
    "noise_schedule": "sqrt",
    "timestep_respacing": "",
    # # # Arguments for dataset and model # # #
    "seq_len": 256,                             # filter data by data['length'] <= seq_len. max is 2096.
    "vocab_size": 729,                          # TODO (alter corruption-730)
    "pretrained_denoiser": "",                  # To use Pretrained de-noising weight, provide .pt name
    "pretrained_embedding": "",                 # To use POZALabs' embedding, provide .pt name
    "freeze_embedding": True,                   # you MUST turn this on with pretrained_embedding
    "use_bucketing": True,
    # # # Arguments for dataset # # #
    "dataset": "ComMU",
    "data_dir": "datasets/ComMU-processed",
    "data_loader_workers": 2,                   # num_workers for DataLoader
    # # # Arguments for corruption # # #
    "use_corruption": False,                    # switch to use corruption
    "corr_available": "mt,mn,rn,rr",            # Available corruptions - TODO: add 'at'
    "corr_max": 0,                              # Max number of corruptions
    "corr_p": 0.5,                              # Probability to choice each corruption
    # "corr_kwargs": "dict(p=0.5,count=3)",     # Keyword arguments for each corruption
    # # # Arguments for model # # #
    "hidden_t_dim": 128,                        # Transformer
    "hidden_dim": 500,                          # Transformer and Embedding
    "dropout": 0.1,
    # # # Not Used
    # "intermediate_dim": 1024,                 # FNet
    # "num_layers": 6,
    # "num_attention_heads" : 10,
    # # # Arguments for other settings # # #
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
})
