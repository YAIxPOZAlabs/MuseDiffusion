# Example script of modification
python3 -m MuseDiffusion modification --distributed \
--use_corruption True \
--corr_available mt,mn,rn,rr \
--corr_max 4 \
--corr_p 0.5 \
--step 1000 \
--top_p 1 \
--clamp_step 0 \
--clip_denoised true \
--sample_seed 123 \
--model_path diffusion_models/{name-of-model-folder}/{weight-file}
