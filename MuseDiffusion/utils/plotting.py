META = (
    ("zero_pad", 1, "black"),
    ("eos", 1, "gray"),
    ("bar", 1, "silver"),
    ("pitch", 128, "firebrick"),
    ("velocity", 64, "red"),
    ("chord", 109, "sandybrown"),
    ("duration", 128, "gold"),
    ("position", 128, "olivedrab"),
    ("bpm", 41, "darkgreen"),
    ("key", 25, "deepskyblue"),
    ("time_signiture", 4, "slategray"),
    ("pitch_range", 8, "royalblue"),
    ("#measrue", 3, "darkorchid"),
    ("instrument", 9, "aqua"),
    ("genre", 3, "sienna"),
    ("meta_velocity", 66, "palegoldenrod"),
    ("track_role", 7, "cornflowerblue"),
    ("rhythm", 3, "lime"),
)


def plot_embedding_tsne(emb_weight, title="Embedding", alpha=1.0, figsize=(16, 16), __meta=META):
    from sklearn.manifold import TSNE
    import seaborn as sns
    import matplotlib.pyplot as plt
    emb_weight = emb_weight.detach().cpu().numpy()
    result = TSNE(n_components=2, random_state=42).fit_transform(emb_weight)
    x = result[:, 0]
    y = result[:, 1]
    hue = sum(map(lambda k: [k[0]] * k[1], __meta), start=[])
    palette = [k[2] for k in __meta]

    fig = plt.figure(figsize=figsize)
    sns.scatterplot(x=x, y=y, hue=hue, palette=palette, alpha=alpha, ax=fig.gca()).set(title=title)
    return fig


def embedding_tsne_trainer_wandb_callback(self):
    import wandb
    logs = {}
    for r, p in zip([0, *self.ema_rate], [self.master_params, *self.ema_params]):
        w = self._master_params_to_state_dict(p, key="word_embedding.weight")
        key = "Embedding-{}".format("master" if r == 0 else "ema{}".format(r))
        fig = plot_embedding_tsne(w, title=key)
        logs[key] = wandb.Image(fig)
    wandb.log(logs)


try:
    from sklearn.manifold import TSNE
    import seaborn as sns
except ImportError:
    import warnings
    warnings.warn("scikit-learn or seaborn is not installed, so you won't be able to log plots.")
    embedding_tsne_trainer_wandb_callback = bool  # dummy function
