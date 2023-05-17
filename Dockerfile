FROM nvcr.io/nvidia/pytorch:21.08-py3

RUN pip install bert_score \
    blobfile \
    nltk \
    numpy \
    packaging \
    psutil \
    PyYAML \
    setuptools \
    spacy \
    torchmetrics \
    tqdm \
    transformers==4.22.2 \
    wandb \
    datasets \
    scipy \
    scikit-learn \
    seaborn \
    pydantic \
    miditoolkit==0.1.16 \
    pretty-midi==0.2.9 \
    pydantic==1.9.1 \
    parmap==1.5.3 \
    logger \
    yacs
