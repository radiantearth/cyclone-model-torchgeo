FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN apt-get update -y && \
    apt-get install -y curl && \
    pip install --upgrade pip && \
    pip install torchgeo[train]==0.1.0 scikit-learn tqdm

COPY run_model.py /app/run_model.py

RUN curl https://zenodo.org/record/5773331/files/last.ckpt?download=1 -o /app/cyclone_example_model.ckpt

ENV CHECKPOINT_LOCATION="/app/cyclone_example_model.ckpt"
ENV INPUT_DATA_VOLUME="/var/data/input"
ENV OUTPUT_DATA_VOLUME="/var/data/output"

ENTRYPOINT [ "python", "/app/run_model.py" ]