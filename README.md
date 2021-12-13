# Cyclone Wind Estimation Model Docker Image

This repository contains code and instructions for creating a Docker image based on the [Tropical
Cyclone Wind Estimation model](https://doi.org/10.5281/zenodo.5773331) that can be used to generate
predictions from new input data. This model was trained on the [Tropical Cyclone Wind Estimation Competition
dataset](https://mlhub.earth/10.34911/rdnt.xs53up) using the [Torchgeo](https://github.com/microsoft/torchgeo)
package. 

This Docker image is available on DockerHub as [radiantearth/cyclone-model-torchgeo](https://hub.docker.com/repository/docker/radiantearth/cyclone-model-torchgeo).

## Citation

Caleb Robinson. (2021). Tropical Cyclone Wind Estimation model (2.0). Zenodo. https://doi.org/10.5281/zenodo.5773331.

## Create Docker Image

```bash
$ docker build -t cyclone-model-torchgeo:1 docker
```

## Run Docker Image

Using `docker run`:

```bash
docker run -it --rm -v $PWD/data/input:/var/data/input -v $PWD/data/output:/var/data/output cyclone-model-torchgeo:1
```

or using Docker Compose:

```bash
export INPUT_DATA=$PWD/data/input;
export OUTPUT_DATA=$PWD/data/output;
docker-compose -f inferencing.yml run inference
```