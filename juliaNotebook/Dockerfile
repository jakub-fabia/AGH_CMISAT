FROM quay.io/jupyter/datascience-notebook:2025-03-25

USER root
RUN apt-get install -y curl

USER jovyan
COPY install.jl install.jl
RUN julia install.jl