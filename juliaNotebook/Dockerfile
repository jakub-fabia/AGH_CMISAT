FROM jupyter/base-notebook:latest

USER root

RUN apt-get update
RUN apt-get install -y curl
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.3-linux-x86_64.tar.gz
RUN tar -xvzf julia-1.11.3-linux-x86_64.tar.gz
RUN cp -r julia-1.11.3 /opt/
RUN ln -s /opt/julia-1.11.3/bin/julia /usr/local/bin/julia

USER jovyan
COPY install.jl install.jl
RUN julia install.jl