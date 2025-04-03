FROM quay.io/jupyter/datascience-notebook:2025-03-25

USER root
RUN apt-get install -y curl

USER jovyan
# Labs 1-4
RUN julia -e 'using Pkg; Pkg.add.(["IJulia", "DataFrames", "Statistics", "Plots", "QuadGK", "CSV", "Interpolations", "Polynomials"])'
# Lab 5
RUN julia -e 'using Pkg; Pkg.add.(["TaylorSeries"])'