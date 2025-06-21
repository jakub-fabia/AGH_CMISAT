FROM quay.io/jupyter/datascience-notebook:2025-03-25

USER root
RUN apt-get install -y curl

USER jovyan
# Labs 1-4
RUN julia -e 'using Pkg; Pkg.add.(["IJulia", "DataFrames", "Statistics", "Plots", "QuadGK", "CSV", "Interpolations", "Polynomials"])'
# Lab 5-10
RUN julia -e 'using Pkg; Pkg.add.(["TaylorSeries", "SpecialPolynomials", "Roots", "ForwardDiff", "Colors"])'
# Lab 11
RUN julia -e 'using Pkg; Pkg.add.(["StaticArrays", "StatsPlots"])'
# Lab 12
RUN julia -e 'using Pkg; Pkg.add.(["FFTW"])'