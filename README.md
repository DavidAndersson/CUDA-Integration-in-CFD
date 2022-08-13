# CUDA-Integration-in-CFD
Using Python and Numba

This was the project of the course "Project in Applied Mechanics" (TME131) at Chalmers, carried out between April and May 2022. 

The goal of the project was to convert a given CFD-sctipt to run completely on the GPU. The initial script had been written with no consideration of eventually being GPU converted, which meant that extensive rewriting had to be done. 

The initial code was written in Python, and it was expected of us to also carry out the CUDA conversion using Python, using Numba. Numba is a translational module which can take Python code and convert it to C++, which is the native language of CUDA. 

In this repo, there is a folder for three stages; firstly, the inital code that was given to us, secondly, the comprehensive rewriting needed to be done in order to be able to convert it. And thirdly, the final CUDA integrated code. 

Disclaimer: The CFD code relies on a linear solver, and this project was unsuccessful in integrating a linear solver which runs on a GPU. Hence its not possible to extract actual results from this project. However it is possible to compare all other subroutines between the CPU and GPU codes, and see significant performance differences. 

As can be seen in the proejct report, the measured performance boost reached a factor 26.  

