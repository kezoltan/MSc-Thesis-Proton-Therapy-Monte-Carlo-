# MSc-Thesis-Proton-Therapy-Monte-Carlo-
Supporting Python code in relation to the MSc thesis titled: _Stochastic Modelling of Proton Transport and Spatial Therapeutic Dose Prediction with Multilevel Monte Carlo_, supervised by Professor Mike Giles at the University of Oxford. Major references that preceed this work are _[Giles, 2008][https://people.maths.ox.ac.uk/gilesm/files/OPRE_2008.pdf]_ and _[Chronholm and Pryer, 2026][https://arxiv.org/abs/2509.13223]_.

The function of this code is to estimate the absorbed dose of a monoenergetic proton beam in a water phantom via different methods. We evaluate the performance and numerical treatment of each of the following (1) the underlying proton transport model (2) the dose computation technique and (3) the Monte Carlo method. Firstly, we evaluate a pair of SDE proton transport models simulated with the Euler-Maruyama discretisation (specifically, a geometry preserving scheme using the Riemannian exponential map on the sphere) where one is parameterised in energy and the other in track length. Secondly, the compare the existing dose computation method in _Chronholm and Pryer_ with a smoother alternative developed in the adjoining thesis to this work. Thirdly, we compare and evaluate the use of standard Monte Carlo against Multilevel Monte Carlo for this task. 

For the MLMC driver code, we credit Peiren Wang for providing us with his completed versions in Python, which have been adapted for this particular application by the author.  

Instructions for a user:
- All parameters are set in doseparams.py, and we recommend that the user reviews these. The most important ones are method, dose_method, file_path, SPATIAL_DIM, sampling_type, and all parameters relating to simulation numbers which are labelled under MC/MLMC params. The former three respectively decide which SDE EM scheme to use, which dose computation method is used, and which Monte Carlo sampling type is used. 
- Run the dose estimation in dose_main.py, which receives all parameters from doseparams.py. 
- All other files need not be changed by the user, these perform the required calculations.

  
