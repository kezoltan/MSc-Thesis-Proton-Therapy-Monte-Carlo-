# MSc-Thesis-Proton-Therapy-Monte-Carlo-
Supporting Python code in relation to the MSc thesis which estimates the absorbed dose of a monoenergetic proton beam in a water phantom.

Instructions:
- Set all parameters in doseparams.py. The most important ones are method, dose_method, and sims_per_cpu - these respectively decide which SDE EM scheme to use, which dose computation method is used, and how many (monte carlo) simulations are performed. The angular scheme is always the geometric Euler scheme. 
- Run the dose estimation in dose_main.py. This also where you select whether you wish to use standard monte carlo, standard mlmc, or antithetic mlmc. 
- All other files need not be changed by the user, these perform the required calculations.   
