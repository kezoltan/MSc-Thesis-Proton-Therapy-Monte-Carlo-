# MSc-Thesis-Proton-Therapy-Monte-Carlo-
Supporting Python code in relation to the MSc thesis which estimates the absorbed dose of a monoenergetic proton beam in a water phantom.

Instructions:
- Set all parameters in doseparams.py. The most important ones are method, dose_method, labelled under mc/mlmc params. The former two respectively decide which SDE EM scheme to use, which dose computation method is used. The angular scheme is always the geometric Euler scheme. Select which estimation method you wish to use here also. 
- Run the dose estimation in dose_main.py, which receives all parameters from doseparams.py. 
- All other files need not be changed by the user, these perform the required calculations.

For the mlmc driver code, we credit Peiren Wang for providing us with his completed versions. These have been edited for this particular application.    
