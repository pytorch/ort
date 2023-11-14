
Ideally, prefer to run the test on the cluster 
Run both of the following:
cd ../experiments/cluster/

python baseline_unittests_experiment.py  #This runs UT of different model setting
python moe_functionaltests_experiment.py  #This runs the UTs for MOE module and gating functions


Less ideally To run all the tests locally:
./run_all.sh 

This requires the following packages to be installed:
RUN pip install pytest
RUN pip install mpi4py
RUN pip install pytest-mpi


IF you want to get the coverage information of the UTs, you need the coverage package to be installed:
RUN pip install coverage

After the install, if you got WARNING: "The scripts coverage, coverage-3.7 and coverage3 are installed in  /home/<your_user_name>/.local/bin  which is not on PATH". Consider adding this directory to PATH by:
RUN export PATH=$PATH:/home/<your_user_name>/.local/bin
