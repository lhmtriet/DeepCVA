This is the README file for the reproduction package of the paper: "DeepCVA: Automated Commit-level Vulnerability Assessment with Deep Multi-task Learning", accepted for publication at the 36th IEEE/ACM International Conference on Automated Software Engineering (ASE 2021).

The package contains the following artefacts:
1. Data: contains the preprocessed datasets as well as commit data we used in our work. However, due to the large size, please download the data from https://figshare.com/s/14a9f463c21de54019b6 into the folder `Data/`
2. Code: contains the source code we used in our work. It's noted that we setup the code to run on a supercomputing cluster that runs on Slurm and has GPUs. Therefore, most of the code must be submitted using bash script (.sh) file, but our code can still be run locally by executing the python file directly.

The dataset of 1,229 VCCs we gathered are given in `data/java_vccs.csv`, respectively.

Before running any code, please install all the required Python packages using the following command: `pip install -r requirements.txt`

## Feature Generation and Baseline Models
1. Train and infer the code features by running `infer_features/infer_pipeline.sh`.
2. Train and infer the ast features by running `jid1=$(sbatch train_feature_model_ast.sh) && sbatch --dependency=afterok:${jid1##* } infer_features_ast.sh` (slurm script)
3. Train and test the ML baseline models for RQ1 by running `model_pipeline.sh`.
4. Train and test the K-means baseline model in RQ1 by running `kmeans_baseline/main.sh`.

## DeepCVA and Model Variants
1. Train and infer the features using `infer_features/infer_features_sequential.sh` (slurm script)
2. The DeepCVA model used in the paper can be trained by running `DeepCVA/multi_task_sequential_crnn.sh`. The variants of DeepCVA (in RQ2) can be obtained from the original DeepCVA model by removing respective components based on the descriptions in the paper.

## 84 software metrics used for baseline models:

27 Change/project/developer metrics: No. of project stars, no. of project forks, no. of commits, no. of hunks, entropy (distribution of modified code across each file), no. of modified files, no. of modified directories, no. of modified subsystems, no. of lines added/deleted, no. of lines in file before the change, whether or not the change is a defect fix, no. of developers changed modified files, time between current and last change, unique changes to modified files, developer's experience, recent developer's experience, developer's experience on a subsystem, no. of added/deleted conditions, no. of added/modified/deleted functions, no. of added/deleted function calls, no. of added/deleted variable assignments

57 Java keywords: abstract, assert, boolean, break, byte, case, catch, char, class, continue, const, default, do, double, else, enum, exports, extends, false, final, finally, float, for, goto, if, implements, import, instanceof, int, interface, long, module, native, new, null, package, private, protected, public, requires, return, short, static, strictfp, super, switch, synchronized, this, throw, throws, transient, true, try, var, void, volatile, while

