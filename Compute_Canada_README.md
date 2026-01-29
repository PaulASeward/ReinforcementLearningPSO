## Creating Multiple Experiments

The copy_and_replace.sh can crete multiple directories fo run experiments from in compute canada. It can be executed as follows:

```bash
./copy_and_replace.sh 11
```

## Moving and Running the Project into Compute Canada

Logging Into ComputeCanada
```
ssh -y <username>@cedar.computecanada.ca
password
MFA Code
```

Making Virtual Environment
```
module load python/3
virtualenv --no-download TF_RL
source TF_RL/bin/activate
pip install -r requirements.txt
```

Running the bash script and checking the status of the job
```
cd <currentdate_f#> && sbatch run_agent.sh && cd ..
sq
```



Copying the files from this project directory in terminal:
```
scp -r run_history/<currentdate_f#> pseward@cedar.computecanada.ca:scratch/
```

Copying from ComputeCanada to Local Machine
```20241119_f11_drqn_decay_rate
scp -r pseward@cedar.computecanada.ca:scratch/<currentdate_f#>/results run_history/
```
