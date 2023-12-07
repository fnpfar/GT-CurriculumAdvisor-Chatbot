# Georgia Tech Curriculum Advisor Chatbot
## CS ECE 8803 Final Project

**Authors:** Brieuc POPPER <bpopper3@gatech.edu>, Francisco Nicolás PÉREZ FERNÁNDEZ <ffernandez35@gatech.edu>

Instructions for the installation, set up, fine-tuning and inference in Georgia Tech PACE-ICE cluster:

0. Edit this script with your PACE-ICE user name:
    0.1. Ctrl+F -> replace all PACEUSER strings on this script by your user ID on PACE (assuming that PACE provides you with the path /home/hice1/PACEUSER/scratch)

1. Setting up PACE-ICE:
    1.1. Move the temp directory to Scratch (to prevent restricted storage home from filling)
        mkdir /home/hice1/PACEUSER/scratch/temp
        export TMPDIR='/home/hice1/PACEUSER/scratch/temp'
    1.2. Install Conda on Scratch:
        mkdir /home/hice1/PACEUSER/scratch/.conda
        ln -s /home/hice1/PACEUSER/scratch/.conda /home/hice1/PACEUSER/.conda
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p /home/hice1/PACEUSER/scratch/conda
        source ~/scratch/conda/etc/profile.d/conda.sh
        mkdir /home/hice1/PACEUSER/scratch/conda/conda_cache
        conda config --add pkgs_dirs /home/hice1/PACEUSER/scratch/conda/conda_cache
    1.3. Create Conda enviroment and install the dependencies:
        source ~/scratch/conda/etc/profile.d/conda.sh
        conda env create -f <THE PATH TO THE PROVIDED conda_enviroment.yml FILE>
    1.4. Create additional directories
        mkdir /home/hice1/PACEUSER/scratch/hfHOME
        mkdir /home/hice1/PACEUSER/scratch/hugCache
    1.5. Activate conda, in the created envLlama enviroment
        conda activate envLlama

2. Fine-tuning of Llama chat (with the provided .parquet datasets):       
    2.1. Allocate GPU (2x V100 during 2h) -> Transfered to GPU machine:
        salloc -N1 --mem-per-gpu=32G -t2:00:00 --gres=gpu:V100:2
    2.2. Activate conda in the new machine:
        source ~/scratch/conda/etc/profile.d/conda.sh
        conda activate envLlama
    2.3. Fine-tune Llama 2 chat with qLora:
        export HF_HOME=/home/hice1/PACEUSER/scratch/hfHOME
        export HF_DATASETS_CACHE=/home/hice1/PACEUSER/scratch/hugCache  
        2.3.1 Edit qLoRaFineTuning.py (qLoRa script for fine-tuning):
             -> Change output directory where the model predictions and checkpoints will be stored (line 85)
             -> Change .parquet dataset input path (line 159) to the path of the provided datasets (coursework.parquet for the coursework >2k QA dataset or generationAndQA.parquet for the 99 QA + curriculum generation merged dataset) (line 159)
             -> Change the path were a training info .csv dataset is stored (line 243)
             -> Change the directory were the fine-tuned model is saved (line 261) -> This is the input to the inference script.
             -> Change the fine tuning parameters if needed.
        2.3.2 Run qLoRaFineTuning.py:
        python <THE PATH TO THE PROVIDED qLoRaFineTuning.py FILE>

3. Run the chatbot (inference with the previously generated fine-tuned model or non-tuned Llama Chat):
    3.1. Allocate GPU (2x V100 during 2h) -> Transfered to GPU machine:
        salloc -N1 --mem-per-gpu=32G -t2:00:00 --gres=gpu:V100:2
    3.2. Activate conda in the new machine:
        source ~/scratch/conda/etc/profile.d/conda.sh
        conda activate envLlama
    3.3. Run the chatbot:
        export HF_HOME=/home/hice1/PACEUSER/scratch/hfHOME
        export HF_DATASETS_CACHE=scratch/hugCache
        3.3.1 Edit chatBot.py (chat-bot inference script):
            -> To use Llama Chat without fine-tuning, comment line 107, and edit the parameter on line 123 so that model=model becomes model=base_model. Else (to use the previously trained weights), left line 107 uncommented and model=model, and edit line 106, with the path of the directory of the fine-tuned model from step 2.3.1.
            -> The context can be edited in line 127: chat = [{"context"}]
        2.3.2 Run chatBot.py:
        python <THE PATH TO THE PROVIDED qLoRaFineTuning.py FILE>
        2.3.3 Use the prompt in the terminal to ask questions

