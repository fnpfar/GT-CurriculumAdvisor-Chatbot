import os

#HF_HOME
#HF_DATASETS_CACHE
#export HF_DATASETS_CACHE=scratch/hugCache



#salloc -N1 --gres=gpu:RTX_6000:1 -t0:45:00            --gres=gpu:A40:1
#export HF_HOME=/home/hice1/PACEUSER/scratch/hfHOME
#export HF_DATASETS_CACHE=scratch/hugCache


#pip installed stuff after module load cuda


#Quadro RTX 6000
# Currently Loaded Modules:
#   1) pace-slurm/2022.06   4) mpc/1.2.1-zoh6w2  (H)   7) gcc/10.3.0-o57x6h             10) xz/5.2.2-kbeci4       (H)  13) slurm/current-4bdz7m  (H)
#   2) gmp/6.2.1-mw6xsf     5) zlib/1.2.7-s3gked (H)   8) libpciaccess/0.16-wfowrn (H)  11) libxml2/2.9.13-d4fgiv (H)  14) mvapich2/2.3.6-ouywal
#   3) mpfr/4.1.0-32gcbv    6) zstd/1.5.2-726gdz (H)   9) libiconv/1.16-pbdcxj     (H)  12) rdma-core/15-t43af2   (H)  15) cuda/12.1.1-6oacj6

#   Where:
#    H:  Hidden Module

print("starting")
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    LlamaConfig,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


# Output directory where the model predictions and checkpoints will be stored
output_dir = "/home/hice1/PACEUSER/scratch/pythonBazar"


import gc
gc.collect()
torch.cuda.empty_cache()


#load new_model from path /home/hice1/PACEUSER/scratch/modelSave

device_map = "auto" 


#use device map with 4 
model_name = "NousResearch/llama-2-7b-chat-hf"

#create a config
""""architectures": [
    "LlamaForCausalLM"
  ],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.31.0.dev0",
  "use_cache": true,
  "vocab_size": 32000"""
"""BartConfig, BertConfig, BertGenerationConfig, BigBirdConfig, BigBirdPegasusConfig, BioGptConfig
, BlenderbotConfig, BlenderbotSmallConfig, BloomConfig, CamembertConfig, LlamaConfig, CodeGenConfig, 
CpmAntConfig, CTRLConfig, Data2VecTextConfig, ElectraConfig, ErnieConfig, FalconConfig, FuyuConfig, GitConfig,
 GPT2Config, GPT2Config, GPTBigCodeConfig, GPTNeoConfig, GPTNeoXConfig, GPTNeoXJapaneseConfig, GPTJConfig,
   LlamaConfig, MarianConfig, MBartConfig, MegaConfig, MegatronBertConfig, MistralConfig, MptConfig,
     MusicgenConfig, MvpConfig, OpenLlamaConfig, OpenAIGPTConfig, OPTConfig, PegasusConfig, PersimmonConfig,
       PLBartConfig, ProphetNetConfig, QDQBertConfig, ReformerConfig, RemBertConfig, RobertaConfig, 
       RobertaPreLayerNormConfig, RoCBertConfig, RoFormerConfig, RwkvConfig, Speech2Text2Config, TransfoXLConfig,
         TrOCRConfig, WhisperConfig, XGLMConfig,
 XLMConfig, XLMProphetNetConfig, XLMRobertaConfig, XLMRobertaXLConfig, XLNetConfig, XmodConfig."""

#cfg=LlamaConfig(architectures=["LlamaForCausalLM"],bos_token_id=1,eos_token_id=2,hidden_act="silu",hidden_size=4096,initializer_range=0.02,intermediate_size=11008,max_position_embeddings=4096,model_type="llama",num_attention_heads=32,num_hidden_layers=32,num_key_value_heads=32,pad_token_id=0,pretraining_tp=1,rms_norm_eps=1e-05,rope_scaling=None,tie_word_embeddings=False,torch_dtype="float16",transformers_version="4.31.0.dev0",use_cache=True,vocab_size=32000)
  

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,

    torch_dtype=torch.float16, #
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model,'/home/hice1/PACEUSER/scratch/modelSave/modelName' )
model = model.merge_and_unload()

#save the model   
#model.save_pretrained('/home/hice1/PACEUSER/scratch/myHomeDiredtly')


#get base_models device map
device_map = base_model.hf_device_map
#Print it
print(device_map)

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

pipe = pipeline(task="conversational", model=model, tokenizer=tokenizer, max_length=4096)



chat = [
  {"role": "user", "content": """Below are all the available graduate level course for the MSECE of Georgia Tech. Please, use this and only this courses in your future responses.
Each line contains a course code, its name, its technical interest area (TIA) and the terms when student can register it.
ECE 6200 : Biomedical Applications of MEMS is in Bioengineering TIA and is taught only in Spring 2024, and Spring 2025.
ECE 6780 : Medical Image Processing is in Bioengineering TIA and is taught only in and Spring 2024.
ECE 6790 : Information Processing Models in Neural Systems is in Bioengineering TIA and is taught only in and Spring 2025.
ECE 6100 : Advanced Computer Architecture is in Computer Systems and Software TIA and is taught only in Spring 2024, Fall 2024, and Spring 2025.
ECE 6110 : CAD for Computer Communication Networks is in Computer Systems and Software TIA and is taught only in and Spring 2024.
ECE 6612 : Computer Network Security is in Computer Systems and Software TIA and is taught only in and Spring 2025.
ECE 8803 CPS : Cyber Physical Design and Analysis (MSCSEC Energy) is in Computer Systems and Software TIA and is taught only in Spring 2024, Fall 2024, and Spring 2025.
ECE 8803 FML : Fundamentals of Machine Learning is in Computer Systems and Software TIA and is taught only in Spring 2024, and Spring 2025.
ECE 8893 AML : Hardware Acceleration for Machine Learning is in Computer Systems and Software TIA and is taught only in and Spring 2024.
ECE 8893 FPG : FPGA Acceleration is in Computer Systems and Software TIA and is taught only in and Spring 2025.
ECE 6250 : Advanced Digital Signal Processing is in Digital Signal Processing TIA and is taught only in Spring 2024, and Spring 2025.
ECE 6254 : Statistical Machine Learning is in Digital Signal Processing TIA and is taught only in Spring 2024, and Spring 2025.
ECE 6255 : Digital Processing of Speech Signals is in Digital Signal Processing TIA and is taught only in Spring 2024, and Spring 2025.
ECE 6258 : Digital Image Processing is in Digital Signal Processing TIA and is taught only in and Fall 2024.
ECE 6270 : Convex Optimization for SP is in Digital Signal Processing TIA and is taught only in and Fall 2024.
ECE 6272 : Fund of Radar Signal Processing is in Digital Signal Processing TIA and is taught only in and Spring 2025.
ECE 6282 : Radar Imaging is in Digital Signal Processing TIA and is taught only in and Spring 2024.
ECE 7751 : Probabilistic Graphical Models in Machine Learning is in Digital Signal Processing TIA and is taught only in Spring 2024, and Spring 2025.
ECE 8803 CAI : Conversational AI is in Digital Signal Processing TIA and is taught only in and Fall 2024.
ECE 8803 FML : Fundamentals of Machine Learning is in Computer Systems and Software TIA and is taught only in Spring 2024, and Spring 2025.
ECE 8803 SLP : Spoken Language Processing with Deep Learning is in Digital Signal Processing TIA and is taught only in and Fall 2024.
ECE 8803 ODM : Online Decision Making in ML is in Digital Signal Processing TIA and is taught only in and Fall 2024.
ECE 6320 : Power Systems Control and Operation is in Electrical Energy TIA and is taught only in and Fall 2024.
ECE 6360 : Microwave Design is in Electromagnetics TIA and is taught only in Spring 2024, and Spring 2025.
ECE 6370 : Electromagnetic Radiation and Antennas is in Electromagnetics TIA and is taught only in Spring 2024, and Spring 2025.
ECE 6380 : Intro to Computational Electromagnetics is in Electromagnetics TIA and is taught only in and Fall 2024.
ECE 6390 : Satellite Comm and Navigation Systems is in Electromagnetics TIA and is taught only in and Fall 2024.
ECE 6412 : Analog Integrated Circuit Design is in Electronic Design & Applications TIA and is taught only in Spring 2024, and Spring 2025.
ECE 6414 : Analog Integrated System Design is in Electronic Design & Applications TIA and is taught only in and Spring 2025.
ECE 6435 : Neuromorphic Analog VLSI Circuits is in Electronic Design & Applications TIA and is taught only in and Spring 2024.
ECE 6450 : Intro to Microelectronics Technology is in Nanotechnology TIA and is taught only in and Fall 2024.
ECE 6453 : Theory of Electronic Devices is in Nanotechnology TIA and is taught only in and Spring 2024.
ECE 6456 : Solar Cells is in Nanotechnology TIA and is taught only in and Fall 2024.
ECE 6460 : Microelectromechanical Devices is in Nanotechnology TIA and is taught only in Spring 2024, and Spring 2025.
ECE 6771 : Optoelectronics: Materials, Processes, Devices is in Nanotechnology TIA and is taught only in and Fall 2024.
ECE 6776 : Microelectronic Systems Packaging Technologies is in Nanotechnology TIA and is taught only in Spring 2024, and Fall 2024.
ECE 6501 : Fourier Optics and Holography is in Optics and Photonics TIA and is taught only in and Spring 2025.
ECE 6771 : Optoelectronics: Materials, Processes, Devices is in Nanotechnology TIA and is taught only in and Fall 2024.
ECE 6550 : Linear Systems and Controls is in Systems and Controls TIA and is taught only in Summer 2024, and Fall 2024.
ECE 6552 : Nonlinear Systems and Control is in Systems and Controls TIA and is taught only in Spring 2024, and Spring 2025.
ECE 6601 : Random Processes is in Telecommunications TIA and is taught only in Summer 2024, and Fall 2024.
ECE 6602 : Digital Communications is in Telecommunications TIA and is taught only in Spring 2024, and Spring 2025.
ECE 6604 : Personal and Mobile Communications is in Telecommunications TIA and is taught only in and Fall 2024.
ECE 6605 : Information Theory is in Telecommunications TIA and is taught only in and Fall 2024.
ECE 6607 : Computer Communication Networks is in Telecommunications TIA and is taught only in and Fall 2024.
ECE 6610 : Wireless Networks is in Telecommunications TIA and is taught only in and Fall 2024.
ECE 6140 : Digital Systems Test is in VLSI Systems and Digital Design TIA and is taught only in and Fall 2024.
ECE 6156 : Hardware Oriented Security and Trust is in VLSI Systems and Digital Design TIA and is taught only in Spring 2024, and Spring 2025.
ECE 6001 : ECE 6001 Technology Entrepreneurship: Teaming, Ideation, and Entrepreneurship is taught in all semesters. 
The provided couse selections depends on the student choosing MSECE non-thesis or MSECE thesis options:
To graduate in the MSECE non-thesis option, the student must register courses in three groups:
Group I: Composed of three graduate level couses (a total of 9 SCH) in one or two TIAs of the student's choosing. 
Group II: Composed of three graduate level courses (a total of 9 SCH). One of them may be in one of the TIAs in the couses selected from group I, 
but the other two must be in different TIAs from the courses selected in group I.
Gorup III: Composed of the Technology Entrepreneurship ECE 6001, which is mandatory, plus three other elective courses (9 SCH).
To graduate in the MSECE thesis option, the student must register courses in three groups:
Group I: Composed of two graduate level couses (a total of 6 SCH) in one or two TIAs of the student's choosing. 
Group II: Composed of two graduate level courses (a total of 6 SCH). Both two courses must be
in technical interest areas different than the technical interest areas of the courses selected in group I.
Gorup III: Composed of the Technology Entrepreneurship course ECE 6001, which is mandatory, plus another elective course (3 SCH).
In addition, the student must complete the Responsible Conduct of Research (RCR) Requirement.
In addition, the student must register in at least 12 SCH corresponding to the thesis. The course code
corresponding with the thesis is ECE 7000 : M.S. Thesis Research.
M.S. thesis option students mus present a "research review" to their advisor and reading committee members so that the 
"Request for Approval of the M.S. Thesis Topic" can be approved by the committee and submitted to the ECE Graduate Affairs Office for processing. 
Start by greeting the student and asking him about his background to better understand how you can help him.
Answer only questions about the Georgia Tech master of science in Electrical and Computer Engineering (MSECE). 
Refuse to answer other questions.
If it is unclear which context the question is in, you may assume that all questions refer to the MSECE of Gerogia Tech.
"""}
]


#printing out the start of the CHAT
print("###############USER################\n")
for i in range(0, len(chat)):
    print(chat[i]['content'])
import os
print("$$$####################GPU USAGE STAT$#@$####################\n")
os.system("nvidia-smi")

print("$$$####################GPU USAGE STAT$#@$####################\n")
while True:

    

    result = pipe(chat)
    txt=result.generated_responses[-1]
    #take off any \n in txt
    txt=txt.replace('\n','')
    

    print("################ASSISTANT#######################\n"+txt)

#chat.append({"role": "assistant", "content": result.generated_responses[-1]})
#print("THE FIRST APPEND ON LINE 90 HAS BEEN MADE"+str(result.generated_responses[-1]))


    nextIn=""
    nextIn=input('\n Enter your next question, or write gpu to see gpu usage stats:\n')
    if nextIn == "gpu":
        os.system("nvidia-smi")
        nextIn=input('\n Enter your next question:\n')

    
    chat.append({"role": "user", "content": nextIn})
