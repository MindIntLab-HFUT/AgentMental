# AgentMental

## Quick Start

Clone this project locally
```bash
git clone https://github.com/MACLAB-HFUT/AgentMental.git
```

Create a new virtual environment, e.g. with conda or miniconda 
```bash
~$ conda create -n agentmental python=3.10
```

Activate the environment:
```bash
~$ conda activate agentmental
```

Install the required packages:
```bash
~$ pip install -r requirements.txt
```

Replace your model, base_url and api_key with your actual model, base_url and API keys in files ```OAI_CONFIG_LIST, generate_response.py, memory.py```.

Run
```bash
~$ python main.py
```

## Data

Download the dataset
```bash
~$ python data_download.py
```

Data processing
```bash
~$ python extract_data.py
~$ python data_process.py
```

Your directory structure should look like this:
```
AgentMental/
├── data/
│   ├── data_download.py
│   ├── data_process.py
│   ├── extract_data.py
│   ├── processed_train_daic_woz
│   └── ... (other dataset files)
├── evaluation
├── scales
├── src
├── result.py
├── requirements.txt
└── README.md
```


