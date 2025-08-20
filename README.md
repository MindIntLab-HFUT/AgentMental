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

Replace your model, base_url and api_key with your actual model, base_url and API keys. It can be placed in the OAI_CONFIG_LIST.

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
~$ python data_process.py
```

