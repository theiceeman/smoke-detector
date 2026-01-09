# Smoke Detector

## Setup

### Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Experiments

### Basic experiment (default settings)
```bash
python experiment.py
```

### Custom experiment parameters
```bash
python experiment.py --data-root dataset --epochs 50 --patience 10
```

### Available arguments
- `--data-root`: Root directory containing dataset (default: "dataset")
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 10)

