# ⚽ EPL Match Data Analysis (2000–2025)

This project analyzes **English Premier League (EPL)** match data from 2000–2025, in order to predict game outcomes for the upcoming 2025-2026 season.
It uses **pandas** for data handling, **scikit-learn** for machine learning, and **kagglehub** to fetch datasets directly from Kaggle.

---

## Installation

### 1. Clone this repository
```bash
git clone https://github.com/apaliwal3/Premier_League_Predictor.git
cd Premier_League_Predictor
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate        # Windows (PowerShell)
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Data

Raw datasets are **not stored in the repository** (see `.gitignore`).  
Instead, they are downloaded automatically with `kagglehub`:

```python
import os
import pandas as pd
import kagglehub

# Download EPL dataset
path = kagglehub.dataset_download("marcohuiii/english-premier-league-epl-match-data-2000-2025")
print("Dataset downloaded to:", path)

# Load first CSV into pandas
csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
df = pd.read_csv(os.path.join(path, csv_file))
print(df.head())
```

Datasets are cached locally (usually in `~/.cache/kagglehub/`).

---

## Usage

- Open and run the Jupyter notebooks (`*.ipynb`) for exploration and analysis.  
- Modify the scripts to test your own models or visualizations.  
- Example notebook: `pl_predictor.ipynb` – a basic match outcome predictor.

---

## Contributing

Contributions are welcome!  
- Fork the repo  
- Create a feature branch (`git checkout -b feature/my-feature`)  
- Commit your changes  
- Open a Pull Request  

---

## 📜 License

This project is licensed under the **MIT License**.
