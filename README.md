# Air Pollution Decision Tree Classifier

## Overview
This project implements a Decision Tree classifier to predict air pollution levels based on various environmental factors. The model is trained using real-world air pollution datasets and aims to assist in understanding air quality patterns.

## Features
- Preprocesses air pollution data.
- Trains a Decision Tree classifier.
- Evaluates the model's performance using accuracy metrics.
- Provides predictions on air pollution levels.

## Installation
To run this project, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Air_Pollution_Decision_Tree.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Air_Pollution_Decision_Tree
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Ensure you have the dataset in the `data/` directory.
2. Run the script:
   ```bash
   python scripts/air_pollution_decision.py
   ```
3. The model will train and display evaluation metrics.

## Project Structure
```
/Air_Pollution_Decision_Tree
│── data/
│   ├── global_air_pollution_data.csv
│── models/
│   ├── air_quality_index_prediction_model.sav
│── scripts/
│   ├── air_pollution_decision.py
│── README.md
│── requirements.txt
│── LICENSE
```

## Dataset
The dataset consists of air pollution metrics such as:
- PM2.5
- PM10
- NO2
- CO
- SO2
- Temperature
- Humidity

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
[Allan Otieno Akumu](https://github.com/AllanOtieno254)

## Acknowledgments
- Open-source air quality datasets.
- Python libraries such as Scikit-learn and Pandas.

