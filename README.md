# Machine Learning for Diamond Price Prediction

This project predicts diamond prices using multiple linear regression models based on carat, cut, and other features from the famous `diamonds.csv` dataset.

## Features
- Data cleaning & outlier filtering
- Encoding categorical features (cut, color, clarity)
- Feature engineering (`carat_root`, `log_price`)
- Multiple linear regression models:
    - Model 1: `carat_root`
    - Model 2: `carat_root` + `cut_num`
    - Model 3: Filtered data to remove outliers
- Visualization of correlation matrices and prediction vs actual price distributions

## Usage
1. Clone the repository
    ```bash
    git clone https://github.com/yihanshen-elsa/machine-learning-algorithms.git
    cd machine-learning-algorithms
    ```

2. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

3. Make sure you have `diamonds.csv` in the same directory.

4. Run the script
    ```bash
    python diamond_price_prediction.py
    ```

## Results
- Found that adding `cut_num` increased overlap area from **62% → 77%**, improving predictions.
- Filtering outliers had minor improvement.

## Future Improvements
- Try adding more features like `color_num`, `clarity_num`.
- Experiment with non-linear models (e.g. XGBoost, Random Forest).
- Cross-validation to better evaluate performance.

## License
MIT
