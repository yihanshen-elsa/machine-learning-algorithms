#Subject: Diamond Price Prediction

import pandas as pd #Imports the Pandas library, which is used for data manipulation and analysis
import numpy as np #Imports the NumPy library, which is used for numerical computing.
import matplotlib.pyplot as plt #Matplotlib is a library for creating static, animated, and interactive visualizations
import seaborn as sns # Seaborn is a statistical data visualization library based on Matplotlib
from sklearn.model_selection import train_test_split #This function is used to split a dataset into training and testing sets
from sklearn.linear_model import LinearRegression #This class is used to build and train a linear regression model
from sklearn.metrics import r2_score #Imports the r2_score function from Scikit-Learn

def train_linear_regression(df, X_columns, Y_column, test_size=0.6, random_state=42, bins=50):
    # Step 4 Define Independent and Dependent Variables
    X = df[X_columns]  # Independent variable
    Y = df[Y_column]  # Dependent variable
    
    # Step 5 Train-Test split into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    
    # Step 6 Apply Linear Regression
    # Train the model
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    # Step 7 Use the test value of X to predict the test values of Y
    Y_pred = model.predict(X_test)

    # Return model and predictions
    return model, X_test, Y_test, Y_pred

def check_overfitting(model, X_train, Y_train, X_test, Y_test, model_name="Model"):
    # Function to check for overfitting by comparing R² scores on training and testing data
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    
    r2_train = r2_score(Y_train, Y_train_pred)
    r2_test = r2_score(Y_test, Y_test_pred)
    
    print(f"{model_name} - R² Score on Training Data: {r2_train:.4f}")
    print(f"{model_name} - R² Score on Testing Data: {r2_test:.4f}")
    
    if r2_train - r2_test > 0.1:  # A threshold of 0.1 is set to detect overfitting
        print(f"{model_name} may be overfitting.")
    else:
        print(f"{model_name} does not show significant overfitting.")

def plot_actual_vs_predicted(Y_test, Y_pred, overlap_area, model_name):
    #Function to plot the Histograms
    plt.figure(figsize=(10, 6))
    hist_actual, bin_edges = np.histogram(Y_test, bins=50, density=True)
    hist_predicted, _ = np.histogram(Y_pred, bins=bin_edges, density=True)
    plt.hist(Y_test, bins=bin_edges, alpha=0.5, label="Actual", density=True)
    plt.hist(Y_pred, bins=bin_edges, alpha=0.5, label="Predicted", density=True)
    plt.xlabel("Log Price")
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"Actual vs Predicted Log Price - {model_name} (Overlap: {overlap_area:.4f})")
    plt.show()

def filter_outliers_by_price(df, threshold=1):
    # Calculate the mean and standard deviation of the price for each carat value
    carat_stats = df.groupby("carat")["price"].agg(["mean", "std"])
    
    # Merge the statistics back into the original DataFrame
    df = df.merge(carat_stats, on="carat", how="left", suffixes=("", "_stats"))
    
    # Calculate the difference between the actual price and the mean price
    df["price_diff"] = abs(df["price"] - df["mean"])
    
    # Filter only the data points where the price difference is within the threshold times the standard deviation
    df_filtered = df[df["price_diff"] <= threshold * df["std"]]
    
    # Print the number of rows before and after filtering
    print(f"Number of original rows: {df.shape[0]}")
    print(f"Number of remaining rows after removing outliers: {df_filtered.shape[0]}")
    
    # Return the filtered DataFrame
    return df_filtered

#Step 1 Load the dataset & know the basic info of our dataset
df = pd.read_csv("diamonds.csv")

# Display basic info
print(df.info()) #It shows up how many kinds of data in this dataset

# Describe dataset
print(df.describe())

# Check for missing values
print(df.isnull().sum()) #summing over columns gives the total number of missing values for each column

# Encode "cut" as numerical values
cut_mapping = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
df["cut_num"] = df["cut"].map(cut_mapping)

# Encode "color" as numerical values
color_mapping = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
df["color_num"] = df["color"].map(color_mapping)

# Encode "clarity" as numerical values (I1 = 0, ..., IF = 7)
clarity_mapping = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}
df["clarity_num"] = df["clarity"].map(clarity_mapping)

# Show Show Variance-Covariance Matrix
cov_matrix = df.select_dtypes(include=["number"]).cov()
print("Variance-Covariance Matrix:")
print(cov_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(cov_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Variance-Covariance Matrix")
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=["number"]).corr(), annot=True, cmap="magma")
plt.title("Correlation Matrix")
plt.show() # We can interpret that carat and the size(x,y,z) have highly positive corretion with price from the matrix


#Step 2 Feature Engineering & know the Degree of dispersion of key data points
# Create new features, define new column 
df["carat_root"] = np.sqrt(df["carat"])
df["log_price"] = np.log(df["price"])

# Show the Coefficient of Variation(CV) of price, carat, cut_num
# A higher CV indicates more dispersed data
# Calculate the mean and the standard deviation of price, carat, cut_num
price_mean = np.mean(df["price"])
price_std = np.std(df["price"], ddof=0)  #ddof=0 means degrees of freedom = 0, which is used for population standard deviation
price_cv = price_std / price_mean
print(f"price_cv: {price_cv:.2f}") # price_CV=1.01, which means extreme variability 
log_price_mean = np.mean(df["log_price"])
log_price_std = np.std(df["log_price"], ddof=0)  
log_price_cv = log_price_std / log_price_mean
print(f"log_price_cv: {log_price_cv:.2f}") # log_price_CV=0.13, as we use logarithm transformation to scale down highly dispersed data
carat_mean = np.mean(df["carat"])
carat_std = np.std(df["carat"], ddof=0)  
carat_cv = carat_std / carat_mean
print(f"carat_cv: {carat_cv:.2f}") # carat_CV=0.59
cut_num_mean = np.mean(df["cut_num"])
cut_num_std = np.std(df["cut_num"], ddof=0)  
cut_num_cv = cut_num_std / cut_num_mean
print(f"cut_num_cv: {cut_num_cv:.2f}")# cut_number_CV=0.38

#Step 3 Scatter Plot of carat_root vs. log_price
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["carat_root"], y=df["log_price"], hue=df["cut"], palette="coolwarm")
plt.xlabel("Square Root of Carat")
plt.ylabel("Log Price")
plt.title("Carat Root vs Log Price")
plt.legend(title="Cut Quality")
plt.show() #The plot indicates that as the carat weight increases, the price also rises exponentially, with higher-quality cuts generally having higher prices compared to lower-quality cuts
#===============================Linear Regression  Model  One====================================================================================
#The Steps 4-6 and part of Step 7 can be encapsulated by a function, say call it "train_linear_regression", in order to avoid unnecessary code duplication:
X_columns = ["carat_root"]
Y_column = "log_price"
model, X_test, Y_test, Y_pred = train_linear_regression(df,X_columns,Y_column)

#Step 7 Measure Model Performance
# Check for overfitting
check_overfitting(model, df[X_columns], df[Y_column], X_test, Y_test, model_name="Model 1")

# Step 8 Plot predicted vs actual values with overlap area(Model one)
# Now we standardize the actual and predicted data to the same scale in order to calculate the overlapping area
hist_actual, bin_edges = np.histogram(Y_test, bins=50, density=True)
hist_predicted, _ = np.histogram(Y_pred, bins=bin_edges, density=True)
overlap_area1 = np.sum(np.minimum(hist_actual, hist_predicted) * np.diff(bin_edges))
print(f"Overlap Area for Model 1: {overlap_area1:.4f}")
plot_actual_vs_predicted(Y_test, Y_pred, overlap_area1, "Model 1")

#==============================Linear Regression  Model  Two================================================================
#Step 9 Add cut_num as a Second Independent Variable(Model two)
X_column = ["carat_root", "cut_num"]
Y_column = "log_price"
model, X_test, Y_test, Y_pred2 = train_linear_regression(df, X_column, Y_column)

# Check for overfitting
check_overfitting(model, df[X_column], df[Y_column], X_test, Y_test, model_name="Model 2")

# Now we standardize the actual and predicted2 data to the same scale in order to calculate the overlapping area
# Get the plot of the overlappig area
hist_actual, bin_edges2 = np.histogram(Y_test, bins=50, density=True)
hist_predicted2, _ = np.histogram(Y_pred2, bins=bin_edges2, density=True)
overlap_area2 = np.sum(np.minimum(hist_actual, hist_predicted2) * np.diff(bin_edges2))
print(f"Overlap Area for Model 2: {overlap_area2:.4f}")
plot_actual_vs_predicted(Y_test, Y_pred2, overlap_area2, "Model 2")
#overlap area 77% is bigger than model one 62%, we can say that adding carat_num improves the regression model!

#==============================Linear Regression  Model Three================================================================
# Additional question: What happens if we filter out "outliers" before setting up the linear regression, 
# so basically using a better dataset at the beginning?
X_column = ["carat_root", "cut_num"]
Y_column = "log_price"
df_filtered = filter_outliers_by_price(df, threshold=1)
model_filtered, X_test, Y_test, Y_pred_filtered = train_linear_regression(df_filtered, X_column, Y_column)
# Check for overfitting
check_overfitting(model_filtered, df_filtered[X_column], df_filtered[Y_column], X_test, Y_test, model_name="Model 3")
# Get the new scatterplot of filtered data
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_filtered["carat_root"], y=df_filtered["log_price"], hue=df_filtered["cut"], palette="coolwarm")
plt.xlabel("Square Root of Carat")
plt.ylabel("Log Price")
plt.title("Carat Root vs Log Price with filtered data")
plt.legend(title="Cut Quality") 
plt.show()

# Now we standardize the actual and predicted data to the same scale in order to calculate the overlap area
# Get the plot of the overlappig area
hist_actual, bin_edges_filtered = np.histogram(Y_test, bins=50, density=True)
hist_predicted_filtered, _ = np.histogram(Y_pred_filtered, bins=bin_edges_filtered, density=True)
overlap_area3 = np.sum(np.minimum(hist_actual, hist_predicted_filtered) * np.diff(bin_edges_filtered))
print(f"Overlap Area for Model 3: {overlap_area3:.4f}")
plot_actual_vs_predicted(Y_test, Y_pred_filtered, overlap_area3, "Model 3")
#The overlap area only increases 6% comparing with Model one
#So we can summarize that filtering the data doesn't improve the model a lot, we should try adding more independent variable!
