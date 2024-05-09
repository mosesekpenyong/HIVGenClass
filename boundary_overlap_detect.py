import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from google.colab import drive

def mark_point_with_ellipse(x, y, min_freq, max_freq):
    plt.scatter(x, y)  # Plot the point
    plt.errorbar(x, y, xerr=min_freq, yerr=max_freq, fmt='o')  # Draw ellipse around the point
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Point with Ellipse')
    plt.show()

def read_input_from_excel(file_path):
    df = pd.read_excel(file_path)
    x = df['X'].values  # Get X values from the Excel file
    y = df['Y'].values  # Get Y values from the Excel file
    min_freq = df['Min_Frequency'].values  # Get minimum frequency values
    max_freq = df['Max_Frequency'].values  # Get maximum frequency values
    return x, y, min_freq, max_freq

# Mount Google Drive
drive.mount('/content/drive')

# Load the dataset
excel_file_path = '/content/drive/My Drive/PythonCodeFiles/Dataset.xlsx'
x, y, min_freq, max_freq = read_input_from_excel(excel_file_path)
for i in range(len(x)):
    mark_point_with_ellipse(x[i], y[i], min_freq[i], max_freq[i])  # Mark each point with an ellipse