import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load and preprocess the data
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/demand.csv")
data = data.dropna()

# Prepare the data for training the model
x = data[["Total Price", "Base Price"]]
y = data["Units Sold"]

# Split the data into training and test sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a Decision Tree Regressor model
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)

# Evaluate the model
ypred = model.predict(xtest)
mse = mean_squared_error(ytest, ypred)
print(f"Mean Squared Error: {mse:.2f}")

# Create the GUI
def predict_demand():
    try:
        # Get input values from the user
        total_price = float(entry_total_price.get())
        base_price = float(entry_base_price.get())

        # Make a prediction using the model
        prediction = model.predict([[total_price, base_price]])[0]

        # Display the result in the result_label
        result_label.config(text=f"Predicted Units Sold: {prediction:.2f}")
    except ValueError:
        # Handle invalid input (non-numeric)
        messagebox.showerror("Input Error", "Please enter valid numeric values for both prices.")

# Create the main window
root = tk.Tk()
root.title("Product Demand Predictor")

# Create labels and entry widgets for the input fields
label_total_price = tk.Label(root, text="Enter Total Price:")
label_total_price.pack(pady=5)
entry_total_price = tk.Entry(root)
entry_total_price.pack(pady=5)

label_base_price = tk.Label(root, text="Enter Base Price:")
label_base_price.pack(pady=5)
entry_base_price = tk.Entry(root)
entry_base_price.pack(pady=5)

# Create a button that will trigger the prediction
predict_button = tk.Button(root, text="Predict Units Sold", command=predict_demand)
predict_button.pack(pady=20)

# Create a label to display the prediction result
result_label = tk.Label(root, text="Predicted Units Sold: ", font=("Helvetica", 14))
result_label.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
