import pandas as pd
import joblib
import customtkinter as ctk
from tkinter import filedialog

'''
Quick and dirty UI script
'''


def load_model(model_path):
    return joblib.load(model_path)


def predict_quality(model, input_data):
    prediction = model.predict(input_data)
    return prediction


class WineQualityApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Wine Quality Predictor")

        self.geometry("450x537")
        self.resizable(False, False)
        self.configure(padx=20, pady=20)

        self.model_path = None
        self.create_widgets()

    def create_widgets(self):
        # Browse Frame
        browse_frame = ctk.CTkFrame(self)
        browse_frame.grid(row=0, column=0, columnspan=2, padx=(0, 10), pady=5, sticky='we')

        browse_button = ctk.CTkButton(browse_frame, text="Browse", command=self.browse_model)
        browse_button.grid(row=0, column=0, padx=(0, 10), pady=5, sticky='w')

        self.model_path_entry = ctk.CTkEntry(browse_frame, width=25)
        self.model_path_entry.grid(row=0, column=1, padx=(0, 10), pady=5, sticky='we')

        browse_frame.columnconfigure(1, weight=1)

        # Input Fields
        self.input_labels = []
        self.input_entries = []

        for i, column in enumerate(X_columns):
            label = ctk.CTkLabel(self, text=column)
            label.grid(row=i + 1, column=0, sticky='w', padx=(0, 10), pady=(5, 0))
            self.input_labels.append(label)

            entry = ctk.CTkEntry(self, width=8)
            entry.grid(row=i + 1, column=1, pady=(5, 0), sticky='we')
            entry.configure(validate='key', validatecommand=(self.register(self.validate_numeric_input), '%P'))
            entry.bind('<KeyRelease>', self.check_fields_filled)
            self.input_entries.append(entry)

        # Predict Button
        self.predict_button = ctk.CTkButton(self, text="Predict", command=self.predict, state='disabled')
        self.predict_button.grid(row=len(X_columns) + 1, column=0, columnspan=2, padx=10, pady=10, sticky='we')

        # Result Label
        self.result_label = ctk.CTkLabel(self, text="Enter data and press predict")
        self.result_label.grid(row=len(X_columns) + 2, column=0, columnspan=2, padx=10, pady=5, sticky='w')

        self.grid_rowconfigure(0, weight=0)  # Browse Frame
        self.grid_rowconfigure(1, weight=1)  # Input fields
        self.grid_rowconfigure(len(X_columns) + 1, weight=0)  # Predict Button
        self.grid_rowconfigure(len(X_columns) + 2, weight=0)  # Result Label
        self.grid_columnconfigure(0, weight=1)  # Labels
        self.grid_columnconfigure(1, weight=1)  # Entries

    def validate_numeric_input(self, value):
        if value == '':
            return True
        try:
            float(value)
            return True
        except ValueError:
            return False

    def check_fields_filled(self, event):

        if self.model_path and all(entry.get() for entry in self.input_entries):
            self.predict_button.configure(state='normal')
        else:
            self.predict_button.configure(state='disabled')

    def browse_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
        if model_path:
            self.model_path_entry.delete(0, 'end')
            self.model_path_entry.insert(0, model_path)
            self.model_path = model_path
            self.check_fields_filled(None)

    def predict(self):
        model = load_model(self.model_path)
        user_input = [entry.get() for entry in self.input_entries]
        input_data = pd.DataFrame([user_input], columns=X_columns)
        prediction = predict_quality(model, input_data)
        self.result_label.configure(text=f"Predicted Wine Quality: {round(prediction[0])}\nRaw Prediction: {prediction[0]}", justify="left")


# Define the features X
X_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
             'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

if __name__ == "__main__":
    app = WineQualityApp()
    app.mainloop()
