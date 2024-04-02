import re
import pandas as pd
import joblib
import customtkinter as ctk
from tkinter import filedialog
import pyperclip
from customtkinter import CTkFont

'''
Quick and dirty UI script
'''

# Define the features X
X_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
             'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']


def shorten_catboost_name(name):
    # catboost models have a stupid long name
    return "CatBoostRegressor" if "catboostcoreCatBoostRegressor" in name else name


def get_model_name_from_pipeline(pipeline):
    model_name = None
    for step in pipeline.steps:
        model_name = re.sub(r'\W+', '', str(step[1]).split("(")[0])
    return model_name


def load_model(model_path):
    return joblib.load(model_path)


def predict_quality(model, input_data):
    prediction = model.predict(input_data)
    return prediction


def validate_numeric_input(value):
    if value == '':
        return True
    try:
        float(value)
        return True
    except ValueError:
        return False


class WineQualityApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.result_frame = None
        self.input_labels = None
        self.input_entries = None
        self.result_label = None
        self.title("Wine Quality Predictor")
        self.geometry("450x710")
        self.resizable(False, True)
        self.configure(padx=20, pady=20)
        self.model_paths = []
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
            entry.configure(validate='key', validatecommand=(self.register(validate_numeric_input), '%P'))
            entry.bind('<KeyRelease>', self.check_fields_filled)
            self.input_entries.append(entry)

        # Predict Button
        self.predict_button = ctk.CTkButton(self, text="Predict", command=self.predict, state='disabled')
        self.predict_button.grid(row=len(X_columns) + 1, column=0, columnspan=2, padx=10, pady=10, sticky='we')

        # Scrolling Frame for Result Label
        self.result_frame = ctk.CTkScrollableFrame(self)
        self.result_frame.grid(row=len(X_columns) + 2, column=0, columnspan=2, padx=10, pady=5,
                               sticky='we')
        self.result_frame.bind("<Button-1>", self.copy_to_clipboard)

        # Result Label
        monospace_font = CTkFont(family="Courier", size=12)
        self.result_label = ctk.CTkLabel(self.result_frame, text=f"Select model(s) and enter data",
                                         justify="left", font=monospace_font)
        self.result_label.pack(side="left", fill="both", expand=False)
        self.result_label.bind("<Button-1>", self.copy_to_clipboard)

        self.grid_rowconfigure(0, weight=0)  # Browse Frame
        self.grid_rowconfigure(1, weight=1)  # Input fields
        self.grid_rowconfigure(len(X_columns) + 1, weight=0)  # Predict Button
        self.grid_rowconfigure(len(X_columns) + 2, weight=1)  # Result Frame
        self.grid_columnconfigure(0, weight=1)  # Labels
        self.grid_columnconfigure(1, weight=1)  # Entries

    def check_fields_filled(self, event):
        if self.model_paths and all(entry.get() for entry in self.input_entries):
            self.predict_button.configure(state='normal')
        else:
            self.predict_button.configure(state='disabled')
            self.result_label.configure(text=f"Select model(s) and enter data")

    def browse_model(self):
        model_paths = filedialog.askopenfilenames(filetypes=[("Pickle Files", "*.pkl")])
        if model_paths:
            self.model_path_entry.delete(0, 'end')
            self.model_paths = model_paths
            self.model_path_entry.insert(0, ", ".join(model_paths))
            self.check_fields_filled(None)

    def predict(self):
        models = []
        model_names = []

        for model_path in self.model_paths:
            models.append(load_model(model_path))
            model_name = get_model_name_from_pipeline(models[-1])
            model_name = shorten_catboost_name(model_name)
            model_names.append(model_name)

        user_input = [entry.get() for entry in self.input_entries]
        input_data = pd.DataFrame([user_input], columns=X_columns)

        max_model_name_length = max(len(model_name) for model_name in model_names)

        predictions = []
        for model, name in zip(models, model_names):
            padded_model_name = name.ljust(max_model_name_length)
            prediction = predict_quality(model, input_data)
            predictions.append((padded_model_name, prediction[0]))

        avg_prediction = sum(pred[1] for pred in predictions) / len(predictions)

        padded_predictions = [f"{pred[0]}\t{pred[1]}" for pred in
                              predictions]

        self.result_label.configure(
            text=f"Predicted Wine Quality: {round(avg_prediction)}\n\nRaw Predictions:\n{'\n'.join(padded_predictions)}",
        )

    def copy_to_clipboard(self, event):
        output_text = self.result_label.cget("text")
        pyperclip.copy(output_text)
        self.show_toast("Text copied to clipboard")

    def show_toast(self, message):
        toast = ctk.CTkToplevel(self)
        toast.geometry("200x50")
        toast.wm_overrideredirect(True)
        toast.wm_geometry(
            "+{}+{}".format(self.winfo_x() + self.winfo_width() - 200, self.winfo_y() + self.winfo_height()))
        label = ctk.CTkLabel(toast, text=message)
        label.pack(fill='both', expand=True, padx=10, pady=10)
        toast.after(1500, toast.destroy)


if __name__ == "__main__":
    app = WineQualityApp()
    app.mainloop()
