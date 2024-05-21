#Imports 
import tkinter as tk  # Import tkinter for GUI elements
from tkinter import ttk, filedialog  # Import ttk for themed widgets and filedialog for file selection
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Import FigureCanvasTkAgg for embedding matplotlib plots in Tkinter
import pandas as pd  # Import pandas for data manipulation
from scipy.optimize import curve_fit  # Import curve_fit for fitting functions to data
import numpy as np  # Import numpy for numerical operations
import os 



# All frames 




# Define the functions and their forms, dictionary 
functions = {
    "Biexponential": "f(t) = B + A1 * e^(-t/tau1) + A2 * e^(-t/tau2)",  # Formula for biexponential function
    "Monoexponential": "f(t) = B + A * e^(-t/tau)",  # Formula for monoexponential function
    "Stretched Exponential": "f(t) = f0 + A * e^(-t/tau)^beta"  # Formula for stretched exponential function
}



# Store parameter values in a dictionary
parameter_values = {}
default_values={}



# Main window setup
root = tk.Tk()  # Creates main window/frame where all the frames will live in 
root.title("Graphical User Interface: Main")  # Set window title
root.geometry("800x600")  # Window size 800 x 600



# 1. File selection frame 
file_frame = ttk.Frame(root)  # Create child frame widget for file selection
file_frame.pack(side=tk.TOP, fill=tk.X)  # Pack frame at the top and horizontally 


file_path_var = tk.StringVar()  # Variable to store selected file path, instance of tk.StringVar



# Function to select a file
def select_file():
    file_path = filedialog.askopenfilename()  # Opens user file dialogue to select a file, stored in file_path
    file_name = os.path.basename(file_path)
    file_path_var.set(file_name)
    if file_path: #If user selected a file
        try:
            data = pd.read_csv(file_path)  # Read the file as a CSV and puts into a pandas DataFrame
            if 't' in data.columns and 'y' in data.columns:  # Check if file contains 't' and 'y' columns
                parameter_values['data'] = data  # Store DataFrame as 'data' in parameter_values dict





                # Calculate and store min and max values for x (t) and y
                x_min, x_max = data['t'].min(), data['t'].max()
                y_min, y_max = data['y'].min(), data['y'].max()
                

                # Set default parameter values
                default_values['A_min'], default_values['A_max'] = y_min, y_max
                default_values['A1_min'], default_values['A1_max'] = y_min, y_max
                default_values['A2_min'], default_values['A2_max'] = y_min, y_max
                default_values['tau_min'], default_values['tau_max'] = x_min, x_max
                default_values['tau1_min'], default_values['tau1_max'] = x_min, x_max
                default_values['tau2_min'], default_values['tau2_max'] = x_min, x_max

                # Initialize parameter values with default values if not already set
                for key, value in default_values.items():
                    if key not in parameter_values:
                        parameter_values[key] = value
                

                # Initialize parameter values with default values
                parameter_values.update(default_values)


                # Print the parameter values for debugging
                print("Default parameter values after file selection:", parameter_values)



                # Plot the raw data
                ax.clear()
                ax.plot(data['t'], data['y'], 'o', label='Data', color='black')
                ax.set_xlabel('t')
                ax.set_ylabel('y')
                canvas.draw()
            else:
                tk.messagebox.showerror("Error", "The file must contain 't' and 'y' columns.")  # Show error if columns are missing
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to read the file: {e}")  # Show error if file read fails



#File button 
file_button = ttk.Button(file_frame, text="Select File", command=select_file)  # Button prompting file select, lives in file frame
file_button.pack(side=tk.LEFT, padx=5, pady=5)  # Pack button on the left with padding



#File label 
file_label = ttk.Label(file_frame, textvariable=file_path_var)  # Label to display selected file path
file_label.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=5, expand=True)  # Pack label to fill x-axis with padding and expansion

# Function to plot a simple parabola (dummy function)
def plot_parabola():
    t = np.linspace(0, 10, 100)  
    y = t**2  
    ax.plot(t, y, label='y = t^2')
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    canvas.draw()

    results = [0,1,2,3]

#Fit and plot button 
fit_plot_button = ttk.Button(file_frame, text="Fit and Plot", command=plot_parabola)  # Button to fit and plot (dummy for now)
fit_plot_button.pack(side=tk.RIGHT, padx=5, pady=5)  # Pack button on the right with padding




# 2. Function selection frame 
function_frame = ttk.Frame(root)  # Create frame for function selection
function_frame.pack(side=tk.TOP, fill=tk.X)  # Pack frame at the top (right below file frame), filling horizontally 



# "Function" label
function_label = ttk.Label(function_frame, text="Function:")  # Label for function selection, placed in function_frame
function_label.pack(side=tk.LEFT, padx=5)  # Pack label on the left with padding

function_var = tk.StringVar(value="Biexponential")  # Variable to store selected function, default to "Biexponential"



# Function scrolldown menu 
function_menu = ttk.OptionMenu(function_frame, function_var, "Biexponential", *functions.keys())  # Dropdown menu to select function, default="Biexponential", list out all function names, selected function stored in function_var
function_menu.pack(side=tk.LEFT, padx=5)  # Pack dropdown menu on the left with padding



# Function display 
function_form_label = ttk.Label(function_frame, text=functions["Biexponential"])  # Label to display selected function's formula, the string call to functions will change accordingly
function_form_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)  # Pack label to fill x-axis with expansion and padding



# Main frame for parameter and results frames
parameters_results_frame = ttk.Frame(root)  # Create frame for parameters and results
parameters_results_frame.pack(side=tk.TOP, fill=tk.X, padx=10)  # Pack frame at the top (right below function frame), filling horizontally



# 3. Parameter boundaries frame 
parameter_frame = ttk.LabelFrame(parameters_results_frame, text="Parameter Boundaries")  # Create frame for parameter boundaries
parameter_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)  # Pack frame on the left (right of parameter boundaries), filling x-axis horizontally



# 4. Results frame 
results_frame = ttk.LabelFrame(parameters_results_frame, text="Results")  # Create frame for results
results_frame.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=5)  # Pack frame on the left, filling x-axis with padding




# 5. Plot frame 
plot_frame = ttk.Frame(root)  # Create frame for plot
plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)  # Pack frame at the top, filling both axes with expansion and padding



#Create dummy plot figure 
figure = plt.Figure(figsize=(6, 4), dpi=100)  # Create matplotlib figure, 6 in x 4 in, 100 dots per inch resolution
ax = figure.add_subplot(111)  # Add subplot to the figure, 1x1 
canvas = FigureCanvasTkAgg(figure, plot_frame)  # Create canvas that allows embedding figure in Tkinter, matplotlib figure into the plot frame
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # Pack canvas to fill both axes with expansion, fills plot frame




#Functions 




# 1. Function to update stored parameter values (inside dict) based on checkbox state, called everytime state changes
def update_stored_parameters(param, var, min_entry, max_entry):
    if var.get():
        parameter_values[param + "_min"] = min_entry.get()
        parameter_values[param + "_max"] = max_entry.get()
    else:
        # Reset to default values 
        parameter_values[param + "_min"] = default_values.get(param + "_min", parameter_values[param + "_min"])
        parameter_values[param + "_max"] = default_values.get(param + "_max", parameter_values[param + "_max"])
    
    # Print the parameter values for debugging
    print("Parameter values after checkbox change:", parameter_values)


def on_focus_in(event, entry, default_text):
    if entry.get() == default_text:
        entry.delete(0, tk.END)
        entry.config(foreground='black')

def on_focus_out(event, entry, default_text):
    if entry.get() == "":
        entry.insert(0, default_text)
        entry.config(foreground='grey')



# 2. Function to update the function form and parameter boundaries display based on the selected function from function menu, creates + deals with all the checkboxes and labels/widgets to the right of the boxes too
def update_function_form_and_parameter_boundaries(*args): #*args: function is a callback function from function_var.trace

    #Function form (menu and label)

    #Changes the function menu text and the function labeling each time a user selects a function 
    selected_function = function_var.get()  # Gets user selected function from function menu 
    function_form_label.config(text=functions[selected_function])  # Update function form label functions call to the dynamic selected function variable
    
    # Reset to default values every time user selects a function
    parameter_values.clear()
    parameter_values.update(default_values)

    # Print the parameter values for debugging
    print("Parameter values after function change:", parameter_values)


    # Clear the parameter inputs everytime user selects a function
    for child in parameter_frame.winfo_children():
        child.destroy()


    #Parameter boundaries 

    # Function name to parameters dictionary, gets right set of parameters according to selected function, makes parameters displayed dynamic
    parameter_labels = {
        "Biexponential": ["B", "A1", "tau1", "A2", "tau2"],
        "Monoexponential": ["B", "A", "tau"],
        "Stretched Exponential": ["f0", "A", "tau", "beta"]
    }[selected_function]  



    for i, param in enumerate(parameter_labels):  # For loop for adding checkboxes and entries for each parameter

        check_var = tk.IntVar(value=0)  # 0/1 variable for checkbox state

        lower_bound_entry = ttk.Entry(parameter_frame, width=10)  # Entry box for lower bound, 10 characters wide
        upper_bound_entry = ttk.Entry(parameter_frame, width=10)  # Entry box for upper bound, 10 characters wide


        # Set default text
        lower_bound_entry.insert(0, str(default_values.get(param + "_min", "")))
        upper_bound_entry.insert(0, str(default_values.get(param + "_max", "")))

        lower_bound_entry.bind("<FocusIn>", lambda event, entry=lower_bound_entry: on_focus_in(event, entry, param + "_min"))
        lower_bound_entry.bind("<FocusOut>", lambda event, entry=lower_bound_entry: on_focus_out(event, entry, param + "_min"))
        upper_bound_entry.bind("<FocusIn>", lambda event, entry=upper_bound_entry: on_focus_in(event, entry, param + "_max"))
        upper_bound_entry.bind("<FocusOut>", lambda event, entry=upper_bound_entry: on_focus_out(event, entry, param + "_max"))


        #Creates checkbox widget and handles what happens when they are checked and unchecked:
        #check_var set to 1 when on and 0 when off, command runs lambda function every time state of checkbox changes
        #The lambda function calls update_parameters with the 0/1 var state, the current looped param (f0, A1, etc), and the lower/upper bound widgets, in turn updating 
        #the parameter_values dictionary appropriately 
        check = ttk.Checkbutton(parameter_frame, variable=check_var, onvalue=1, offvalue=0,
                                command=lambda var=check_var, p=param, me=lower_bound_entry, ma=upper_bound_entry: update_stored_parameters(p, var, me, ma))  # Checkbox to enable/disable parameter
        

        # Creates checkboxes, param labels, and min + max entries: __ < f0 < __ format
        check.grid(row=i, column=0, sticky="e", padx=5, pady=2)
        lower_bound_entry.grid(row=i, column=1, padx=5, pady=2)
        ttk.Label(parameter_frame, text="<").grid(row=i, column=2)
        ttk.Label(parameter_frame, text=f"{param}").grid(row=i, column=3, padx=5, pady=2, sticky="ew")
        ttk.Label(parameter_frame, text="<").grid(row=i, column=4)
        upper_bound_entry.grid(row=i, column=5, padx=5, pady=2)


    

# 3. Function to update the results display based on the selected function from function menu
def update_results_display(*args): #*args: function is a callback function from function_var.trace

    selected_function = function_var.get()  # Get selected function

    #Clear the results frame everytime user selects a function
    for child in results_frame.winfo_children():  # Clear previous results display
        child.destroy()


    # Define result entries based on the selected function
    result_labels = {
        "Biexponential": ["B", "A1", "tau1", "A2", "tau2"],
        "Monoexponential": ["B", "A", "tau"],
        "Stretched Exponential": ["f0", "A", "tau", "beta"]
    }[selected_function]  


    results_vars = {}  # Dictionary to store result variables


    for i, result in enumerate(result_labels):  # Initialize the results variables and create labels and entries for them
        results_vars[result] = tk.StringVar(value="")  # Reset the result variables, can integrate this with real results, just add parameter key-result value pairs to this results_vars dict
        ttk.Label(results_frame, text=f"{result} =").grid(row=i, column=0, sticky="e", padx=5, pady=2)  # Labels for result parameters
        result_label = ttk.Label(results_frame, textvariable=results_vars[result], width=20)  # Label to display result value
        result_label.grid(row=i, column=1, padx=5, pady=2, sticky="w")  # Grid layout for result label




#Ensures that the three update functions are called everytime a function is selected from the menu
function_var.trace('w', update_function_form_and_parameter_boundaries)  # Sets up a trace on function_var variable, whenever function_var is written to/changed, the update_function_form_and_parameter_boundaries function gets called (as well as update_stored_parameters)
function_var.trace('w', update_results_display)  # Sets up a trace on function_var variable, whenever function_var is written to/changed, the update_results_display function gets called



# Initialize the default function form and parameters on startup accordingly based on default function_var ("Biexponential in this case")
update_function_form_and_parameter_boundaries()  # Initialize function form
update_results_display()  # Initialize results display



# Start the GUI loop!!
root.mainloop()  # Start the Tkinter responsive main loop


