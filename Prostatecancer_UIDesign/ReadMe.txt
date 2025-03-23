To run this project in VS Code, follow these steps:

First, create a new Python virtual environment and install the required packages:
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install streamlit pandas numpy scikit-learn plotly statsmodels
Create a project directory and copy these files into it:
app.py
utils.py
models.py
Create a .streamlit folder and add the config.toml file
Place your dataset (marksheet.csv) in the project directory

Run the Streamlit app:

streamlit run app.py
The app will be available at http://localhost:8501