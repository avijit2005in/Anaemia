import streamlit as st
import pandas as pd
import pickle, joblib
import seaborn as sns

# Load the saved model
model = pickle.load(open('./model/knn.pkl', 'rb'))
ct1 = joblib.load('./model/processed_anaemia_scaled')


def predict_batch(data):
    #data.drop(['id'], axis=1, inplace=True)  # Excluding id column
    data.loc[data["Gender"] == 'F', "Gender"] = 0
    data.loc[data["Gender"] == 'M', "Gender"] = 1
    newprocessed1 = pd.DataFrame(ct1.transform(data), columns=data.columns)
    predictions = pd.DataFrame(model.predict(newprocessed1), columns=['diagnosis'])

    final = pd.concat([predictions, data], axis=1)

    return final.round(2)

def predict(data):

    if data.iloc[0][0] == 'F':
        data['Gender'] = data['Gender'].replace('F', 0)
    if data.iloc[0][0] == 'F':
        data['Gender'] = data['Gender'].replace('M', 1)

    newprocessed1 = pd.DataFrame(ct1.transform(data), columns=data.columns)
    prediction = pd.DataFrame(model.predict(newprocessed1), columns=['diagnosis'])
    prediction.index = ['Patient1']
    final = pd.concat([prediction, data], axis=1)
    final.index = ['Patient1']
    return prediction, final.round(2)

#st.title("Anaemia Classification")
st.sidebar.title("Input Patient Test Details")
html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Anaemia Prediction </h2>
    </div>

    """
st.markdown(html_temp, unsafe_allow_html=True)
st.text("")


st.sidebar.header("Input Features")
options_form = st.sidebar.form("options_form")
gender = options_form.text_input("Gender")
hgb = options_form.text_input("Hemoglobin, Range: 11-16 g/dL")
mch = options_form.text_input("Mean cell hemoglobin Range: 27-32 pg")
mchc = options_form.text_input("Mean cell hemoglobin concentration Range: 31-37 g/dL")

add_data = options_form.form_submit_button("Predict")

if add_data:
    data_list = [gender, hgb, mch, mchc]
    data_df = pd.DataFrame([data_list])
    data_df.columns = ['Gender', 'Hemoglobin', 'MCH', 'MCHC']
    data_df.index = ['Patient1']

    st.table(data_df)
    try:
        result, final = predict(data_df)
        cm = sns.light_palette("blue", as_cmap=True)
        st.write(result)
        #st.table(final.style.background_gradient(cmap=cm))
    except:
        st.write("No input data. Fill the form with correct range of values !")

uploaded_file = st.file_uploader("Choose file", type=["csv", "xlsx"], accept_multiple_files=False)

if uploaded_file is not None:
    if uploaded_file.name.lower().endswith('.csv'):
        df_values = pd.read_csv(uploaded_file)
    if uploaded_file.name.lower().endswith('.xlsx'):
        df_values = pd.read_excel(uploaded_file, sheet_name="Sheet1")

if st.button("Predict_Batch"):

    try:
        result = predict_batch(df_values)
        cm = sns.light_palette("blue", as_cmap=True)
        st.table(result.style.background_gradient(cmap=cm))
    except:
        st.write("No input data. Upload csv or excel file !")

