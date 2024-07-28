import streamlit as st
import pandas as pd
import os
from pandasai.llm.openai import OpenAI
from pandasai import SmartDataframe
from pandasai.llm import AzureOpenAI
from pandasai.responses.response_parser import ResponseParser
# from pandasai.responses.streamlit_response import StreamlitResponse
from langchain.agents.agent_types import AgentType
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain.llms import AzureOpenAI


### Functions

# Modify the output
def mod_output(output):
    mod_out = output.split("\n")
    final_out = mod_out[0]
    return final_out

def prepare_dataframe(df):
    """
    Make data column names lowercase and replace spaces with underscores
    """
    df.columns = [x.replace(' ', '_').lower() for x in df.columns]
    return df

# Streamlit UI setup
st.set_page_config(page_title="Chat with Your Dataset", page_icon=":robot:")
st.header("Let's have a chat about your data...")
st.write("This app allows you to select a CSV file and ask questions about it.")

# Path to the directory containing CSV files
data_directory = "data"

# Get a list of CSV files in the data directory
csv_files = [file for file in os.listdir(data_directory) if file.endswith('.csv')]
dataset_names = [file.replace('.csv', '') for file in csv_files]

# Dropdown to select the dataset
selected_dataset = st.selectbox("Select a Dataset:", dataset_names)

placeholder = st.empty()

def get_text():
    input_text = placeholder.text_input("Enter your question: ", value="", key="input")
    return input_text

user_input = get_text()

class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return

def read_csv_with_dtype_conversion(file_path):
    # Read the CSV file to inspect column dtypes
    df = pd.read_csv(file_path)
    # print("df", df.dtypes)

    # Dictionary to store columns to convert to desired dtypes
    convert_to_dtype = {}

    # Iterate over columns to identify float64 and int64 columns
    for col_name, dtype in df.dtypes.items():
        if dtype == 'float64':
            convert_to_dtype[col_name] = 'float32'
        elif dtype == 'int64':
            convert_to_dtype[col_name] = 'int32'

    # Read CSV file again with specified dtype conversions
    df_converted = pd.read_csv(file_path, dtype=convert_to_dtype)
    # print("df_converted", df_converted.dtypes)

    return df_converted


# Load selected CSV file
if selected_dataset:
    file_name = selected_dataset + '.csv'
    file_path = os.path.join(data_directory, file_name)
    df = read_csv_with_dtype_conversion(file_path)
    # df = pd.read_csv(file_path)
    df = prepare_dataframe(df)

    # Initialize the AI model (OpenAI or AzureOpenAI)
    azure_llm = AzureOpenAI(
        azure_endpoint = "url",
        deployment_name = "gpt-35-turbo",
        api_version = "2023-05-15",
        api_token = "api_token",
        openai_api_type = "azure",
        model_name="gpt-3.5-turbo",
        temperature=0, 
    )

    agent = create_pandas_dataframe_agent(
        azure_llm,
        df,
        verbose=True,
    )

    # Create SmartDataframe instance with the configured AI model
    smart_df = SmartDataframe(
        df, 
        config={
            "llm": azure_llm,
            'response_parser': StreamlitResponse,
            },   
            )

    # Process user input upon button click
    if st.button("Submit") and user_input:
        try:
            # Chat with the SmartDataframe and get response
            # response = smart_df.chat(user_input)
            response = agent.run(user_input)

            # Display the response
            if response is not None:
                st.write(response)
                if 'output' in response:
                    final_output = response['output'].strip() 
                    st.write(final_output)
                else:
                    pass
            else:
                pass

        except Exception as e:
            st.error(f"Error: {e}")