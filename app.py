import pandas as pd
# from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.llm import AzureOpenAI

df = pd.read_csv("data/test_data_cluster_02.csv")
print(df)
# print(df.head())

openai_llm = OpenAI(
    api_token="token",
)

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

df1 = SmartDataframe(df, config={"llm": openai_llm})
df2 = SmartDataframe(df, config={"llm": azure_llm})

print(df1.chat("How many Watermelons in this table?"))
print(df2.chat("How many Watermelons in this table?"))