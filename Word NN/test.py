import json
import yfinance as yf
from requests.models import HTTPError


def get_stock_price(symbol):
    try:
        ticker = yf.Ticker(symbol).info
        market_price = ticker["currentPrice"]
        previous_close_price = ticker["regularMarketPreviousClose"]
        info = {
            "symbol": symbol,
            "current_price": market_price,
            "previous_close_price": previous_close_price,
        }
        return json.dumps(info)
    except HTTPError:
        return None


import openai

openai.api_key = "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"

functions = [
    {
        "name": "get_stock_price",
        "description": "Get current stock price for given stock symbol. If function returns None then it means that symbol is invalid. function returns current_price which is current price of the stock",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol of the company",
                }
            },
            "required": ["symbol"],
        },
    }
]

messages = [{'role': 'user', 'content': "what is the current stock price of alibaba"}]

response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto", 
    )

response_message = response["choices"][0]["message"]

# this one just stores the mapping of the name of the function to actual function
available_functions = {
    "get_stock_price": get_stock_price,
}
# function name passed to us by GPT
function_name = response_message["function_call"]["name"]
# get the function reference 
function_to_call = available_functions[function_name]
# get the arguments of this function passed to us by GPT
function_args = json.loads(response_message["function_call"]["arguments"])
# Call the function reference by passing the argument.
function_args(['symbol'])

#append the response_message to our input message array. 
#This is required for GPT to establish context between function calls and results.
messages.append(response_message)
#append the actual function response to the same message array.
messages.append(
    {
        "role": "function",
        "name": function_name,
        "content": str(function_response),
    }
)
#Call gpt again with input message, which has three things now. 
#1. Actual question "what is the current stock price of alibaba"
#2. Derived function name and argument from about question
#3. JSON result of the function call.
second_response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
)  # get a new response from GPT where it can see the function response

print(second_response["choices"][0]["message"]["content"])
