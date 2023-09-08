# Prompt Leak Detection
## Overview
This project aims to detect whether a given message is an attempt at a prompt leak. It utilizes various tests on both the input (user-inserted text) and the output of a language model (LLM) to determine if there has been an attempt at a prompt leak.
## Installation
To run this project, you'll need to install the required packages. You can do this using pip by running the following command in your terminal:
`pip install -r requirements.txt`

Make sure you have the necessary dependencies installed.

## Configuration
Secret keys should be inserted into the `secret_key.py` file. These keys may be required for certain functionalities within the program.

## Usage
To use the program, follow these steps:

1. Run the Streamlit application by executing the following command in your terminal:
`streamlit run main.py`
2. Once the application is running, you can enter your prompt into the input text box.
3. The bot will then attempt to detect whether the message is an attempt at a prompt leak.
4. The program will perform various tests on both the input message and the output of the language model.
5. You will receive a system message at the end of the response made by the AI, indicating whether there has been an attempt at a prompt leak or not.
