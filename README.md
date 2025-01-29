# Resume Parser

This project is a web application that extracts structured data from a resume in PDF format using Streamlit and OpenAI's GPT-4 model.

## Features

- Upload a resume in PDF format
- Extract structured data such as education, work experience, skills, and links
- Download the extracted data in JSON format

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Copy the [.env.sample](http://_vscodecontentref_/1) to [.env](http://_vscodecontentref_/2) and add your OpenAI API key:
    ```sh
    cp .env.sample .env
    ```

## Usage

1. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload a resume in PDF format and click "Process Resume" to extract the data.

## Project Structure

- [app.py](http://_vscodecontentref_/3): Main application file for the Streamlit app.
- [Resume.py](http://_vscodecontentref_/4): Contains the Pydantic models for the resume data.
- [requirements.txt](http://_vscodecontentref_/5): Lists the Python dependencies.
- [.env.sample](http://_vscodecontentref_/6): Sample environment file for storing the OpenAI API key.
- [.gitignore](http://_vscodecontentref_/7): Specifies files and directories to be ignored by Git.

## License

This project is licensed under the MIT License.