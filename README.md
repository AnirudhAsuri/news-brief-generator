# News Brief Generator

Get the essence of any news article in seconds, powered by the high-speed Groq API.

***

### Table of Contents
* [About The Project](#about-the-project)
* [Key Features](#key-features)
* [Tech Stack](#tech-stack)
* [Getting Started](#getting-started)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [License](#license)
* [Contact](#contact)

***

## About The Project

In an age of information overload, staying updated with the news can be a time-consuming task. The **News Brief Generator** is a tool designed to solve this problem. By providing a URL to a news article, this application leverages the incredible speed of the Groq™ LPU Inference Engine to deliver a concise, accurate, and easy-to-read summary, allowing you to stay informed without reading through lengthy articles.

## Key Features

* **Blazing Fast Summaries:** Utilizes the Groq API for near-instant summary generation.
* **URL-Based Input:** Simply provide a link to a news article to get started.
* **Accurate & Concise Content:** The AI extracts the most critical points from the article.
* **Clean & Simple UI:** A straightforward interface that focuses on functionality.

***

## Tech Stack

* **Backend:** Python
* **Frontend:** Streamlit
* **LLM Provider:** Groq API
* **Web Scraping:** BeautifulSoup
* **Environment Management:** python-dotenv

***

## Getting Started

Follow these instructions to get a local copy of the project set up and running.

### Prerequisites

* Python 3.9+
* Git

### Installation

1.  **Clone the repository** to your local machine.
    ```sh
    git clone [https://github.com/AnirudhAsuri/news-brief-generator.git](https://github.com/AnirudhAsuri/news-brief-generator.git)
    cd news-brief-generator
    ```

2.  **Create and activate a virtual environment.**
    * For macOS/Linux:
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```
    * For Windows:
        ```sh
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install the required packages.**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables.**
    * Create a file named `.env` in the root directory.
    * Add your Groq API key to this file:
        ```env
        GROQ_API_KEY="gsk_YourSecretApiKeyHere"
        ```

***

## Usage

To start the application, run the following command in your terminal:
```sh
streamlit run app.py
