# QA - Bot - Langchain

Brief description or tagline for your project.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Overview

Provide a brief introduction to your project. Explain its purpose, features, and any relevant information.

## Installation

Describe the steps to install and set up the project locally. Include any dependencies or requirements.

# Clone the repository
git clone https://github.com/your-username/your-repo.git

# Change directory
cd your-repo

# Install dependencies
pip install -r requirements.txt
or
poetry install

## Usage

## Run the application
uvicorn main:app --reload


### API Endpoints
Document the available API endpoints along with their functionalities.

``` 
POST /qa
Endpoint to run the QA system.
Accepts a PDF file and a list of questions.
Returns a streaming response with question-answer pairs in JSON format.
```

## Configuration
Explain any configuration settings or environment variables required for the project. Include details about the OpenAI API key and how to set it.
