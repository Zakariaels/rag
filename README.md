# RAG TUTORIAL - BASIC [Embedding / Indexing / Retrieval / Generation]

## Setup

### Environment Variables

1. Copy the `.env.example` file to a new file named `.env`:

    ```
    cp .env.example .env
    ```

2. Open the `.env` file and replace the placeholder values with your actual configuration:

    ```
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
    LANGCHAIN_API_KEY=<your-langchain-api-key>
    LANGCHAIN_PROJECT=<your-langchain-project>
    OPENAI_API_KEY=<your-openai-api-key>
    USER_AGENT=my_custom_agent
    ```

    Replace the following:

    - `<your-langchain-api-key>`: Your LangChain API key
    - `<your-langchain-project>`: Your LangChain project name
    - `<your-openai-api-key>`: Your OpenAI API key

    Note:

    - `LANGCHAIN_TRACING_V2` is set to `true` by default.
    - `LANGCHAIN_ENDPOINT` is pre-configured.
    - `USER_AGENT` is set to "my_custom_agent". Modify if needed.

### Dependencies

This project uses pip-tools to manage dependencies. Follow these steps to set up your environment:

1. Create a virtual environment:

    ```
    python -m venv venv
    ```

2. Activate the virtual environment:

    - On Windows:
        ```
        venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```
        source venv/bin/activate
        ```

3. Install pip-tools:

    ```
    pip install pip-tools
    ```

4. Install project dependencies:
    ```
    pip-sync
    ```

## Using pip-tools

We use pip-tools to manage project dependencies. Here's how to use it:

### Adding new dependencies

1. Add the new package to `requirements.in`.
2. Run `pip-compile requirements.in` to update `requirements.txt`.
3. Run `pip-sync` to install the new package.

### Updating dependencies

1. Update the version in `requirements.in` (if it's a direct dependency).
2. Run `pip-compile --upgrade requirements.in` to upgrade all packages.
3. Run `pip-sync` to apply the changes to your environment.

### Removing dependencies

1. Remove the package from `requirements.in`.
2. Run `pip-compile requirements.in` to update `requirements.txt`.
3. Run `pip-sync` to apply the changes to your environment.

Remember to commit both `requirements.in` and `requirements.txt` after making changes.

## Running the Project

1. Ensure your virtual environment is activated.
2. Run the main script:
    ```
    python src/basic_rag/main.py
    ```
