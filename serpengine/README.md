# Link Search Agent

**Link Search Agent** helps you collect relevant links from popular search engines. We are using this module to collect product information-related URLs from the web.

## Installation

1. **Clone the Repository:**

    ```bash
    git clone [repo]
    ```

2. **Activate Your Project Environment:**

    Ensure you have your Python environment activated (e.g., using `venv`, `virtualenv`, or `conda`).

3. **Navigate to the Link Search Agent Folder and Install:**

    ```bash
    cd link_search_agent
    pip install -e .
    ```

    This installs the package in editable mode, allowing you to make changes to the source code that are immediately reflected.

## Usage

Using the **Link Search Agent** is straightforward. Simply initialize the `RelevantLinkSearcher` and call the `collect` method with your desired query and parameters.

### Example

```python
from relevant_link_searcher import RelevantLinkSearcher

# Initialize the searcher
link_searcher = LinkSearchAgent()

# Collect links based on a query
result_data = link_searcher.collect(
    query="STM32 Microprocessor",
    num_urls=5,
    search_sources=["google_search_via_api", "google_search_via_request_module"],
    keyword_match_based_link_validation=["STM32"],
    allowed_domains=["digikey.com"],
    output_format="json"  # or "linksearch"
)

print(result_data)
```

### Parameters

- `query` (str): The search query.
- `validation_conditions` (Dict, optional): Additional validation rules for filtering links.
- `num_urls` (int): Number of links to retrieve.
- `search_sources` (List[str]): Search sources to use (e.g., `"google_search_via_api"`, `"google_search_via_request_module"`).
- `allowed_countries` (List[str], optional): List of country codes to allow.
- `forbidden_countries` (List[str], optional): List of country codes to forbid.
- `allowed_domains` (List[str], optional): List of domains to allow.
- `forbidden_domains` (List[str], optional): List of domains to block.
- `filter_llm` (bool, optional): Whether to use AI-based filtering.
- `output_format` (str): Output format, either `"json"` or `"linksearch"`.

### Output

- **JSON Format:**

    ```json
    {
        "operation_result": {
            "total_time": 1.234,
            "errors": []
        },
        "results": [
            {
                "link": "https://digikey.com/product1",
                "metadata": "",
                "title": ""
            },
            ...
        ]
    }
    ```

- **LinkSearch Objects:**

    A list of `LinkSearch` objects with attributes `link`, `metadata`, and `title`.

## Features

- **Search Modules:**
  
  - **Simple Google Search Module:** Scrapes Google search results directly from the HTML.
  - **Google Search API Module:** Utilizes the Google Custom Search API for fetching search results.

- **Filters:**

  - **Allowed Domains:**
    - **Description:** Restricts search results to specified domains. For example, setting `allowed_domains=["digikey.com"]` ensures only links from Digi-Key are collected.
  
  - **Keyword Match Based Link Validation:**
    - **Description:** Ensures that the collected links contain specific keywords. For instance, `keyword_match_based_link_validation=["STM32"]` filters out any links that do not include the keyword "STM32".

  - **Allowed Countries (Optional):**
    - **Description:** Filters links based on the top-level domain (TLD) to include only those from specified countries.

  - **Forbidden Countries (Optional):**
    - **Description:** Excludes links from specified countries based on their TLD.

  - **Additional Validation Conditions:**
    - **Description:** Allows for custom validation logic to further filter links based on user-defined criteria.

- **Output Formats:**
  
  - **JSON:** Provides a structured dictionary containing operation results and the list of collected links.
  - **LinkSearch Objects:** Returns a list of `LinkSearch` dataclass instances for flexible manipulation within Python.

- **Error Handling and Logging:**
  
  - Captures and logs errors encountered during the search and filtering processes, facilitating easier debugging and maintenance.

- **Extensibility:**
  
  - Designed to be easily extendable, allowing integration of additional search sources or more sophisticated filtering mechanisms as needed.

## Requirements

Ensure you have the following dependencies installed. They are listed in the `requirements.txt` file:

```plaintext
requests>=2.25.1
python-dotenv>=0.19.0
beautifulsoup4>=4.9.3
```

You can install them via:

```bash
pip install -r requirements.txt
```

## Configuration

Before using the Link Search Agent, set up your environment variables:

1. **Create a `.env` File:**

    ```env
    GOOGLE_API_KEY=your_google_api_key
    GOOGLE_CSE_ID=your_custom_search_engine_id
    ```

2. **Ensure the `.env` File is in the Root Directory or the Directory Where the Script Runs.**



