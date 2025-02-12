# SERPEngine 

**SERPEngine** Production grade search module to find links through search engines. 
- uses google search API
- made for production.  You need API keys
- includes various filters including LLM based one. So you can filter the links based on domain, metadata, 


## Installation

1. **Clone the Repository:**

    ```bash
    pip install serpengine [repo]
    ```

2. Usage: 


```python
from serpengine import SERPEngine

# Initialize the searcher
serpengine = SERPEngine()

    result_data = serpengine.collect(
        query="best food in USA",
        num_urls=5,
        search_sources=["google_search_via_api"],
        regex_based_link_validation=False,             
        allow_links_forwarding_to_files=False,        
        output_format="json"  # or "linksearch"
    )
    print(result_data)
```


## Getting Google Credentials:

Create or Select a Google Cloud Project:
Go to the Google Cloud Console.
Create a new project (or select an existing one).
Enable the Custom Search API:

In the Cloud Console, navigate to APIs & Services > Library.
Search for "Custom Search API".
Click on it and then press the "Enable" button.
Create Credentials (API Key):

Once the API is enabled, go to APIs & Services > Credentials in the sidebar.
Click "Create Credentials" and choose "API key".
A dialog will display your new API key. Copy this key.

Getting the Custom Search Engine ID (GOOGLE_CSE_ID)
This ID tells the API which search engine configuration to use.

Visit the Google Custom Search Engine (CSE) Site:

Go to Google Custom Search Engine.
Create a Custom Search Engine:

Click "Add" or "New Search Engine".
In the "Sites to search" field, you can either enter a specific website (if you want to restrict the search) or enter a placeholder like *.com to allow broader searches.
Fill in the other required fields (like a name for your search engine) and click "Create".
Retrieve Your CSE ID:

Once your search engine is created, go to its Control Panel.

Look for the "Search engine ID" (often labeled as cx). It will be a string of characters.

Copy this ID.





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



