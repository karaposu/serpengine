````markdown
# SERPEngine

**SERPEngine** – Production-grade search module to find links through search engines.

* uses Google Search API  
* made for production – you need API keys  
* includes various filters (including an LLM-based one) so you can filter links by domain, metadata, etc.  
* returns structured dataclasses (`SearchHit`, `SERPMethodOp`, `SerpEngineOp`)

---

## 1. Installation

```bash
pip install serpengine
````

---

## 2. Environment variables

Create a `.env` (or export vars manually):

```env
GOOGLE_SEARCH_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXX
GOOGLE_CSE_ID=yyyyyyyyyyyyyyyyyyyyyyyyy:zzz
```

Both values are required; the engine will raise if either is missing.

---

## 3. Dataclass cheat-sheet

| Class          | What it represents                                                                                                                                                 |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `SearchHit`    | One URL result (`link`, `title`, `metadata`).                                                                                                                      |
| `UsageInfo`    | Billing info (currently just `cost: float`).                                                                                                                       |
| `SERPMethodOp` | Output of **one** search method.<br>Fields: `name`, `results: List[SearchHit]`, `usage`, `elapsed_time`.                                                           |
| `SerpEngineOp` | Aggregated result of a full `collect()` call.<br>Fields: `usage`, `methods: List[SERPMethodOp]`, `results`, `elapsed_time`.<br>➕ helper `all_links() -> List[str]` |

---

## 4. Quick-start (sync)

```python
from serpengine.serpengine import SERPEngine

engine = SERPEngine()

op = engine.collect(
    query="best pizza in Helsinki",
    num_urls=5,
    search_sources=["google_search_via_api"],   # or add "google_search_via_request_module"
    output_format="object"                      # default
)

print(op.elapsed_time, "sec")
print(op.all_links())
```

---

## 5. Quick-start (async)

```python
import asyncio
from serpengine.serpengine import SERPEngine

async def main():
    eng = SERPEngine()
    op = await eng.collect_async(
        query="python tutorials",
        num_urls=6,
        search_sources=["google_search_via_api", "google_search_via_request_module"],
        output_format="object"
    )
    print(op.all_links())

asyncio.run(main())
```

LM filtering are applied automatically.

---

## Getting Google Credentials

### 1. Create or Select a Google Cloud Project

1. Open the **[Google Cloud Console](https://console.cloud.google.com/)**.
2. Either **create a new project** or **select an existing one** from the project picker.

### 2. Enable the **Custom Search API**

1. In the left-hand menu, navigate to **APIs & Services → Library**.
2. Search for **“Custom Search API”**.
3. Click the result, then press **Enable**.

### 3. Create Credentials (API Key)

> This becomes your **`GOOGLE_SEARCH_API_KEY`**.

1. Still under **APIs & Services**, choose **Credentials**.
2. Click **Create Credentials → API key**.
3. Copy the key shown in the dialog and keep it safe.

---

## Getting the Custom Search Engine ID (`GOOGLE_CSE_ID`)

### 1. Open Google Custom Search Engine

Go to **[cse.google.com/cse](https://cse.google.com/cse/)**.

### 2. Create a New Search Engine

1. Click **Add** (or **“New Search Engine”**).
2. In **“Sites to search”**, you can:

   * Enter a specific domain (e.g., `example.com`) **or**
   * Use a wildcard like `*.com` (if you intend to search the entire web—later you can enable *Search the entire web* in the control panel).
3. Give your CSE a **name**, then click **Create**.

### 3. Retrieve Your CSE ID

1. Open the **Control Panel** for the search engine you just created.
2. Locate the **“Search engine ID”** (sometimes labeled **cx**).
3. Copy that string—this is your **`GOOGLE_CSE_ID`**.

---

## Next Steps

Add both credentials to your environment, e.g. in a `.env` file:

```env
GOOGLE_SEARCH_API_KEY=YOUR_API_KEY_HERE
GOOGLE_CSE_ID=YOUR_CSE_ID_HERE
```

Then load them in your code (the **SERPEngine** does this automatically with `python-dotenv`).

---

### Output

* **JSON Format:**

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
          }
      ]
  }
  ```

* **Filters:**

  * **Allowed Domains:** Restricts search results to specified domains.
    Example: `allowed_domains=["digikey.com"]`
  * **Keyword Match Based Link Validation:** Ensures links contain certain keywords.
    Example: `keyword_match_based_link_validation=["STM32"]`
  * **Allowed Countries (Optional):** Filters links by TLD to include only specified countries.
  * **Forbidden Countries (Optional):** Excludes links from specified countries based on their TLD.
  * **Additional Validation Conditions:** Custom logic to further filter links.

* **Output Formats:**

  * **JSON:** Structured dictionary with operation results and links.
  * **Objects:** List of `LinkSearch` dataclass instances for flexible manipulation.

* **Error Handling and Logging:** Captures and logs errors during search and filtering for easier debugging.

* **Extensibility:** Designed for easy extension—add new search sources or advanced filters as needed.

```
```
