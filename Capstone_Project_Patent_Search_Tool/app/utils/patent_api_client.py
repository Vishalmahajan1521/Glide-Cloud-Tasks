import requests
from app.core.config import settings

def fetch_patent_data(patent_id: str) -> dict:
    # Using PatentsView API (free, USPTO-backed)
    url = f"https://api.patentsview.org/patents/query?q={{\"patent_number\":\"{patent_id}\"}}&f=[\"patent_title\",\"patent_abstract\",\"patent_description\",\"assignee_organization\",\"patent_date\",\"patent_firstnamed_inventor_name\",\"patent_type\",\"patent_country_code\"]"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if not data.get('patents'):
        raise ValueError(f"No data found for patent {patent_id}")

    patent = data['patents'][0]

    # Extract text
    title = patent.get('patent_title', [''])[0] if patent.get('patent_title') else ''
    abstract = patent.get('patent_abstract', [''])[0] if patent.get('patent_abstract') else ''
    description = patent.get('patent_description', [''])[0] if patent.get('patent_description') else ''
    text = f"{title} {abstract} {description}".strip()

    # Extract metadata
    metadata = {
        'patent_id': patent_id,
        'title': title,
        'assignee': patent.get('assignee_organization', [''])[0] if patent.get('assignee_organization') else '',
        'jurisdiction': patent.get('patent_country_code', ['US'])[0],
        'filing_year': int(patent.get('patent_date', '20000101')[:4]) if patent.get('patent_date') else 2000,
        'patent_class': [],  # PatentsView doesn't provide this directly; can add if needed
    }

    return {'text': text, 'metadata': metadata}