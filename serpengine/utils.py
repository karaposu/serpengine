def parse_tld(link: str) -> str:
    """
    Placeholder for a function that extracts the top-level domain.
    e.g. return 'com' if link is 'https://example.com/somepage'.
    """
    if '.' in link:
        return link.split('.')[-1].split('/')[0]
    return ''