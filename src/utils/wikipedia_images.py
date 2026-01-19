"""
Actor Image Fetcher using Pywikibot
Fetches actor profile images from Wikipedia/Wikimedia Commons
"""

import re
from typing import Optional, Dict

try:
    import pywikibot
    from pywikibot import pagegenerators
    PYWIKIBOT_AVAILABLE = True
except ImportError:
    PYWIKIBOT_AVAILABLE = False
    print("Warning: pywikibot not installed. Run: pip install pywikibot")


class WikipediaActorImageFetcher:
    """Fetch actor images from Wikipedia using Pywikibot."""
    
    def __init__(self, language='en'):
        """
        Initialize Wikipedia image fetcher.
        
        Args:
            language: Wikipedia language code (default: 'en' for English)
        """
        if not PYWIKIBOT_AVAILABLE:
            raise ImportError("pywikibot is not installed. Install with: pip install pywikibot")
        
        self.site = pywikibot.Site(language, 'wikipedia')
        self.commons = pywikibot.Site('commons', 'commons')
    
    def get_actor_page(self, actor_name: str) -> Optional[pywikibot.Page]:
        """
        Get Wikipedia page for an actor.
        
        Args:
            actor_name: Name of the actor
            
        Returns:
            Wikipedia Page object or None if not found
        """
        try:
            # Try direct page lookup
            page = pywikibot.Page(self.site, actor_name)
            if page.exists():
                return page
            
            # Try searching for the actor
            search_results = list(self.site.search(actor_name, total=5))
            for result in search_results:
                page = pywikibot.Page(self.site, result.title)
                if page.exists() and self._is_person_page(page):
                    return page
            
            return None
        except Exception as e:
            print(f"Error finding page for {actor_name}: {e}")
            return None
    
    def _is_person_page(self, page: pywikibot.Page) -> bool:
        """Check if page is about a person."""
        # Check if page has infobox for person/actor
        text = page.text.lower()
        person_indicators = [
            'infobox person',
            'infobox actor',
            'infobox actress',
            '{{birth date',
            'occupation',
            'years active'
        ]
        return any(indicator in text for indicator in person_indicators)
    
    def get_actor_image(self, actor_name: str) -> Optional[str]:
        """
        Get image URL for an actor from Wikipedia.
        
        Args:
            actor_name: Name of the actor
            
        Returns:
            Image URL or None if not found
        """
        page = self.get_actor_page(actor_name)
        if not page:
            return None
        
        try:
            # Get main image from page
            images = list(page.imagelinks())
            
            if not images:
                return None
            
            # Get first image (usually the main profile picture)
            main_image = images[0]
            
            # Get the image page from Commons
            if main_image.site != self.commons:
                # Try to find image on Commons
                commons_image = pywikibot.FilePage(self.commons, main_image.title())
            else:
                commons_image = main_image
            
            # Get image URL
            if hasattr(commons_image, 'get_file_url'):
                return commons_image.get_file_url()
            elif hasattr(commons_image, 'fileUrl'):
                return commons_image.fileUrl()
            
            return None
            
        except Exception as e:
            print(f"Error fetching image for {actor_name}: {e}")
            return None
    
    def get_actor_info(self, actor_name: str) -> Dict[str, str]:
        """
        Get comprehensive actor information from Wikipedia.
        
        Args:
            actor_name: Name of the actor
            
        Returns:
            Dictionary with actor info including image URL
        """
        page = self.get_actor_page(actor_name)
        
        if not page:
            return {
                'name': actor_name,
                'image_url': None,
                'page_url': None,
                'summary': None
            }
        
        try:
            # Get image
            image_url = self.get_actor_image(actor_name)
            
            # Get page URL
            page_url = page.full_url()
            
            # Get summary (first paragraph)
            summary = page.extract()[:500] if hasattr(page, 'extract') else None
            
            return {
                'name': actor_name,
                'image_url': image_url,
                'page_url': page_url,
                'summary': summary,
                'page_title': page.title()
            }
            
        except Exception as e:
            print(f"Error getting info for {actor_name}: {e}")
            return {
                'name': actor_name,
                'image_url': None,
                'page_url': page.full_url() if page else None,
                'summary': None
            }


def get_actor_photo_url(actor_name: str, language='en') -> Optional[str]:
    """
    Simple function to get actor photo URL from Wikipedia.
    
    Args:
        actor_name: Name of the actor
        language: Wikipedia language (default: 'en')
        
    Returns:
        Image URL or None
    """
    if not PYWIKIBOT_AVAILABLE:
        print("pywikibot not available. Install with: pip install pywikibot")
        return None
    
    try:
        fetcher = WikipediaActorImageFetcher(language)
        return fetcher.get_actor_image(actor_name)
    except Exception as e:
        print(f"Error: {e}")
        return None


# Alternative: Simple scraping approach without Pywikibot
def get_actor_image_simple(actor_name: str) -> Optional[str]:
    """
    Simpler approach using Wikipedia API directly (no pywikibot needed).
    
    Args:
        actor_name: Name of the actor
        
    Returns:
        Image URL or None
    """
    import requests
    
    try:
        # Wikipedia API endpoint
        url = "https://en.wikipedia.org/w/api.php"
        
        # Search for the page
        search_params = {
            'action': 'query',
            'list': 'search',
            'srsearch': actor_name,
            'format': 'json'
        }
        
        response = requests.get(url, params=search_params, timeout=10)
        search_results = response.json()
        
        if not search_results.get('query', {}).get('search'):
            return None
        
        # Get first result's page title
        page_title = search_results['query']['search'][0]['title']
        
        # Get page image
        image_params = {
            'action': 'query',
            'titles': page_title,
            'prop': 'pageimages',
            'format': 'json',
            'pithumbsize': 500  # Image width in pixels
        }
        
        response = requests.get(url, params=image_params, timeout=10)
        data = response.json()
        
        pages = data.get('query', {}).get('pages', {})
        for page_id, page_data in pages.items():
            if 'thumbnail' in page_data:
                return page_data['thumbnail']['source']
        
        return None
        
    except Exception as e:
        print(f"Error fetching image for {actor_name}: {e}")
        return None


if __name__ == "__main__":
    # Demo usage
    print("=" * 70)
    print("WIKIPEDIA ACTOR IMAGE FETCHER - DEMO")
    print("=" * 70)
    print()
    
    # Test actors
    test_actors = [
        "Leonardo DiCaprio",
        "Tom Hanks",
        "Meryl Streep"
    ]
    
    print("Method 1: Simple API approach (no pywikibot needed)")
    print("-" * 70)
    for actor in test_actors:
        print(f"\nFetching image for: {actor}")
        image_url = get_actor_image_simple(actor)
        if image_url:
            print(f"✅ Image URL: {image_url}")
        else:
            print(f"❌ No image found")
    
    print("\n" + "=" * 70)
    print("\nMethod 2: Pywikibot approach (requires pywikibot installation)")
    print("-" * 70)
    
    if PYWIKIBOT_AVAILABLE:
        for actor in test_actors:
            print(f"\nFetching info for: {actor}")
            info = get_actor_info_simple(actor)
            print(f"Name: {info.get('name')}")
            print(f"Image: {info.get('image_url', 'Not found')}")
    else:
        print("Pywikibot not installed. Run: pip install pywikibot")
    
    print("\n" + "=" * 70)
    print("\nRECOMMENDATION:")
    print("Use the SIMPLE API method (get_actor_image_simple)")
    print("- No additional dependencies")
    print("- Faster and more reliable")
    print("- Uses Wikipedia's public API")
    print("=" * 70)


def get_actor_info_simple(actor_name: str) -> Dict[str, str]:
    """
    Get actor info using simple Wikipedia API (no pywikibot).
    
    Args:
        actor_name: Name of the actor
        
    Returns:
        Dictionary with actor info
    """
    import requests
    
    try:
        url = "https://en.wikipedia.org/w/api.php"
        
        # Search for the page
        search_params = {
            'action': 'query',
            'list': 'search',
            'srsearch': actor_name,
            'format': 'json'
        }
        
        response = requests.get(url, params=search_params, timeout=10)
        search_results = response.json()
        
        if not search_results.get('query', {}).get('search'):
            return {'name': actor_name, 'image_url': None}
        
        page_title = search_results['query']['search'][0]['title']
        
        # Get comprehensive page info
        info_params = {
            'action': 'query',
            'titles': page_title,
            'prop': 'pageimages|extracts|info',
            'exintro': True,
            'explaintext': True,
            'pithumbsize': 500,
            'inprop': 'url',
            'format': 'json'
        }
        
        response = requests.get(url, params=info_params, timeout=10)
        data = response.json()
        
        pages = data.get('query', {}).get('pages', {})
        for page_id, page_data in pages.items():
            return {
                'name': actor_name,
                'page_title': page_data.get('title'),
                'image_url': page_data.get('thumbnail', {}).get('source'),
                'page_url': page_data.get('fullurl'),
                'summary': page_data.get('extract', '')[:500]
            }
        
        return {'name': actor_name, 'image_url': None}
        
    except Exception as e:
        print(f"Error: {e}")
        return {'name': actor_name, 'image_url': None}
