"""
Quick script to check what movies are available on TMDB pages
"""
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

def check_page(page, vote_count=5, vote_average=3.0):
    params = {
        'api_key': API_KEY,
        'page': page,
        'sort_by': 'popularity.desc',
        'vote_count.gte': vote_count,
        'vote_average.gte': vote_average,
        'include_adult': 'false',
    }
    
    response = requests.get(f"{BASE_URL}/discover/movie", params=params)
    
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])
        total_pages = data.get('total_pages', 0)
        total_results = data.get('total_results', 0)
        
        print(f"\n📄 Page {page} of {total_pages}")
        print(f"Total movies matching filters: {total_results}")
        print(f"Movies on this page: {len(results)}\n")
        
        if results:
            print("Sample movies:")
            for i, movie in enumerate(results[:5], 1):
                print(f"{i}. {movie['title']} ({movie.get('release_date', 'N/A')[:4]})")
                print(f"   Rating: {movie['vote_average']}/10, Votes: {movie['vote_count']}")
        else:
            print("❌ No movies found on this page")
            
        return total_pages, total_results
    else:
        print(f"❌ Error: {response.status_code}")
        return 0, 0

if __name__ == "__main__":
    print("🔍 Checking TMDB Movie Availability")
    print("=" * 50)
    
    # Check with current filters
    print("\n📊 Current filters: vote_count >= 5, vote_average >= 3.0")
    total_pages, total_results = check_page(501, vote_count=5, vote_average=3.0)
    
    if total_pages == 0 or total_results == 0:
        print("\n💡 Trying looser filters...")
        print("\n📊 Looser filters: vote_count >= 1, vote_average >= 3.0")
        check_page(501, vote_count=1, vote_average=3.0)
        
        print("\n📊 Even looser: vote_count >= 1, vote_average >= 0")
        check_page(501, vote_count=1, vote_average=0)
