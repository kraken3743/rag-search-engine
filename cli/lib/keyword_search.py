from lib.search_utils import load_movies

# def clean_text(text):
#       return text.lower()

def search_command(query, n_results):
        movies = load_movies()
        res = []
        for movie in movies:
                if query in movie['title'].lower():
                    res.append(movie)
                if len(res) == n_results:
                    break
        return res 
        pass