# README

This repo uses the `mtg-sdk-python` to compare all cards in existence and form a large n x n 
similarity matrix where the ijth component compares card i with j. 

The end result is a dictionary where the key is the `multiverse_id` and the values are a list of top 50 associated cards and their similarity scores.

We add a migration script as well where the output is inserted into a Redis server as a sorted set. Moreover,
we also migrate a hash for card image urls and names associated with the `multiverse_id` along with a complete list of card names.

Example:

```
python migration.py all localhost 6379
```

One might want to use this in Django after performing the migration. Simply hook up your `views.py` file to Redis.

Example:

```
import redis

POOL = redis.ConnectionPool(host='127.0.0.1', port=6379, db=0) 

def get_names(request):
	r = redis.Redis(connection_pool=POOL)
	return render(request, 'cardnamess.html', {'cardnames':[r.smembers('cardnames')]})
```