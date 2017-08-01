# README

This repo uses the `mtg-sdk-python` to compare all cards in existence and form a large n x n 
similarity matrix where the ijth component compares card i with j. The end result is a dictionary
where the key is the `multiverse_id` and the values are a list of top 50 associated cards and their
similarity scores.

We add a migration script as well where the output is inserted into a Redis server as a sorted set.

Example:

```
python migration.py all localhost 6379
```