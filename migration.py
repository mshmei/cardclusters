"""
The MIT License (MIT)

Copyright (c) 2017 Michael Songhao Mei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import redis
from cardclusters import *

class redisMigrate(cardCluster):
	@timeit
	def __init__(self, set, host, port):
		super(redisMigrate, self).__init__(set)
		self.host = host
		self.port = port

	@timeit
	def migrate(self):
		print('Establishing connection with Redis server...')
		r = redis.StrictRedis(host = self.host, port = self.port, db = 0)
		print('Generating hashes to store...')
		sim_dict = self.generate_hashes()
		# Stores the multiverse_id key and associated card values with scores in a sorted set
		print('Storing into a sorted set...')
		for key, values in sim_dict.items():
			for i in range(len(values[0])):
				r.zadd(':'.join(['multiverse_id', str(key)]), values[1][i], values[0][i])
		# Stores the multiverse_id key and associated name and image url
		for key, values in self.cards.items():
			r.hmset(str(key), {props:value for props, value in values.items() if props in ('name', 'imageurl')})
			r.sadd('cardnames', values['name'])
		return ('Migration completed.')

if __name__ == '__main__':
	rm = redisMigrate(set = sys.argv[1], 
					  host = sys.argv[2], 
					  port = sys.argv[3])
	rm.migrate()
	
