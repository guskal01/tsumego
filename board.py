import random
import numpy as np

class Board:
	def __init__(self, height, width):
		self.height = height
		self.width = width
		self.a = [[0]*width for _ in range(height)]
		self.turn = 1
		self.passes = 0
		self.moves = 0
		self.dir = [(-1, 0), (0, 1), (1, 0), (0, -1)]
		self.zobrist_table = np.random.randint(-(1<<63), 1<<63, (height, width, 2), dtype=np.int64)
		self.zobrist_turn = random.randint(-(1<<63), (1<<63)-1)
		self.zobrist_hash = 0
		self.zobrist_seen = set()
		self.reset_superko()

	def has_liberty(self, r, c, col=None, visited=None):
		if(col == None):
			col = self.a[r][c]
		if(r < 0 or c < 0):
			return False
		if(r >= self.height or c >= self.width):
			return col == 1
		if(visited == None): visited = set()
		if((r, c) in visited): return False
		visited.add((r, c))
		if(self.a[r][c] == 0): return True
		if(self.a[r][c] != col): return False
		for dr, dc in self.dir:
			if(self.has_liberty(r+dr, c+dc, col, visited)): return True
		return False

	def is_inside(self, r, c):
		return r >= 0 and r < self.height and c >= 0 and c < self.width

	def set_stone(self, r, c, color):
		if(self.a[r][c] != 0): self.zobrist_hash ^= self.zobrist_table[r][c][self.a[r][c]-1]
		self.a[r][c] = color
		if(self.a[r][c] != 0): self.zobrist_hash ^= self.zobrist_table[r][c][self.a[r][c]-1]

	def get_stone(self, r, c):
		return self.a[r][c]

	def switch_turn(self):
		self.turn ^= 3
		self.zobrist_hash ^= self.zobrist_turn

	def reset_superko(self):
		self.zobrist_seen.clear()
		self.zobrist_seen.add(self.zobrist_hash)

	def is_legal(self, r, c):
		if(r == -1 and c == -1): return True #pass
		if(not self.is_inside(r, c)): return False
		if(self.a[r][c] != 0): return False
		self.set_stone(r, c, self.turn)
		if(self.has_liberty(r, c)):
			if(self.zobrist_hash^self.zobrist_turn in self.zobrist_seen):
				self.set_stone(r, c, 0)
				return False # uncommon but possible
			self.set_stone(r, c, 0)
			return True
		
		visited = set()
		h = 0
		for dr, dc in self.dir:
			if(self.is_inside(r+dr, c+dc) and self.a[r+dr][c+dc] != self.turn and not self.has_liberty(r+dr, c+dc)):
				h ^= self.hypothetical_floodfill(r+dr, c+dc, visited, 0)
		if(h == 0):
			self.set_stone(r, c, 0)
			return False # nothing captured, suicide
		if(self.zobrist_hash^self.zobrist_turn^h in self.zobrist_seen):
			self.set_stone(r, c, 0)
			return False
		self.set_stone(r, c, 0)
		return True

	def get_legal_moves(self):
		ret = [(-1, -1)]
		for r in range(self.height):
			for c in range(self.width):
				if(self.is_legal(r, c)):
					ret.append((r, c))
		return ret

	def floodfill(self, r, c, new_col, old_col=None):
		if(old_col == None): old_col = self.a[r][c]
		if(not self.is_inside(r, c)): return
		if(self.a[r][c] != old_col): return
		self.set_stone(r, c, new_col)
		for dr, dc in self.dir:
			self.floodfill(r+dr, c+dc, new_col, old_col)

	# Calculate zobrist difference if floodfilled
	def hypothetical_floodfill(self, r, c, visited, new_col, old_col=None):
		if(old_col == None): old_col = self.a[r][c]
		if(not self.is_inside(r, c)): return 0
		if(self.a[r][c] != old_col): return 0
		if((r,c) in visited): return 0
		visited.add((r,c))
		assert old_col != 0 and new_col == 0
		ret = self.zobrist_table[r][c][old_col-1]
		for dr, dc in self.dir:
			ret ^= self.hypothetical_floodfill(r+dr, c+dc, visited, new_col, old_col)
		return ret

	def play(self, r, c):
		assert self.is_legal(r, c)
		if(r == -1 and c == -1): #pass
			self.passes += 1
			self.switch_turn()
			self.reset_superko()
			return
		self.passes = 0
		self.set_stone(r, c, self.turn)
		for dr, dc in self.dir:
			if(self.is_inside(r+dr, c+dc) and self.a[r+dr][c+dc] == self.turn^3 and not self.has_liberty(r+dr, c+dc)):
				self.floodfill(r+dr, c+dc, 0)

		self.switch_turn()
		self.zobrist_seen.add(self.zobrist_hash)
		self.moves += 1

	def black_won(self):
		for r in range(self.height):
			for c in range(self.width):
				if(self.a[r][c] == 2): return False
		return True

	def game_over(self):
		return self.passes >= 3 or self.moves >= self.height*self.width*5

	def features(self):
		ret = np.zeros((5, self.height, self.width), dtype=np.float32)
		for r in range(self.height):
			for c in range(self.width):
				if(self.get_stone(r, c) == 0):
					ret[0][r][c] = 1
				elif(self.get_stone(r, c) == 1):
					ret[1][r][c] = 1
				elif(self.get_stone(r, c) == 2):
					ret[2][r][c] = 1
				else: assert False
				if(self.turn == 1):
					ret[3][r][c] = 1
				elif(self.turn == 2):
					ret[4][r][c] = 1
				else: assert False
		return ret

	def best_move(self, y):
		sign = -1
		if(self.turn == 2): sign = 1

		t = (self.turn-1)*(self.height*self.width+1)

		movenum = self.height*self.width + t
		move = (-1, -1) #pass
		best = sign*y[self.height*self.width + t] #pass
		for r in range(self.height):
			for c in range(self.width):
				if(sign*y[r*self.width+c + t] > best and self.is_legal(r, c)):
					best = sign*y[r*self.width+c + t]
					move = (r, c)
					movenum = r*self.width+c + t
		return movenum, move

	def __str__(self):
		L = []
		for r in range(self.height):
			for c in range(self.width):
				L.append('.BW'[self.a[r][c]] + ' ')
			L.append('B\n')
		for c in range(self.width):
			L.append('B ')
		L.append('\n')
		L.append('?BW'[self.turn] + " to play")
		return ''.join(L)
