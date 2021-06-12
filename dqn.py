import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from tsumegoboard import Board
from collections import deque

IN_CHANNELS = 5
assert torch.cuda.is_available()
device = torch.device('cpu') # 'cuda'
#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.fastest = True

class Network(nn.Module):
	def __init__(self, height, width, layers):
		super().__init__()

		self.channels = 32

		self.conv = []
		self.conv.append(nn.Conv2d(IN_CHANNELS, self.channels, 3, padding=1))
		for i in range(layers-2):
			self.conv.append(nn.Conv2d(self.channels, self.channels, 3, padding=1))
		self.conv.append(nn.Conv2d(self.channels, 3, 3, padding=1))
		self.conv = nn.ModuleList(self.conv)
		self.fc = nn.Linear(height*width*3, 2*(height*width+1))

	def forward(self, x):
		for c in self.conv:
			x = F.relu(c(x))
		x = torch.flatten(x, 1)
		x = self.fc(x)
		return x

def features_from_board(board):
	ret = np.zeros((IN_CHANNELS, board.height, board.width), dtype=np.float32)
	for r in range(board.height):
		for c in range(board.width):
			if(board.get_stone(r, c) == 0):
				ret[0][r][c] = 1
			elif(board.get_stone(r, c) == 1):
				ret[1][r][c] = 1
			elif(board.get_stone(r, c) == 2):
				ret[2][r][c] = 1
			else: assert False
			if(board.turn == 1):
				ret[3][r][c] = 1
			elif(board.turn == 2):
				ret[4][r][c] = 1
			else: assert False
	return ret

def best_legal_move(y, board):
	sign = -1
	if(board.turn == 2): sign = 1

	t = (board.turn-1)*(board.height*board.width+1)

	movenum = board.height*board.width + t
	move = (-1, -1) #pass
	best = sign*y[board.height*board.width + t] #pass
	for r in range(board.height):
		for c in range(board.width):
			if(sign*y[r*board.width+c + t] > best and board.is_legal(r, c)):
				best = sign*y[r*board.width+c + t]
				move = (r, c)
				movenum = r*board.width+c + t
	return movenum, move

def build_board(height, width):
	b = Board(height, width)
	b.set_stone(1, 1, 2)
	b.set_stone(2, 2, 2)
	b.set_stone(1, 3, 2)
	b.reset_superko()
	return b

def train(height, width, gui=None):
	net = Network(height, width, 6).to(device)
	optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)

	replay_buffer = deque(maxlen=10000)
	gamma = 1
	rounds = 0
	while(1):
		buffer = []
		for i in range(20):
			game_buffer = []
			b = build_board(height, width)
			while(1):
				x = features_from_board(b)
				y = net(torch.from_numpy(x).to(device).unsqueeze(0)).detach().cpu().numpy()
				noise = np.random.normal(0, .1, y.shape)
				y += noise
				if(random.random() < gamma): y = noise
				movenum, move = best_legal_move(y[0], b)

				game_buffer.append([x, movenum])
				b.play(*move)
				if(b.black_won()):
					for i in range(len(game_buffer)):
						game_buffer[i].append(0)
					print("B", end="", flush=True)
					break
				if(b.game_over()):
					for i in range(len(game_buffer)):
						game_buffer[i].append(1)
					print("W", end="", flush=True)
					break
			buffer += game_buffer
		print()
		replay_buffer.extend(buffer)
		buffer += random.sample(replay_buffer, min(len(replay_buffer), len(buffer)))
		random.shuffle(buffer)
		buffer_x = []
		buffer_move = []
		buffer_result = []
		for i in range(len(buffer)):
			buffer_x.append(buffer[i][0])
			buffer_move.append(buffer[i][1])
			buffer_result.append(buffer[i][2])
		buffer_x = torch.tensor(np.stack(buffer_x, axis=0)).to(device)
		buffer_move = torch.tensor(np.stack(buffer_move, axis=0)).to(device)
		buffer_result = torch.tensor(np.stack(buffer_result, axis=0), dtype=torch.float32).to(device)
		for i in range(0, len(buffer), 32):
			x = buffer_x[i:i+32]
			move = buffer_move[i:i+32]
			result = buffer_result[i:i+32]
			y = net(x)
			y = y.gather(dim=1, index=move.unsqueeze(1)).flatten()
			loss = F.mse_loss(y, result)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		if(rounds % 50 == 0 and rounds != 0):
			#exit()
			test(height, width, net)
			if(rounds%100 == 0): playgame(height, width, net, gui)
		rounds += 1
		gamma *= 0.98
		print("GAMMA:", gamma)

def test(height, width, net):
	print("TESTING...")
	gamecnt = 100
	wins = 0
	for game in range(gamecnt):
		b = build_board(height, width)
		moves = 0
		while(1):
			if(moves%2 == 0):
				move = b.get_legal_moves()
				if(len(move) == 1): move = move[0]
				else: move = random.sample(move[1:], 1)[0]
			else:
				x = torch.from_numpy(features_from_board(b)).unsqueeze(0)
				y = net(x.to(device)).detach().cpu().numpy()
				movenum, move = best_legal_move(y[0], b)
			b.play(*move)
			moves += 1
			if(b.black_won()):
				break
			if(b.game_over()):
				wins+=1
				break
	print("Wins:", wins)
	time.sleep(3)

def playgame(height, width, net, gui=None):
	b = build_board(height, width)
	while(1):
		x = torch.from_numpy(features_from_board(b)).unsqueeze(0)
		y = net(x.to(device)).detach().cpu().numpy()
		movenum, move = best_legal_move(y[0], b)
		
		b.play(*move)
		if(b.black_won()):
			print("BLACK WON")
			break
		if(b.game_over()):
			print("WHITE WON")
			break

		print("WR:", y[0][movenum])
		print(*move)
		print(b)
		print()
		if(gui): gui.update(b)

		time.sleep(0.7)
