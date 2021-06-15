import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from tsumegoboard import Board
from collections import deque

SIMULS = 20 # number of games played at the same time, in one round
BATCH_SIZE = 32
assert torch.cuda.is_available()
device = torch.device('cuda')
#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.fastest = True

class Network(nn.Module):
	def __init__(self, height, width, layers):
		super().__init__()

		self.channels = 32

		self.conv = []
		self.conv.append(nn.Conv2d(5, self.channels, 3, padding=1))
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

	experience = deque(maxlen=10000)
	gamma = 1
	rounds = 0
	while(1):
		boards = []
		for i in range(SIMULS):
			boards.append(build_board(height, width))
		games = [[] for _ in range(SIMULS)]
		ended = 0
		while(ended < SIMULS):
			x = []
			for i in range(SIMULS):
				x.append(boards[i].features())
			x = np.stack(x)
			y = net(torch.from_numpy(x).to(device)).detach().cpu().numpy()
			for i in range(SIMULS):
				if(boards[i].black_won() or boards[i].game_over()):
					continue
				noise = np.random.normal(0, .1, y[i].shape)
				y[i] += noise
				if(random.random() < gamma): y[i] = noise
				movenum, move = boards[i].best_move(y[i])
				games[i].append([x[i], movenum])
				boards[i].play(*move)

				winner = None
				if(boards[i].black_won()): winner = 0
				elif(boards[i].game_over()): winner = 1
				if(winner != None):
					for j in range(len(games[i])):
						games[i][j].append(winner)
					print("BW"[winner], end="", flush=True)
					ended += 1
		print()
		train_data = []
		for i in range(SIMULS):
			train_data += games[i]

		experience.extend(train_data)
		train_data += random.sample(experience, min(len(experience), len(train_data)))
		random.shuffle(train_data)
		train_x = []
		train_move = []
		train_result = []
		for i in range(len(train_data)):
			train_x.append(train_data[i][0])
			train_move.append(train_data[i][1])
			train_result.append(train_data[i][2])
		train_x = torch.tensor(np.stack(train_x, axis=0)).to(device)
		train_move = torch.tensor(np.stack(train_move, axis=0)).to(device)
		train_result = torch.tensor(np.stack(train_result, axis=0), dtype=torch.float32).to(device)
		for i in range(0, len(train_data), BATCH_SIZE):
			x = train_x[i:i+BATCH_SIZE]
			move = train_move[i:i+BATCH_SIZE]
			result = train_result[i:i+BATCH_SIZE]
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
				x = torch.from_numpy(b.features()).unsqueeze(0)
				y = net(x.to(device)).detach().cpu().numpy()
				movenum, move = b.best_move(y[0])
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
		x = torch.from_numpy(b.features()).unsqueeze(0)
		y = net(x.to(device)).detach().cpu().numpy()
		movenum, move = b.best_move(y[0])
		
		b.play(*move)
		if(gui): gui.update(b)

		print("WR:", y[0][movenum])
		print(*move)
		print(b)
		print()

		if(b.black_won()):
			print("BLACK WON")
			break
		if(b.game_over()):
			print("WHITE WON")
			break

		time.sleep(0.7)
