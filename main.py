import random
from tsumegoboard import Board
import threading
import time
import dqn

use_gui = False
height = 5
width = 7

def main():
	while(gui and not gui.ready):
		pass
	dqn.train(height, width, gui)

	exit()

	# Testing:
	start = time.time()
	s = 0
	for i in range(20000):
		b = Board(height, width)
		b.set_stone(1, 1, 2)
		b.set_stone(2, 2, 2)
		b.set_stone(1, 3, 2)
		b.reset_superko()
		random.seed(i)
		while(1):
			if(b.black_won()):
				break
			if(b.game_over()):
				break
			m = b.get_legal_moves()
			s += len(m)
			r,c = random.sample(m, 1)[0]
			b.play(r, c)
	print(s) # 13495803
	print(time.time()-start)

if(use_gui):
	from gui import TsumegoApp
	gui = TsumegoApp(height, width)
else:
	gui = None
t = threading.Thread(target=main)
t.start()
if(gui): gui.run()

