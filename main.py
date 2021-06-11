import random
from board import Board
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
	b = Board(height, width)
	while 1:
		#time.sleep(0.5)
		m = b.get_legal_moves()[1:]
		if(len(m) == 0):
			print("Restarting")
			b = Board(height, width)
			m = b.get_legal_moves()[1:]
		#print("Legal moves:", m)
		r,c = random.sample(m, 1)[0]
		#print("Move:", r, c)
		b.play(r, c)
		if(use_gui): gui.update(b)
		#print(b)

if(use_gui):
	from gui import TsumegoApp
	gui = TsumegoApp(height, width)
else:
	gui = None
t = threading.Thread(target=main)
t.start()
if(gui): gui.run()

