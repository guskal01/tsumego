import os
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation

nof_games = []
time = []
winrate = []
x = 'nof_games' # 'nof_games' or 'time'

other_runs = []

def update():
	assert(x in ['nof_games', 'time'])
	plt.clf()
	for run in other_runs:
		plt.plot(run[1] if x=='time' else run[0], run[2])
	plt.plot(time if x=='time' else nof_games, winrate)
	plt.xlabel('Time (s)' if x=='time' else 'Training games')
	plt.ylabel('Winrate')
	plt.axis((0, None, 0, 1))
	plt.pause(0.001)

def add_point(id, _nof_games, _time, _winrate, save):
	if(save):
		os.makedirs('runs', exist_ok=True)
		file = open(f'runs/{id}.txt', 'a+')
		file.write(f'{_nof_games},{_time},{_winrate}\n')
		file.close()

	nof_games.append(_nof_games)
	time.append(_time)
	winrate.append(_winrate)
	update()

def add_run(id):
	file = open(f'runs/{id}.txt', 'r')
	s = file.read().strip().split('\n')
	file.close()
	other_runs.append(list(zip(*[map(float, x.split(',')) for x in s])))
	update()

matplotlib.rcParams['toolbar'] = 'None'
plt.ion()
update()
plt.rcParams['figure.raise_window'] = False
