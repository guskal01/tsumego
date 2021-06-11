from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.graphics import *
from kivy.clock import Clock

from board import Board

class GoPos(Widget):
	def __init__(self, r, c, **kwargs):
		super(GoPos, self).__init__(**kwargs)
		self.stone = 0
		self.row = r
		self.col = c
		self.bind(size=self.update_canvas)

	def update_canvas(self, *args):
		self.canvas.clear()
		with self.canvas:
			Color(0.85, 0.68, 0.49, 1)
			Rectangle(pos=self.pos, size=self.size)
			Color(0, 0, 0, 1)

			Line(width=2, points=[self.pos[0] + self.width/2, self.pos[1] + 0, self.pos[0] + self.width/2, self.pos[1] + self.height/2])
			Line(width=2, points=[self.pos[0] + self.width/2, self.pos[1] + self.height/2, self.pos[0] + self.width, self.pos[1] + self.height/2])
			if(self.row != 0):
				Line(width=2, points=[self.pos[0] + self.width/2, self.pos[1] + self.height/2, self.pos[0] + self.width/2, self.pos[1] + self.height])
			if(self.col != 0):
				Line(width=2, points=[self.pos[0] + 0, self.pos[1] + self.height/2, self.pos[0] + self.width/2, self.pos[1] + self.height/2])

			if(self.stone == 1):
				Color(0, 0, 0, 1)
			elif(self.stone == 2):
				Color(1, 1, 1, 1)
			if(self.stone):
				size = 0.9
				Ellipse(pos=[self.pos[0]+self.size[0]*(1-size)/2, self.pos[1]+self.size[1]*(1-size)/2], size=[self.size[0]*size, self.size[1]*size])

class GoBoard(GridLayout):
	def __init__(self, height, width, **kwargs):
		super(GoBoard, self).__init__(**kwargs)
		self.cols = width
		self.a = [[0]*width for _ in range(height)]
		for r in range(height):
			for c in range(width):
				self.a[r][c] = GoPos(r, c)
				self.add_widget(self.a[r][c])

	def set_stone(self, r, c, color):
		self.a[r][c].stone = color

	def draw(self):
		for r in self.a:
			for c in r:
				c.update_canvas()

class TsumegoApp(App):
	def __init__(self, height, width):
		super(TsumegoApp, self).__init__()
		self.height = height
		self.width = width
		self.ready = False
		Clock.schedule_once(self.is_ready, 0)

	def is_ready(self, c):
		self.ready = True

	def build(self):
		self.gui_board = GoBoard(self.height+1, self.width+1)
		for i in range(self.height):
			self.gui_board.set_stone(i, self.width, 1)
		for i in range(self.width+1):
			self.gui_board.set_stone(self.height, i, 1)
		return self.gui_board

	def update(self, board):
		assert board.height == self.height and board.width == self.width
		for r in range(self.height):
			for c in range(self.width):
				self.gui_board.set_stone(r, c, board.a[r][c])
		self.gui_board.draw()
