import cv2
import sys
import os
import threading
import numpy as np
import statistics
import plotly.express as px
import dash_bootstrap_components as dbc # -- library dash css bootstrap -- #
import plotly.graph_objects as go
#from skimage import data

#from queue import Queue
from PIL import Image as theImage
import base64
from io import BytesIO

from plotly.graph_objs import *
from dash import Dash, html, dcc, Input, Output, State, callback_context # -- library dash -- #
from flask import Flask, jsonify, request, Response # -- library flask -- #
from dash.exceptions import PreventUpdate # -- library prevent update -- #

#colors = {
#	'background': '#18191a',
#	'text': '#ccc'
#}

external_stylesheets = ['https://unpkg.com/boxicons@2.1.1/css/boxicons.min.css', dbc.themes.CYBORG]
server = Flask(__name__)
server_app = Dash(__name__, server=server, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server_app.title = 'FISH COUNTER'
server_app._favicon = ("imbig_icon01.ico")

#frame = []

def gstreamer_pipeline(
	sensor_id=0,
	capture_width=1920,
	capture_height=1080,
	display_width=800,
	display_height=378,
	framerate=30,
	flip_method=0,
):
	return (
		"nvarguscamerasrc sensor-id=%d !"
		"video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
		"nvvidconv flip-method=%d ! "
		"video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
		"videoconvert ! "
		"video/x-raw, format=(string)BGR ! appsink"
		% (
			sensor_id,
			capture_width,
			capture_height,
			framerate,
			flip_method,
			display_width,
			display_height,
		)
	)

class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

	def __del__(self):
		self.video.release()

	def get_frame(self):
		global image
		success, image = self.video.read()
		ret, jpeg = cv2.imencode('.jpg', image)
		return jpeg.tobytes()


def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
			   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@server.route('/video_feed')
def video_feed():
	return Response(gen(VideoCamera()),
					mimetype='multipart/x-mixed-replace; boundary=frame')

def Average(lst):
    return sum(lst) / len(lst)

app_title = html.Div(children=[
	html.H1('FISH COUNTER', className='app-title')

])

app_videoStreaming = html.Div(children=[
	html.Div(children=[
		html.H5('Live Stream', className='app-image-title'),
		html.Img(src='/video_feed', className='app-image')
	], className='app-container-image'),
	html.Div(children=[
		html.H5('Machine Learning', className='app-image-title'),
		html.Img(src='assets/TILAPIA.jpg', className='app-image', id='ml_image')
	], className='app-container-image'),
], className='app-container01')

app_countBtn = html.Div(children=[
	dbc.Button('Count', size='lg', class_name='app-countBtn', id='btn_count', n_clicks=0)	
], className='app-container02')

app_table = html.Div(children=[
	html.Div(children=[
		html.Span('Frequency', className='app-table-title'),
		html.Span('0', className='app-table-list', id='max_result')
	], className='app-container-table'),
	html.Div(children=[
		html.Span('Median', className='app-table-title'),
		html.Span('0', className='app-table-list', id='min_result')
	], className='app-container-table'),
	html.Div(children=[
		html.Span('Average', className='app-table-title'),
		html.Span('0', className='app-table-list', id='average_result')
	], className='app-container-table')
], className='app-container03')

server_app.layout = html.Div(children=[
	app_title,
	app_videoStreaming,
	app_countBtn,
	dcc.Interval(id='clock', interval=500, n_intervals=0, max_intervals=-1, disabled=True),
	html.Div(children=dbc.Progress(id='progress_bar', value=0, animated=True, striped=True, class_name='bar'), className='bar_container'),
	app_table,
	html.Div(children=dbc.Button(id='btn_shutdown', children='SHUTDOWN', class_name='btn_shutdown', n_clicks=0), className='btn_shutdown_container'),
        html.Div(id='shutdown', style={'display': 'none'})
])

@server_app.callback(
	Output('ml_image', 'src'),
	Output('max_result', 'children'),
	Output('min_result', 'children'),
	Output('average_result', 'children'),
	Input('btn_count', 'n_clicks')
)
def cb_count(btn_count):
	trigger = callback_context.triggered[0]
	ext = ['png', 'jpg', 'gif']

	event = threading.Event()

	if trigger['prop_id'].split('.')[0] == 'btn_count':
		list_fish_number = []
		#last_image = []
		#print(result)
		for i in range(0, 20):
			#yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
			#ret, jpeg = cv2.imencode('.jpg', frame)
			#print(image)
			#print(success)
			#img = cv2.imread('images/test02/image_' + str(i+100) + '.jpg')
			img = image
			dim_img = img.shape
			g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			g = cv2.bitwise_not(g)
			height, width= g.shape[:2]
			ROI= np.array([[(150,344),(150,32),(787,16),(787,347)]], dtype= np.int32)
			blank= np.zeros_like(g)
			roi_g= cv2.fillPoly(blank, ROI,255)
			roi_img= cv2.bitwise_and(g, roi_g)

			A = (np.mean(roi_img))/(np.median(roi_img))
			B = np.sqrt(np.mean(roi_img))/(np.median(roi_img))
			y = (-np.mean(roi_img) + np.median(roi_img))/np.median(roi_img)
			Z = A +4*B
			print(np.mean(roi_img))
			print(np.std(roi_img))
			X = np.mean(roi_img)+(1)*np.std(roi_img)+(np.mean(roi_img)/   np.std(roi_img))
			print(X)
			(thresh, bw) = cv2.threshold(roi_img, X-5, 255, cv2.THRESH_BINARY)
                        #141.9 cutoff
			strel = np.ones((4,4),np.uint8)
			ellip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
			result = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ellip)

			#strel = np.ones((4,4),np.uint8)
			#ellip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
			#result = cv2.morphologyEx(result1, cv2.MORPH_OPEN, ellip)

			mask = np.full(result.shape,255)
			bwboundary= mask - result
			bwboundary = bwboundary.astype(np.uint8)
			dim_bwboundary = bwboundary.shape
			canny = cv2.Canny(bwboundary, 30, 150, 3)
			dilated = cv2.dilate(canny, (1,1), iterations = 3)
			(cnt, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
			rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			cv2.drawContours(rgb, cnt, -1, (0,255,0),2)
			fish_number = len(cnt)
			list_fish_number = np.append(list_fish_number, fish_number)
			#print(len(list_fish_number))
			#print(fish_number)
			#list_fish_number[i] = roi_img
			
			#result[i] = result

			if i == 4:
				final_image = rgb

			event.wait(0.5)

			#global value_bar

			#value_bar = ((i+1) / 20) * 100

			#return [None, value_bar, f'{value_bar} %', None, None, None]

		#print(list_fish_number)
		#max_number = max(list_fish_number)
		
		max_number = (3*np.median(list_fish_number))-(2*np.mean(list_fish_number))
		min_number = np.median(list_fish_number)

		#ave_number = Average(list_fish_number)
		ave_number = np.median(list_fish_number)
#(np.mean(list_fish_number)+min_number+max_number)/3
		
		pil_img = theImage.fromarray(final_image) # PIL image object
		prefix = "data:image/png;base64,"
		with BytesIO() as stream:
			pil_img.save(stream, format="png")

		return [pil_img, int(max_number), int(min_number), int(ave_number)]

	else:
		raise PreventUpdate

@server_app.callback(
	Output('progress_bar', 'value'),
	Output('clock', 'disabled'),
	Output('btn_count', 'n_cicks'),
	Output('clock', 'n_intervals'),
	Input('btn_count', 'n_clicks'),
	Input('clock', 'n_intervals')
)
def update_progress(btn_count, n):
	trigger = callback_context.triggered[0]
	progress = min(((n/20) * 100) % 110, 100)

	print(progress)

	if btn_count > 0:
		if progress < 100:
			return [progress, False, 1, n]

		elif progress == 100:
			return [0, True, 0, 0]

		else:
			raise PreventUpdate
	else:
		raise PreventUpdate

@server_app.callback(
	Output('shutdown', 'children'),
	Input('btn_shutdown', 'n_clicks')
)
def shutdown(btn_shutdown_clk):
	trigger = callback_context.triggered[0]

	if trigger['prop_id'].split('.')[0] == 'btn_shutdown':
		os.system("shutdown now -h")
		return ""

	else:
		raise PreventUpdate

if __name__ == '__main__':
	server.run(debug=False, port=8080, threaded=True)
