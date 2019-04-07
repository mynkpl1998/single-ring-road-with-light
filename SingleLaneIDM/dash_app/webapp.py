import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objs as go
import pickle
import copy

def readPickle(fname):
	with open(fname, "rb") as handle:
		data = pickle.load(handle)

	return copy.deepcopy(data)


local10m_loc = "SingleLaneIDM/Common/localview10mtrs/dataset.pkl"
comm10m_reg2_loc = "SingleLaneIDM/Common/comm10m_reg2/dataset.pkl"
comm10m_reg4_loc = "SingleLaneIDM/Common/comm10m_reg4/dataset.pkl"

local10m_img_loc = "SingleLaneIDM/Common/localview10mtrs/Images/"
comm10m_reg2_img_loc = "SingleLaneIDM/Common/comm10m_reg2/Images/"
comm10m_reg4_img_loc = "SingleLaneIDM/Common/comm10m_reg4/Images/"


vtp_data_dict = {}
vtp_data_dict["local10m"] = readPickle(local10m_loc)
vtp_data_dict["comm10m_reg2"] = readPickle(comm10m_reg2_loc)
vtp_data_dict["comm10m_reg4"] = readPickle(comm10m_reg4_loc)

# Create Traffic Density Dropdown list
traffic_density_dropdown_list = []
for density in vtp_data_dict["local10m"]["data"].keys():
	d = {}
	d["label"] = "Num. Vehicles = %d"%(density)
	d["value"] = density
	traffic_density_dropdown_list.append(d)


app = dash.Dash(__name__)

# Bootstrap CSS
app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})

# Define all variables here
webpage_title = "Age Optimal Updates for Autonomous Driving"
repository_name = "single-ring-road-with-light"

# Set background color
colors = {
	'background': '#111211'
}

# Header 
app.layout = html.Div(style={'backgroundColor': colors["background"]}, children=html.Div([


		# ---- Header Starts ---- #
		html.Div([

			html.Div([

					html.H1(style={'fontColor': 'blue', 'fontFamily': 'HelveticaNeue'}, children=webpage_title),
					html.H6(children="Mayank K. Pal, Sanjit K. Kaul, Saket Anand", style={'fontFamily': 'HelveticaNeue'})
				], className="eight columns"),

			html.Div([

				html.Img(src="https://i.imgur.com/vaYjkdF.png",
					style={
						"height": "70%",
						"width": "70%",
						"float": "right",
						"positive":  "relative",
						"margin-top": 8
					})

				], className="four columns")

			], className="row"),
		# ---- Header Ends here ---- #
		html.Br(),
		html.Hr(),

		# ---- Visualizing Policy Starts ---- #	
		html.Div([

			# Title
			html.Div([

				html.H2(html.U("Visualizing Trained Policy"), style={'fontFamily': 'HelveticaNeue', 'text-align': 'center'})

				]),

			html.Br(),

			html.Div([

				dcc.Dropdown(
					id="vtp-case-dropdown",
					options=[
						{'label': "Only Local View (10 metres)", 'value': 'local10m'},
						{'label': "Extended view (10 metres), full access to Comm. ", 'value': 'comm10m_reg2'},
						{'label': "Extended view (10 metres), restricted access to comm.", 'value': 'comm10m_reg4'},
					],
					placeholder = "Select a case")

				], className="five columns", style={"fontFamily": "HelveticaNeue", "fontWeight": "bold", 'float': 'left'}),

			html.Div([

				dcc.Dropdown(
					id="vtp-density-dropdown",
					options=traffic_density_dropdown_list,
					placeholder = "Select traffic density, (num vehicles)",)

				], className="five columns", style={"fontFamily": "HelveticaNeue", "fontWeight": "bold", 'float': 'right'})

			], className = "row"),


		html.Div([

				html.Br(),
				html.Label(id="slider-text", style={"fontFamily": "HelveticaNeue", "fontWeight": "bold", 'fontSize': 20}),
				html.Br(),
				dcc.Slider(id="time-slider", min=0, max=vtp_data_dict["local10m"]["episode-length"]-1, step=1),
				html.Label(children="Jump to step : (max = %d)"%(vtp_data_dict["local10m"]["episode-length"]), style={"fontFamily": "HelveticaNeue", "fontWeight": "bold", 'fontSize': 20}),
				dcc.Input(id="set-time-slider", type="text", max=vtp_data_dict["local10m"]["episode-length"], debounce=True, min=0)

			], className = "row"),

		html.Div([

			html.Div([

				html.H2(children="RL Agent State Input", style={'text-align' : 'center', 'fontFamily': "Old Standard TT, serif", 'fontSize': 20, "margin-left": 140, "margin-top": 30}, className="six columns"),

				html.Img(id="vtp-frames",
					style={
						"margin-top": 20,
						"margin-left": 150,
						"height": "auto",
						"width": "auto",
						"max-width": "1000px",
					})

				], className="six columns"),

			html.Div([

				dcc.Graph(id="vtp-action-distribution")

				], className="six columns", style={"fontFamily": "HelveticaNeue"})

			], className="row"),

				html.Div([


			html.Div([

				dcc.Graph(id="vtp-cum-reward")

				], className="six columns"),

			html.Div([

				dcc.Graph(id="vtp-agent-vel")

				], className="six columns", style={"fontFamily": "HelveticaNeue"})

			], className="row")

		# ---- Visualizing Policy Ends ---- #

		], className = "ten columns offset-by-one")
	)

@app.callback(Output('vtp-agent-vel', 'figure'),
	[Input('vtp-case-dropdown', 'value'), Input('vtp-density-dropdown', 'value'), Input('time-slider', 'value')])
def update_vts_graphs(case, traffic_density, time_slider_value):
	
	if case == None or traffic_density == None or time_slider_value == None:
		figure = {
			'data': [],
			'layout': go.Layout(title = "Instantenous Agent Speed",xaxis = {'title':'Time Step, Time Period = %.1fs'%(vtp_data_dict["local10m"]["time-period"])}, yaxis = {'title':"Agent Speed (km/hr)"}, font={"family": "Old Standard TT, serif", 'size': 15})
		}

		return figure

	else:
		agent_vel_trace = go.Scatter(

			y = vtp_data_dict[case]["data"][traffic_density][0]["agent_vel"][0:time_slider_value] * 3.6,
			x = np.arange(vtp_data_dict[case]["episode-length"])[0:time_slider_value]
			)

		figure = {
			'data': [agent_vel_trace],
			'layout': go.Layout(title = "Instantenous Agent Speed",xaxis = {'title':'Time Step, Time Period = %.1fs'%(vtp_data_dict["local10m"]["time-period"])}, yaxis = {'title':"Agent Speed (km/hr)"}, font={"family": "Old Standard TT, serif", 'size': 15})
		}

		return figure

@app.callback(Output('vtp-frames', 'src'),
	[Input('vtp-case-dropdown', 'value'), Input('vtp-density-dropdown', 'value'), Input('time-slider', 'value')])
def update_vts_graphs(case, traffic_density, time_slider_value):

	print(time_slider_value)
	if case == None or traffic_density == None or time_slider_value == None:
		pass
	else:
		if case == "local10m":
			imgPath = "https://raw.githubusercontent.com/mynkpl1998/" + repository_name + "/master/SingleLaneIDM/Common/localview10mtrs/Images/" + str(traffic_density) + "_%d.png"%(time_slider_value)
		elif case == "comm10m_reg2":
			imgPath = "https://raw.githubusercontent.com/mynkpl1998/" + repository_name + "/master/SingleLaneIDM/Common/comm10m_reg2/Images/" + str(traffic_density) + "_%d.png"%(time_slider_value)
		elif case =="comm10m_reg4":
			imgPath = "https://raw.githubusercontent.com/mynkpl1998/" + repository_name + "/master/SingleLaneIDM/Common/comm10m_reg4/Images/" + str(traffic_density) + "_%d.png"%(time_slider_value)
		
		return imgPath

@app.callback(Output('slider-text', 'children'),
	[Input('time-slider', 'value')])
def update_slider_text(slider_value):
	if slider_value == None:
		return ["Current simulation time : 0.0 seconds, steps : 0"]
	else:
		return ["Current simulation time : %.1f seconds, steps = %d"%(vtp_data_dict["local10m"]["time-period"] * (slider_value), (slider_value))]

@app.callback(Output('set-time-slider', 'placeholder'),
	[Input('time-slider', 'value')])
def update_input_box_text(slider_value):
	if slider_value == None:
		pass
	else:
		box_text = str(slider_value)
		return box_text

@app.callback(Output('time-slider', 'value'),
	[Input('set-time-slider', 'value')])
def update_time_slider_value(set_time_slider_value):
	if set_time_slider_value == None:
		pass
	else:
		try:
			return int(set_time_slider_value)
		except:
			return int(0)


@app.callback(Output('vtp-cum-reward', 'figure'),
	[Input('vtp-case-dropdown', 'value'), Input('vtp-density-dropdown', 'value'), Input('time-slider', 'value')])
def update_vts_graphs(case, traffic_density, time_slider_value):
	
	if case == None or traffic_density == None or time_slider_value == None:
		figure = {
			'data': [],
			'layout': go.Layout(title = "Cumulative Reward",xaxis = {'title':'Time Step, Time Period = %.1fs'%(vtp_data_dict["local10m"]["time-period"])}, yaxis = {'title':"Cum. Reward"}, font={"family": "Old Standard TT, serif", 'size': 15})
		}

		return figure

	else:
		agent_rew_trace = go.Scatter(

			y = vtp_data_dict[case]["data"][traffic_density][0]["reward"][0:time_slider_value],
			x = np.arange(vtp_data_dict[case]["episode-length"])[0:time_slider_value]
			)

		figure = {
			'data': [agent_rew_trace],
			'layout': go.Layout(title = "Cumulative Reward", xaxis = {'title':'Time Step, Time Period = %.1fs'%(vtp_data_dict["local10m"]["time-period"])}, yaxis = {'title':"Cum. Reward"}, font={"family": "Old Standard TT, serif", 'size': 15})
		}
		return figure

@app.callback(Output('vtp-action-distribution', 'figure'),
	[Input('vtp-case-dropdown', 'value'), Input('vtp-density-dropdown', 'value'), Input('time-slider', 'value')])
def update_vts_action_dist(case, traffic_density, time_slider_value):

	if case == None or traffic_density == None or time_slider_value == None:
		
		figure = {
			'data': [],
			'layout': go.Layout(title = "Policy Distribution", yaxis = {'title':"Probability", "range": [0,1]}, font={"family": "Old Standard TT, serif", 'size': 15})
		}
		return figure

	else:
		distribution = vtp_data_dict[case]["data"][traffic_density][0]["probs"][time_slider_value]
		print(distribution)
		x_data = list(distribution.keys())
		y_data = []
		for element in x_data:
			y_data.append(distribution[element])

		dist_trace = go.Bar(
			x = x_data,
			y = y_data
			)

		figure = {
			'data': [dist_trace],
			'layout': go.Layout(title = "Policy Distribution", yaxis = {'title':"Probability", "range": [0,1],}, font={"family": "Old Standard TT, serif", 'size': 15})
		}

		return figure

if __name__ == "__main__":

	ADDRESS = "0.0.0.0"
	PORT = "8051"
	app.run_server(debug=True, host=ADDRESS, port=PORT)