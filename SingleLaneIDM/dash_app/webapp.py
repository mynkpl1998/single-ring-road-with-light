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

# --------------- Read data for results section -------------------#
local_view_res_datafile = "SingleLaneIDM/OnlyLocalView/Results/dataset.pkl"
full_comm_res_datafile = "SingleLaneIDM/CommFullAcess/Results/dataset.pkl"
restricted_comm_res_data_file = "SingleLaneIDM/CommRestrictedAccess/Results/dataset.pkl"
res_data_dict = {}
res_data_dict["res-local-view"] = readPickle(local_view_res_datafile)
res_data_dict["res-full-comm"] = readPickle(full_comm_res_datafile)
res_data_dict["res-restricted-comm"] = readPickle(restricted_comm_res_data_file)

case_reverse_map = {}
case_reverse_map["res-local-view"] = "Only Local View"
case_reverse_map["res-full-comm"] = "Full access to Communication"
case_reverse_map["res-restricted-comm"] = "Restricted access to Communication"

# --------------- Read data for results section -------------------#

# ---------------- Cases Map ---------------------------#
case_description = {}
case_description["res-local-view"] = "In Local View Only case, Ego vehicle perception range is limited by the data which can be sensed locally using lidars and radars. We set 20 metres of local view (including both front and back) in our simulations. Agent is rewarded for moving. We found that learned policy never picks the speed beyond which it can't decelerate for a given deceleration rate, if a vehicle suddenly comes infront of it. Thus, limiting the maximum speed at which Ego vehcile can travel in free-space."
case_description["res-full-comm"] = "In Local View with Full Access to Communication, Ego vehicle has access to a local view and can receive information of the regions apart from what is locally available over the network. We termed this communicable region as Extended View. We set 20 metres of Extended View (including both ahead and behind the ego vehicle) in simulations. Agent is rewarded for moving and a small reward is added whenever agent chooses not to query which discourages unnecessary communication. We found learned policy was able to pick much higher speed without colliding into other vehciles compared to only Local View. This is because, Communication extends the range of perception of the Ego vehcile which leads much higher driving utlities. In this case we assumed single query can fetch information of whole extended view."
case_description["res-restricted-comm"] = "In Local View with Restricted Access to Communication, Ego vehcile has access to local view and can receive information of the regions apart from what is locally available over the network. However, Communication systems have fixed data rate which limits the amount of information which can be exchanged over the network. To simulate the same, extended view is divided into further smaller regions, which restricts the whole information of extended view to be exchanged in a single query. Agent is rewarded for moving and a small reward is added whenever agent chooses not to query which discourages unnecessary communication. We found agent learned to pick a speed much higher than what is allowed by local view and comparable to when agent has access to whole information of extended view. "
# ---------------- Cases Map ---------------------------#


# Create Traffic Density Dropdown list
traffic_density_dropdown_list = []
for density in vtp_data_dict["local10m"]["data"].keys():
	d = {}
	d["label"] = "Num. Vehicles = %d"%(density)
	d["value"] = density
	traffic_density_dropdown_list.append(d)


app = dash.Dash(__name__)


# Bootstrap CSS
app.css.append_css({'external_url': 'https://codepen.io/mynkpl1998/pen/QPgaXw.css'})

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
					html.H6(children=[html.A("Mayank K. Pal", href="https://scholar.google.co.in/citations?user=ZVRzQ9AAAAAJ&hl=en"), html.A(", Sanjit K. Kaul", href="https://scholar.google.co.in/citations?user=XGNQPRsAAAAJ&hl=en"), html.A(", Saket Anand", href="https://scholar.google.co.in/citations?user=YmYvVEQAAAAJ&hl=en")], style={'fontFamily': 'HelveticaNeue'})
				], className="eight columns", style = {"margin-top": 20}),

			html.Div([

				html.Img(src="https://i.imgur.com/vaYjkdF.png",
					style={
						"height": "70%",
						"width": "70%",
						"float": "right",
						"positive":  "relative",
						"margin-top": 20
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
				html.Label(children="Jump to step : (max = %d)"%(vtp_data_dict["local10m"]["episode-length"] - 1), style={"fontFamily": "HelveticaNeue", "fontWeight": "bold", 'fontSize': 20}),
				dcc.Input(id="set-time-slider", type="text", max=vtp_data_dict["local10m"]["episode-length"] - 1, debounce=True, min=0)

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

			], className="row"),

		# ---- Visualizing Policy Ends ---- #

		html.Hr(),

		html.Div([

			html.Div([

				html.H2(html.B("Results"), style={'fontFamily': 'HelveticaNeue', 'text-align': ''})

				], className="row"),

			html.Div([

					html.Label(children="Select case(s) : ", style={"fontFamily": "HelveticaNeue", "fontWeight": "bold", 'fontSize': 20}),
					dcc.Checklist(id="res-checklist",
    				options=[
        					{'label': 'Only Local View  ', 'value': 'res-local-view'},
        					{'label': 'Full access to Communication  ', 'value': 'res-full-comm'},
        					{'label': 'Restricted access to Communication  ', 'value': 'res-restricted-comm'}
    					],
    					values=['res-local-view'], style={"margin-left": 20, 'fontFamily': 'HelveticaNeue', 'fontSize': 18,}, labelStyle= {'display': "inline-block"}),

					html.Br(),
					html.Label(children="Explanation : ", style={"fontFamily": "HelveticaNeue", "fontWeight": "bold", 'fontSize': 20}),
					html.Label(id="case-explnation-block", style={"fontFamily": "HelveticaNeue", 'fontSize': 18}),


				], className="row"),

			html.Div([

					html.Div([
						dcc.Graph(id="res-avg-speed")
					], className="six columns"),

					html.Div([
						dcc.Graph(id="res-avg-reward")
					], className="six columns")

				
				], className="row"),

			html.Br(),


			html.Div([

					html.Div([

						dcc.Dropdown(id="plan-action-dropdown",
							options= [
								{"label": "Accelerate", "value": "acc"},
								{"label": "Decelerate", "value": "dec"},
								{"label" :"Do-Nothing (acc= 0)", "value": "do-nothing"}
							],
							placeholder = "Planning Action"
							)

						], className="five columns", style = {"fontFamily" : "HelveticaNeue", "fontWeight": "bold"}),

					html.Div([

						dcc.Dropdown(id="query-action-dropdown",
							options= [
								{"label": "Accelerate", "value": "acc"},
								{"label": "Decelerate", "value": "dec"},
								{"label" :"Do-Nothing (acc= 0)", "value": "don"}
							],
							value = ["acc"]

							)

						], className="five columns",)

				], className="row"),

			html.Div([

					html.Div([
						dcc.Graph(id="res-act-dist")
					], className="five columns"),

					html.Div([
						dcc.Graph(id="res-reg-dist")
					], className="five columns")

				
				], className="row")

		], className="row")


		], className = "ten columns offset-by-one")
	)

def cal_planner_percentage(data_dict):
	num_episodes = data_dict["num_episodes"]
	episode_length = data_dict["episode-length"]
	
	possible_actions = {'acc': 0, 'dec':1, 'do-nothing':2}
	act_dist = {}
	densities = []

	for density in data_dict["data"].keys():

		densities.append(density)

		act_dist[density] = copy.deepcopy(possible_actions)

		for episode_num in data_dict["data"][density]:
			for step in range(0, episode_length):
				act_dist[density][data_dict["data"][density][episode_num]["planner_actions"][step]] += 1

	return densities, act_dist

def avg_speed(data_dict):
	global_vel_data = {}

	for density in data_dict["data"].keys():
	    
	    vel_data = []
	    
	    for episode_num in data_dict["data"][density].keys():
	            
	            episode_data = []
	            
	            for step in range(0, len(data_dict["data"][density][episode_num]["agent_vel"])):
	                episode_data.append(data_dict["data"][density][episode_num]["agent_vel"][step])
	            
	            vel_data.append(episode_data)
	    
	    global_vel_data[density] = copy.deepcopy(vel_data)



	avg_speeds = []
	densities = []

	for density in global_vel_data.keys():
	    densities.append(density)
	    
	    speed_sum = 0.0
	    elements_count = 0
	    for episode in global_vel_data[density]:
	        for vel in episode:
	            speed_sum += vel
	            elements_count += 1
	    
	    avg_speeds.append((speed_sum/elements_count) * 3.6)

	return densities, avg_speeds

def avg_cum_reward(data_dict):
	num_episodes = data_dict["num_episodes"]
	episode_length = data_dict["episode-length"]
	avg_rewards = []
	densities = []

	for density in data_dict["data"].keys():
	    densities.append(density)
	    
	    cum_reward_sum = 0.0
	    elements_count = 0
	    
	    for episode in data_dict["data"][density]:
	        cum_reward_sum += data_dict["data"][density][episode]["cum_reward"]
	    avg_rewards.append(cum_reward_sum/num_episodes)

	return densities, avg_rewards

@app.callback(Output('case-explnation-block', 'children'),
	[Input('res-checklist', 'values')])
def update_explanation_block(values):

	if len(values) == 0:
		return "Select a Case"
	elif len(values) > 1:
		return "Camparing Different Cases."
	else:
		return case_description[values[0]]

@app.callback(Output('res-act-dist', 'figure'),
	[Input('res-checklist', 'values'), Input('plan-action-dropdown', 'value')])
def update_results_action_distribution(Checklist_val, planning_action_value):

	#print(Checklist_val)
	if len(Checklist_val) == 0 or planning_action_value == None:

		figure = {
			'data': [],
			'layout': go.Layout(title = "Action Distribution", xaxis={'title': 'Traffic Density'}, yaxis = {'title': 'Percentage (%)'}, font={"family": "Old Standard TT, serif", 'size': 15})
		}

		return figure

	else:
		layout_data = []
		for case in Checklist_val:
			x_data, act_dict = cal_planner_percentage(res_data_dict[case])

			y_data = []
			for dens in act_dict:
				y_data.append(act_dict[dens][planning_action_value])

			act_trace = go.Bar(
					y = y_data,
					x = x_data,
					name = case_reverse_map[case]
				)

			layout_data.append(act_trace)

		figure = {
			'data': layout_data,
			'layout': go.Layout(title = "Action Distribution", xaxis={'title': 'Traffic Density'}, yaxis = {'title': 'Percentage (%)'}, font={"family": "Old Standard TT, serif", 'size': 15})
		}

		return figure

			


@app.callback(Output('res-avg-speed', 'figure'),
	[Input('res-checklist', 'values')])
def update_results_avg_speed(values):
	if len(values) == 0:

		figure = {
			'data': [],
			'layout': go.Layout(legend = {'orientation' : "h"}, title = "Average Agent Speed", yaxis = {'title': 'Speed (km/hr)'}, font={"family": "Old Standard TT, serif", 'size': 15})
		}

		return figure
	else:
		layout_data = []

		for case in values:
			x_data, y_data = avg_speed(res_data_dict[case])
			#print(x_data)
			vel_trace = go.Bar(
					y = y_data,
					x = x_data,
					name = case_reverse_map[case]
				)
			layout_data.append(vel_trace)


		figure = {
			'data': layout_data,
			'layout': go.Layout(legend = {'orientation' : "h"}, title = "Average Agent Speed", yaxis = {'title': 'Speed (km/hr)'}, font={"family": "Old Standard TT, serif", 'size': 15})
		}

		return figure


@app.callback(Output('res-avg-reward', 'figure'),
	[Input('res-checklist', 'values')])
def update_results_rewards(values):
	if len(values) == 0:

		figure = {
			'data': [],
			'layout': go.Layout(legend = {'orientation' : "h"}, title = "Average Cumulative Reward", xaxis={'title': 'Traffic Density'}, yaxis = {'title': 'Cumulative Reward'}, font={"family": "Old Standard TT, serif", 'size': 15})
		}

		return figure
	else:
		layout_data = []

		for case in values:
			x_data, y_data = avg_cum_reward(res_data_dict[case])
			#print(x_data)
			vel_trace = go.Bar(
					y = y_data,
					x = x_data,
					name = case_reverse_map[case]
				)
			layout_data.append(vel_trace)


		figure = {
			'data': layout_data,
			'layout': go.Layout(legend = {'orientation' : "h"}, title = "Average Cumulative Reward", yaxis = {'title': 'Cumulative Reward'}, font={"family": "Old Standard TT, serif", 'size': 15})
		}

		return figure

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

	#print(time_slider_value)
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
		#print(distribution)
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