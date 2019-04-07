import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__)

# Bootstrap CSS
app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})

# Define all variables here
webpage_title = "Age Optimal Updates for Autonomous Driving"

# Set Webpage title
print(app.server)

# Header 
app.layout = html.Div(

	html.Div([
		html.Div([

			html.Div([

					html.H1(style={'fontColor': 'blue', 'fontFamily': 'HelveticaNeue'}, children=webpage_title),
					html.H6(children="Mayank K. Pal, Sanjit K. Kaul, Saket Anand", style={'fontFamily': 'HelveticaNeue'})
				], className="nine columns"),

			html.Div([

				html.Img(src="/dash_app/data/logo.png")

				], className="three columns")

			], className="row")
		], className = "ten columns offset-by-one")
	)


if __name__ == "__main__":
	ADDRESS = "0.0.0.0"
	app.run_server(debug=True, host=ADDRESS)