from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
from plotly.subplots import make_subplots
import requests
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import FastICA

server = Flask(__name__)
app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/', external_stylesheets=[dbc.themes.MORPH])
CONNSTR = f'postgresql://Blaine-Mason:ALrWl9PMXok2@ep-shy-poetry-131814.us-east-2.aws.neon.tech/neondb'
app.server.config['SQLALCHEMY_DATABASE_URI'] = CONNSTR
db = SQLAlchemy(app.server)
migrate = Migrate(app.server, db)

class binary(db.Model):
    __tablename__ = 'binarysystem'
    filename = db.Column(db.String, primary_key=True)
    mjd = db.Column(db.Integer())
    apogeeID = db.Column(db.String())
    visitno =  db.Column(db.Integer())
    fluxa = db.Column(db.ARRAY(db.Float()))
    wavelengtha =  db.Column(db.ARRAY(db.Float()))
    fluxb = db.Column(db.ARRAY(db.Float()))
    wavelengthb = db.Column(db.ARRAY(db.Float()))
    fluxc = db.Column(db.ARRAY(db.Float()))
    wavelengthc = db.Column(db.ARRAY(db.Float()))
    ampa =  db.Column(db.Float())
    vhelioa = db.Column(db.Float())
    fwhma = db.Column(db.Float())
    ampb = db.Column(db.Float())
    vheliob = db.Column(db.Float())
    fwhmb = db.Column(db.Float())
    SNR = db.Column(db.Float())

    def __init__(self, filename, mjd, apogeeID, visitno, fluxa, wavelengtha, fluxb, wavelengthb, fluxc, wavelengthc,
                 ampa, vhelioa, fwhma, ampb, vheliob, fwhmb, SNR):
        self.filename = filename
        self.mjd = mjd
        self.apogeeID = apogeeID
        self.visitno =  visitno
        self.fluxa = fluxa
        self.wavelengtha = wavelengtha
        self.fluxb = fluxb
        self.wavelengthb = wavelengthb
        self.fluxc = fluxc
        self.wavelengthc = wavelengthc
        self.ampa = ampa
        self.vhelioa = vhelioa
        self.fwhma = fwhma
        self.ampb = ampb
        self.vheliob = vheliob 
        self.fwhmb = fwhmb
        self.SNR = SNR
        
    def __repr__(self):
        return f"<Star {self.apogeeID}>"
#-------------------------------------------------------------
class ccf_final(db.Model):
    __tablename__ = 'ccfdatacomplete'
    apogeeID = db.Column(db.String(), primary_key=True)
    nvisits =  db.Column(db.Integer())
    mjd = db.Column(db.ARRAY(db.Integer()))
    ccf = db.Column(db.ARRAY(db.Float()))

    def __init__(self, apogeeID, nvisits, mjd, ccf):
       self.apogeeID = apogeeID
       self.nvisits = nvisits
       self.mjd = mjd
       self.ccf = ccf

    def __repr__(self):
        return f"<Star {self.apogeeID}>"
#-------------------------------------------------------------
#HELPERS
def ICA_helper(data):
    ica = FastICA(n_components=2, whiten='unit-variance')
    S_ica_ = ica.fit_transform(data.T) # Estimate the sources
    return S_ica_
# @app.callback(Output(component_id='CCF', component_property= 'figure'),
#               [Input(component_id='star_dropdown', component_property= 'value')], prevent_initial_call=True)
# def ccf_plot(star_dropdown):
#     app_id = star_dropdown.split(" ")[0]
#     data = requests.get(f'http://127.0.0.1:8050/get-binary/{app_id}').json()
#     data = data[star_dropdown.replace(" ","")]
#     df = pd.DataFrame.from_dict(data)

@app.callback(Output(component_id='ICA', component_property= 'figure'),
              [Input(component_id='star_dropdown', component_property= 'value')], prevent_initial_call=True)
def ICA_plot(star_dropdown):
    app_id = star_dropdown.split(" ")[0]
    data = requests.get(f'http://127.0.0.1:8050/get-binary/{app_id}').json()
    data = data[star_dropdown.replace(" ","")]
    df = pd.DataFrame.from_dict(data)
    ica_A = np.vstack((df['A'][1], df['A'][0]))
    ica_B = np.vstack((df['B'][1], df['B'][0]))
    ica_C = np.vstack((df['C'][1], df['C'][0]))
    s_A = ICA_helper(ica_A)
    s_B = ICA_helper(ica_B)
    s_C = ICA_helper(ica_C)

    fig = make_subplots(rows=1, cols=3)
    fig.add_trace(go.Scatter(x = df['A'][1], y = s_A[:,0], mode="lines", line_color='rgb(38,84,124)'), row=1, col=1)
    fig.add_trace(go.Scatter(x = df['A'][1], y = s_A[:,1], mode="lines", line_color='rgb(239,71,11)'), row=1, col=1)
    fig.add_trace(go.Scatter(x = df['B'][1], y = s_B[:,0], mode="lines", line_color='rgb(38,84,124)'), row=1, col=2)
    fig.add_trace(go.Scatter(x = df['B'][1], y = s_B[:,1], mode="lines", line_color='rgb(239,71,11)'), row=1, col=2)
    fig.add_trace(go.Scatter(x = df['C'][1], y = s_C[:,0], mode="lines", line_color='rgb(38,84,124)'), row=1, col=3)
    fig.add_trace(go.Scatter(x = df['C'][1], y = s_C[:,1], mode="lines", line_color='rgb(239,71,11)'), row=1, col=3)
    sb2_data = requests.get(f'http://127.0.0.1:8050/get-av/{app_id}').json()
    sb2_data = sb2_data[star_dropdown.replace(" ","")]
    amp = sb2_data["AMP"]
    vhelio = sb2_data["VHELIO"]
    fwhm = sb2_data["FWHM"]
    fig.update_layout(title_text=f"AMP: {amp} \nVHELIO: {vhelio} \nFWHM: {fwhm}")
    return fig  

@app.callback(Output(component_id='waveplot', component_property= 'figure'),
              [Input(component_id='star_dropdown', component_property= 'value')], prevent_initial_call=True)
def wavelength_plot(star_dropdown):
    data = requests.get(f'http://127.0.0.1:8050/get-binary/{star_dropdown[0:18]}').json()
    data = data[star_dropdown.replace(" ","")]
    df = pd.DataFrame.from_dict(data)
    fig = make_subplots(rows=1, cols=3)
    fig.add_trace(go.Scatter(x = df['A'][1], y = df['A'][0], mode="lines",
    line_color='blue'), row=1, col=1)
    fig.add_trace(go.Scatter(x = df['B'][1], y = df['B'][0], mode="lines",
    line_color='green'), row=1, col=2)
    fig.add_trace(go.Scatter(x = df['C'][1], y = df['C'][0], mode="lines",
    line_color='red'), row=1, col=3)
    return fig

@app.callback(Output('star_dropdown', 'options'),
              [Input('snr', 'value')], prevent_initial_call=True)
def update_options(value):
    ids = requests.get(f'http://127.0.0.1:8050/get-snr/{value}')
    result = dict(sorted(ids.json().items(), key=lambda item: item[1]))
    print(result)
    all_ids_snr = list(set([i[1][0] for i in result.items()]))
    all_ids = [i[1][0]  + " " + str(i[1][1]) for i in result.items()]
    ret_list = []
    for x,y in zip(all_ids_snr, all_ids):
        ret = {'label': x, 'value': y}
        ret_list.append(ret)
    print(ret_list)
    return ret_list

@app.callback(Output('mjd_drop', 'options'),
              [Input('snr', 'value'), Input('star_dropdown', 'value')], prevent_initial_call=True)
def update_mjd_options(snr, mjd):
    ids = requests.get(f'http://127.0.0.1:8050/get-snr/{snr}')
    result = dict(sorted(ids.json().items(), key=lambda item: item[1]))
    print(result)
    all_mjd = [i[1][0] for i in result.items()]
    all_ids = [i[1][0]  + " " + str(i[1][1]) for i in result.items()]
    ret_list = []
    for x,y in zip(all_ids_snr, all_ids):
        ret = {'label': x, 'value': y}
        ret_list.append(ret)
    print(ret_list)
    return ret_list
#-------------------------------------------------------------
#HTML
controls = dbc.Card(
    [
        html.Div(
            [
                
                dbc.Label("SNR Dropdown Menu"),
                #INSERT DROPDOWN,
            dcc.Dropdown( id = 'snr',
            options = [
                {'label':'50', 'value':'50'},
                {'label':'100', 'value':'100'},
                {'label':'150', 'value':'150'},
                {'label':'200', 'value':'200'},
                {'label':'250', 'value':'250'},
                {'label':'200', 'value':'200'},
                {'label':'350', 'value':'350'},
                {'label':'400', 'value':'400'},
                {'label':'450', 'value':'450'},
                {'label':'500', 'value':'500'},
                ],
                clearable=False),
            
        ]
        ),
        html.Div(
            [
                dbc.Label("STARZ Dropdown Menu"),
                dcc.Dropdown( id = 'star_dropdown')
            ]
        ),
        html.Div(
            [
                dbc.Label("MJD"),
                dcc.Dropdown( id = 'mjd_drop')
            ]
        ),
    ],
    body=True,
)
#DEFINE app.layout HERE
app.layout = dbc.Container(
    [
        html.H1("APOGEE ICA "),
        # html.Hr(),
        dbc.Row([
            dbc.Col(controls,md=6),
        ],
        align = "center",
        ),
        dbc.Row([
            dbc.Col([dcc.Graph(id = 'waveplot'), dcc.Graph(id = 'ICA')]),
            
        ],
        align = "center",
        ),
    ],
    fluid=True,
)
#-------------------------------------------------------------
#ROUTES
#RETURN HERE
@app.server.route('/addccf', methods=['POST'])
def handle_ccf():
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            temp_star = ccf_final(apogeeID= data["apogeeID"], nvisits= data["nvisits"], mjd= data["mjd"], ccf= data["ccf"])
            db.session.add(temp_star)
            db.session.commit()
            return {"message": f"Star {temp_star.apogeeID} has been created successfully."}
        else:
            return {"error": "The request payload is not in JSON format"}
        
@app.server.route('/test', methods=['POST'])
def handle_test():
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            temp_star = binary(filename = data["filename"], apogeeID= data["apogeeID"], visitno= data["visitno"], fluxa= data["fluxa"], wavelengtha= data["wavelengtha"], fluxb= data["fluxb"], wavelengthb= data["wavelengthb"], fluxc= data["fluxc"], wavelengthc= data["wavelengthc"],
                 ampa= data["ampa"], vhelioa= data["vhelioa"], fwhma= data["fwhma"], ampb= data["ampb"], vheliob= data["vheliob"], fwhmb= data["fwhmb"], SNR= data["SNR"], mjd= data["mjd"])
            db.session.add(temp_star)
            db.session.commit()
            return {"message": f"Star {temp_star.apogeeID} has been created successfully."}
        else:
            return {"error": "The request payload is not in JSON format"}

@app.server.route('/get-snr/<target_snr>', methods=['GET'])
def handle_snr(target_snr):
    sample = dict()
    star = binary.query.filter(binary.SNR >= target_snr)
    i= 0
    for obj in star:
        sample[str(i)] = [obj.apogeeID,obj.SNR]
        i += 1
    return jsonify(sample)

@app.server.route('/get-av/<apogee_id_str>', methods=['GET'])
def handle_av(apogee_id_str):
    response = dict()
    star = binary.query.filter(binary.apogeeID == apogee_id_str).all()
    for obj in star:
        sample = dict()
        sample['AMP'] = [obj.ampa, obj.ampb]
        sample['VHELIO'] = [obj.vhelioa, obj.vheliob]
        sample['FWHM'] = [obj.fwhma, obj.fwhmb]
        response[str(apogee_id_str)+str(obj.SNR)] = sample
    return jsonify(response)

@app.server.route('/get-binary/<apogee_id_str>', methods=['GET'])
def handle_apID(apogee_id_str):
    response = dict()
    star = binary.query.filter(binary.apogeeID == apogee_id_str).all()
    for obj in star:
        sample = dict()
        sample['A'] = [obj.fluxc, obj.wavelengthc]
        sample['B'] = [obj.fluxb, obj.wavelengthb]
        sample['C'] = [obj.fluxa, obj.wavelengtha]
        sample['MJD'] = obj.mjd
        response[str(apogee_id_str)+str(obj.SNR)] = sample
    return jsonify(response)

@app.server.route('/get-ccf/<apogee_id>', methods=['GET'])
def handle_ccf_apID(apogee_id_str):
    response = dict()
    appid = binary.query.filter(binary.apogeeID == apogee_id_str)
    nvisits = appid.nvisits
    for i in range(nvisits):
        sample = dict()
        sample[str(appid.mjd[i])] = appid.ccf[i]
    response = sample
    return jsonify(response)


if __name__ == '__main__':
    app.run_server(debug=True)