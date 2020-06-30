#!/usr/bin/python
import numpy as np
from flask import Flask
from flask_restx import Api, Resource, fields
from model_deployment import predict

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Used Vehicle Price Prediction',
    description='Machine Learning model that predicts the price of a used car in the US')

ns = api.namespace('predict', 
     description='Price Prediction')
   
parser = api.parser()

parser.add_argument(
    'year', 
    type=int, 
    required=True, 
    help='year model of the car', 
    location='args')

parser.add_argument(
    'mileage', 
    type=int, 
    required=True, 
    help='mileage of the car', 
    location='args')

parser.add_argument(
    'state', 
    type=str, 
    required=True, 
    help='US state', 
    location='args')

parser.add_argument(
    'make', 
    type=str, 
    required=True, 
    help='maker of the car', 
    location='args')

parser.add_argument(
    'model', 
    type=str, 
    required=True, 
    help='model of the car', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class PriceApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
    
        return {
         "result": np.around(predict(args.year,args.mileage,args.state,args.make,args.model),decimals=0)
        }, 200
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
