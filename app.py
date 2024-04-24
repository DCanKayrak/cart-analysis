from flask import Flask
from flask import request
import aiV2

app = Flask(__name__)

@app.route('/products')
def products():
    return aiV2.orders_order_items['product_id'].to_list()

@app.route('/users')
def users():
    return aiV2.orders_order_items['customer_id'].unique().tolist()

# '9ef432eb6251297304e76186b10a928d'
@app.route('/predict')
def make_prediction():
    user_id = request.args.get('userId',type= str)
    products = aiV2.orders_order_items[aiV2.orders_order_items['customer_id'] == user_id]['product_id'].unique()
    print(products)
    return aiV2.get_recommendations(user_id)

if __name__ == '__main__':
    app.run(debug=True)
