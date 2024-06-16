from flask import Flask, request, jsonify
import aiV2
from flask_cors import CORS
from collections import defaultdict
import aiV3
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/products')
def products():
    # Extract product_id and category_name columns
    products_data = aiV2.df_products[['product_id', 'product_category_name']].drop_duplicates()
    # Format data as a list of dictionaries
    products_list = products_data.rename(columns={'product_id': 'id', 'product_category_name': 'category'}).to_dict(orient='records')
    
    # Return the formatted data as JSON
    return jsonify(products_list)

@app.route('/users')
def users():
    return jsonify(aiV2.orders_order_items['customer_id'].unique().tolist())

@app.route('/userv2')
def usersWith1():
    # Get unique user IDs
    users = aiV2.orders_order_items['customer_id'].unique()
    # Create a list to store predictions for each user
    user_predictions_list = []
    
    # Iterate over each user and get predictions
    for user_id in users:
        # Get product predictions for the user
        predictions = aiV2.get_recommendations(user_id)
        
        # Get product category names for the predicted products
        predicted_products = aiV2.df_products[aiV2.df_products['product_id'].isin(predictions)]
        
        # Group predictions by category
        grouped_predictions = defaultdict(list)
        for _, row in predicted_products.iterrows():
            grouped_predictions[row['product_category_name']].append(row['product_id'])
        
        # Format the grouped predictions
        formatted_predictions = [{'category': category, 'product_ids': ids} for category, ids in grouped_predictions.items()]
        
        user_predictions_list.append({
            'user_id': user_id,
            'products': formatted_predictions
        })
    
    # Return the user predictions as a JSON response
    return jsonify(user_predictions_list)

@app.route('/predict')
def make_prediction():
    user_id = request.args.get('userId', type=str)
    products = aiV2.orders_order_items[aiV2.orders_order_items['customer_id'] == user_id]['product_id'].unique()
    print(products)
    return jsonify(aiV2.get_recommendations(user_id))

@app.route('/userv3')
def usersWith():
    user_predictions_list = aiV2.get_user_recommendations()
    return jsonify(user_predictions_list)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'customers' in request.files:
        aiV3.df_customers = pd.read_csv(request.files['customers'])
    if 'products' in request.files:
        aiV3.df_products = pd.read_csv(request.files['products'])
    if 'order_items' in request.files:
        aiV3.df_order_items = pd.read_csv(request.files['order_items'])
    if 'orders' in request.files:
        aiV3.df_orders = pd.read_csv(request.files['orders'])

    aiV3.train_model()

    return jsonify({"message": "Files uploaded and model trained successfully!"})

@app.route('/top-products')
def top_products():
    top_products = aiV2.df_products['product_id'].value_counts().head(10).index.tolist()
    top_products_names = aiV2.df_products[aiV2.df_products['product_id'].isin(top_products)][
        'product_category_name'].tolist()
    return jsonify(top_products_names)

@app.route('/top-users')
def top_users():
    top_users = aiV2.df_orders['customer_id'].value_counts().head(10).index.tolist()
    return jsonify(top_users)

if __name__ == '__main__':
    app.run(debug=True)
