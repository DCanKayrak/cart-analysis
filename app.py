from flask import Flask, request, jsonify
import aiV2
from collections import defaultdict

app = Flask(__name__)

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
def usersWith():
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

if __name__ == '__main__':
    app.run(debug=True)
