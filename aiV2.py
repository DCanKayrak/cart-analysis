import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import random

# Ürün isimleri
product_names = [
    'Wireless Mouse', 'Bluetooth Headphones', 'Laptop Stand', 'Portable Charger', 'USB-C Hub', 'Smartwatch', 
    'Noise Cancelling Headphones', 'Fitness Tracker', 'External Hard Drive', 'LED Desk Lamp', 'Ergonomic Chair', 
    'Mechanical Keyboard', 'Gaming Mouse', '4K Monitor', 'Smartphone Stand', 'Wireless Earbuds', 'Bluetooth Speaker', 
    'Smart Home Hub', 'Robot Vacuum', 'Smart Light Bulb', 'Portable Projector', 'Webcam', 'Streaming Microphone', 
    'Graphics Tablet', 'Smart Thermostat', 'Electric Toothbrush', 'Digital Photo Frame', 'E-Reader', 'Smart Lock', 
    'Wireless Security Camera', 'Air Purifier', 'Electric Kettle', 'Espresso Machine', 'Blender', 'Slow Cooker', 
    'Rice Cooker', 'Air Fryer', 'Food Processor', 'Stand Mixer', 'Toaster Oven', 'Electric Griddle', 'Instant Pot', 
    'Electric Skillet', 'Cordless Drill', 'Tool Set', 'Circular Saw', 'Electric Screwdriver', 'Lawn Mower', 
    'Leaf Blower', 'Pressure Washer', 'Garden Hose', 'Garden Shears', 'Smart Sprinkler', 'Fitness Bike', 'Treadmill', 
    'Yoga Mat', 'Dumbbell Set', 'Kettlebell', 'Resistance Bands', 'Foam Roller', 'Weight Bench', 'Jump Rope', 
    'Pull-Up Bar', 'Running Shoes', 'Hiking Boots', 'Tent', 'Sleeping Bag', 'Camping Stove', 'Backpack', 'Flashlight', 
    'Binoculars', 'Fishing Rod', 'Cooler', 'Inflatable Kayak', 'Paddle Board', 'Snorkel Set', 'Wet Suit', 'Life Jacket', 
    'Picnic Blanket', 'Portable Grill', 'Fire Pit', 'Outdoor Speaker', 'Hammock', 'Solar Charger', 'Water Bottle', 
    'Travel Mug', 'Luggage Set', 'Packing Cubes', 'Travel Pillow', 'Travel Adapter', 'Portable WiFi Hotspot', 
    'Noise-Cancelling Earplugs', 'Travel Backpack', 'Action Camera', 'Drone', 'Digital Camera', 'Camera Lens', 
    'Tripod', 'Memory Card', 'Camera Bag', 'Photo Printer'
]

# Veri setlerini okuyun
df_customers = pd.read_csv('data/olist_customers_dataset.csv')
df_products = pd.read_csv('data/olist_products_dataset.csv')
df_order_items = pd.read_csv('data/olist_order_items_dataset.csv')
df_orders = pd.read_csv('data/olist_orders_dataset.csv')

# Gerekli veri birleştirmelerini yapın
orders_order_items = df_orders.merge(df_order_items, on='order_id').head(100)

# Kullanıcıların aldığı ürünleri temsil eden bir özellik vektörü oluşturma
user_product_matrix = orders_order_items.pivot_table(index='customer_id', columns='product_id', aggfunc='size', fill_value=0)

# KNN modelini oluşturma
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(user_product_matrix)

# Yeni bir kullanıcı için önerileri alma
def get_recommendations(user_id, k=5):
    user_index = user_product_matrix.index.get_loc(user_id)
    distances, indices = knn_model.kneighbors([user_product_matrix.iloc[user_index]], n_neighbors=k+1)
    indices = indices.flatten()[1:]
    similar_users = user_product_matrix.iloc[indices]
    recommended_products = similar_users.sum().sort_values(ascending=False).head(10)
    recommended_product_ids = recommended_products.index.tolist()
    return recommended_product_ids

# Kullanıcı ürün önerilerini ve rastgele ürün isimlerini döndürme
def get_user_recommendations():
    users = orders_order_items['customer_id'].unique()
    user_predictions_list = []
    
    for user_id in users:
        predictions = get_recommendations(user_id)
        predicted_products = df_products[df_products['product_id'].isin(predictions)]
        
        grouped_predictions = defaultdict(list)
        for _, row in predicted_products.iterrows():
            grouped_predictions[row['product_category_name']].append(row['product_id'])
        
        formatted_predictions = [{'category': category, 'product_ids': [random.choice(product_names) for _ in ids]} for category, ids in grouped_predictions.items()]
        
        user_predictions_list.append({
            'user_id': user_id,
            'products': formatted_predictions
        })
    
    return user_predictions_list
