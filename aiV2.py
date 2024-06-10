import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import random

# Ürün isimleri
product_names = [
    'Mouse sem Fio', 'Fones de Ouvido Bluetooth', 'Suporte para Laptop', 'Carregador Portátil', 'Hub USB-C', 'Relógio Inteligente', 
    'Fones de Ouvido com Cancelamento de Ruído', 'Rastreador de Atividade', 'Disco Rígido Externo', 'Lâmpada de Mesa LED', 'Cadeira Ergonômica', 
    'Teclado Mecânico', 'Mouse para Jogos', 'Monitor 4K', 'Suporte para Smartphone', 'Fones de Ouvido sem Fio', 'Alto-falante Bluetooth', 
    'Hub para Casa Inteligente', 'Aspirador de Pó Robô', 'Lâmpada Inteligente', 'Projetor Portátil', 'Webcam', 'Microfone para Streaming', 
    'Mesa Digitalizadora', 'Termostato Inteligente', 'Escova de Dentes Elétrica', 'Porta-retratos Digital', 'Leitor de E-books', 'Fechadura Inteligente', 
    'Câmera de Segurança sem Fio', 'Purificador de Ar', 'Chaleira Elétrica', 'Máquina de Espresso', 'Liquidificador', 'Panela de Cozimento Lento', 
    'Panela de Arroz', 'Fritadeira sem Óleo', 'Processador de Alimentos', 'Batedeira', 'Forno Elétrico', 'Chapa Elétrica', 'Panela de Pressão Elétrica', 
    'Frigideira Elétrica', 'Furadeira sem Fio', 'Conjunto de Ferramentas', 'Serra Circular', 'Parafusadeira Elétrica', 'Cortador de Grama', 
    'Soprador de Folhas', 'Lavadora de Alta Pressão', 'Mangueira de Jardim', 'Tesouras de Jardim', 'Aspersor Inteligente', 'Bicicleta Ergométrica', 'Esteira', 
    'Tapete de Yoga', 'Conjunto de Halteres', 'Kettlebell', 'Faixas de Resistência', 'Rolo de Espuma', 'Banco de Peso', 'Corda de Pular', 
    'Barra de Tração', 'Tênis de Corrida', 'Botas de Caminhada', 'Barraca', 'Saco de Dormir', 'Fogareiro de Camping', 'Mochila', 'Lanterna', 
    'Binóculos', 'Vara de Pesca', 'Cooler', 'Caiaque Inflável', 'Prancha de Stand Up Paddle', 'Kit de Mergulho', 'Roupa de Neoprene', 'Colete Salva-vidas', 
    'Manta para Piquenique', 'Grelha Portátil', 'Fogueira Portátil', 'Alto-falante Externo', 'Rede', 'Carregador Solar', 'Garrafa de Água', 
    'Caneca de Viagem', 'Conjunto de Malas', 'Cubos Organizadores', 'Travesseiro de Viagem', 'Adaptador de Viagem', 'Hotspot WiFi Portátil', 
    'Protetores Auriculares com Cancelamento de Ruído', 'Mochila de Viagem', 'Câmera de Ação', 'Drone', 'Câmera Digital', 'Lente de Câmera', 
    'Tripé', 'Cartão de Memória', 'Bolsa para Câmera', 'Impressora de Fotos'
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
