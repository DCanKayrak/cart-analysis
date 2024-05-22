import pandas as pd

# Veri setlerini okuyun
df_customers = pd.read_csv('data/olist_customers_dataset.csv')
df_products = pd.read_csv('data/olist_products_dataset.csv')
df_order_items = pd.read_csv('data/olist_order_items_dataset.csv')
df_orders = pd.read_csv('data/olist_orders_dataset.csv')

# Gerekli veri birleştirmelerini yapın
orders_order_items = df_orders.merge(df_order_items, on='order_id').head(100)

def get_recommend(user_id):
    products = orders_order_items[orders_order_items['customer_id'] == user_id]['product_id']

    # Diğer aynı ürünleri alan kullanıcıları bul.
    similar_users = get_similar_users(products[0])

def get_similar_users(product_id):
    users = orders_order_items[orders_order_items['product_id'] == product_id]['customer_id'].unique()
    return users

from sklearn.neighbors import NearestNeighbors

# Veri birleştirmesi yapıldıktan sonra gerekli veri işlemleri

# Kullanıcıların aldığı ürünleri temsil eden bir özellik vektörü oluşturma
user_product_matrix = orders_order_items.pivot_table(index='customer_id', columns='product_id', aggfunc='size', fill_value=0)

# KNN modelini oluşturma
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(user_product_matrix)

# Yeni bir kullanıcı için önerileri alma
def get_recommendations(user_id, k=5):
    user_index = user_product_matrix.index.get_loc(user_id)
    distances, indices = knn_model.kneighbors([user_product_matrix.iloc[user_index]], n_neighbors=k+1)
    # İlk sonuç, kullanıcının kendisi olacağı için onu atlıyoruz
    indices = indices.flatten()[1:]
    similar_users = user_product_matrix.iloc[indices]
    # Benzer kullanıcıların aldığı ürünleri öneri olarak döndür
    recommended_products = similar_users.sum().sort_values(ascending=False).head(10)
    recommended_product_ids = recommended_products.index.tolist()
    print(recommended_product_ids)

    print(recommended_products)
    return recommended_product_ids