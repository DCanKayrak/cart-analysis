import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Veri setlerini okuyun
df_customers = pd.read_csv('data/olist_customers_dataset.csv')
df_order_items = pd.read_csv('data/olist_order_items_dataset.csv')
df_orders = pd.read_csv('data/olist_orders_dataset.csv')
df_products = pd.read_csv('data/olist_products_dataset.csv')

# Gerekli veri birleştirmelerini yapın
df_customers_orders = df_customers.merge(df_orders, on='customer_id')
orders_order_items = df_orders.merge(df_order_items, on='order_id')

# Üst müşteri ID'lerini bulun
top_customer_ids = orders_order_items['customer_id'].value_counts().head(10).index

# Tüm müşteriler için önerileri alacak bir fonksiyon oluşturun
def get_all_recommendations(cosine_sim, orders_data, top_customer_ids):
    all_recommendations = {}
    for customer_id in top_customer_ids:
        if customer_id == user_id:
            recommendations = get_recommendations(customer_id, cosine_sim, orders_data)
            all_recommendations[customer_id] = recommendations
            break
    return all_recommendations

# Öneri sistemi oluşturmak için gerekli işlemleri gerçekleştirin
def create_recommendations():
    # Müşteri ve ürün verilerini birleştirin
    orders_order_items = df_orders.merge(df_order_items, on='order_id')
    
    # Üst müşteri ID'lerini filtreleyin
    top_customer_data = orders_order_items[orders_order_items['customer_id'].isin(top_customer_ids)]
    
    # Ürünleri birleştirin
    orders_data = top_customer_data.groupby('customer_id')['product_id'].apply(list).reset_index()
    
    # TfidfVectorizer kullanarak ürün özellik vektörlerini oluşturun
    vectorizer = TfidfVectorizer(lowercase=False)
    orders_data['product_id_str'] = orders_data['product_id'].apply(lambda x: ' '.join(x))
    tfidf_matrix = vectorizer.fit_transform(orders_data['product_id_str'])
    
    # Cosine similarity kullanarak müşteriler arasındaki benzerliği hesaplayın
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Tüm müşteriler için önerileri alın
    all_recommendations = get_all_recommendations(cosine_sim, orders_data, top_customer_ids)
    
    return all_recommendations

# Müşteriye öneri yapmak için bir fonksiyon tanımlayın
def get_recommendations(customer_id, cosine_sim, orders_data):
    if customer_id in orders_data['customer_id'].values:
        idx = orders_data[orders_data['customer_id'] == customer_id].index[0]
        
        # Cosine benzerlik skorunu alın
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Benzerlik skorlarına göre sıralama yapın
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Kendi dışındaki en benzer 5 müşteriyi alın
        similar_customers_indices = [i[0] for i in sim_scores[1:6]]
        
        # Benzer müşterilerin aldığı ürünleri önerin
        recommendations = orders_data.iloc[similar_customers_indices]['product_id']
        
        return recommendations.tolist()
    else:
        return "Müşteri bulunamadı."

# Kullanıcı ID'sini belirtin
user_id = 'fc3d1daec319d62d49bfb5e1f83123e9'  # Örnek kullanıcı ID'si

# Önerileri oluşturun
all_recommendations = create_recommendations()

# Önerileri gösterin
for customer_id, recommendations in all_recommendations.items():
    print("Müşteri ID:", customer_id)
    print("Önerilen Ürünler:")
    print(recommendations)
    print()