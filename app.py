from flask import Flask
import ai

app = Flask(__name__)

@app.route('/')
def ana_sayfa():
    ai.user_id = 'daf15f1b940cc6a72ba558f093dc00dd'
    return ai.create_recommendations()

@app.route('/products')
def products():
    return ai.df_order_items['product_id'].to_list()

if __name__ == '__main__':
    app.run(debug=True)
