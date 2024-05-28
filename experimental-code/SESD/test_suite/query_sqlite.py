import sqlite3

conn = sqlite3.connect('./restaurants/restaurants-db.added-in-2020.sqlite')
cursor = conn.cursor()
cursor.execute('select locationalias0.* , restaurantalias0.name from location as locationalias0 join restaurant as restaurantalias0 on locationalias0.restaurant_id = restaurantalias0.restaurant_id where locationalias0.city_name = \"san francisco\" and restaurantalias0.name = \"jamerican cuisine\"')
print(cursor.fetchall())

conn.close()