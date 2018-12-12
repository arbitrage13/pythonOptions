import sqlite3
import databaseFunc

connection = sqlite3.connect('optionsBook.db')

c = connection.cursor()

#c.execute('''DROP TABLE optionsBook''')
#connection.commit()

#create table
c.execute("""CREATE TABLE IF NOT EXISTS optionsBook (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    date text,
    symbol text,
    ordertype text,
    side text,
    amount int,
    price real
)""")

##############################

databaseFunc.insertOption('2018-09-05','s50u18c1125','l','o',10,20)

#purchases = [('2018-09-05','s50u18c1125','l','o',10,20),('2018-09-05','s50u18c1125','l','o',10,30)]
#insertMany(purchases)
#viewOption('s50u18c1125')
#c.connection("SELECT * FROM optionsbook")
#print(c.fetchall())

