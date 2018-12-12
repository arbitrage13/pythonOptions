class option:
    def __init__(self,date,symbol,ordertype, side, amount, price):
        self.date = date
        self.symbol = symbol
        

def insertOption(date, symbol, ordertype, side, amount, price):
    ID = None
    with connection:
        c.execute("""INSERT INTO optionsBook VALUES (?,?,?,?,?,?,?)""",(ID, date, symbol, ordertype, side, amount, price))
    connection.commit()
    print('Data was inserted')
    

def viewOption(symbol):
    with connection:
        c.execute("SELECT * FROM optionsBook WHERE symbol =?",(symbol,))
        
        option = c.fetchall()
        for row in option:
            print(row)
    

def delOption(ID):
    with connection:
        c.execute("DELETE FROM optionsBook WHERE ID = (?)", [(ID)])
    connection.commit()
    print('Data was delete')

def updateOption(ID,priceChange):
    with connection:
        c.execute("UPDATE optionsBook SET price = (?) WHERE ID = (?)", ([priceChange, ID]))
    connection.commit()
    print('Date was updated')
