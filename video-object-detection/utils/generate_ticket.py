from datetime import datetime

def ticket(shopping_list, video = False):
    shopping_ticket = open("shopping_ticket.txt", "w")

    prices = {"Brick of gazpacho": 3.23, "Can of olives": 1.85, "Chocolate bar": 2.25,  "Coke": 0.8,  "Potato chips": 3.25,  "Tuna": 2.9,  "Yogur": 0.95}

    final_charge = 0

    print('\n\t\tAlba & Ambar\'s supermarket')
    shopping_ticket.write('\n\t\tAlba & Ambar\'s supermarket\n\n')

    now = datetime.now()
    print('Date:' + str(now.date()) + '\t\t\t\tTime:' + str(now.hour) + ':' + str(now.minute) + ':' + str(now.second))
    shopping_ticket.write('Date:' + str(now.date()) + '\t\t\t\tTime:' + str(now.hour) + ':' + str(now.minute) + ':' + str(now.second) + '\n\n')

    print('Product\t\t\tPrice\t    Units\tTotal')
    shopping_ticket.write('Product\t\t       Price\t   Units\tTotal\n')

    for product, price in prices.items():

        n_units = shopping_list.count(product)
        
        # Treatment for videos (depends on the speed of the video)
        # With the camera used and the speed of movement, an object is seen for approximately 100 frames
        if video:
            n_units = round(n_units/100)

        if n_units == 0:
            continue

        total = n_units * price

        print(f'{product:20}{price:8.2f} €{n_units:9}{total:12.2f} €')
        shopping_ticket.write(f'{product:20}{price:8.2f} €{n_units:9}{total:12.2f} €\n')
        final_charge += total

    print('_____________________________________________________')
    shopping_ticket.write('_____________________________________________________\n')

    print(f'\nTotal price{final_charge:40.2f} €')
    shopping_ticket.write(f'\nTotal price{final_charge:40.2f} €')

    shopping_ticket.close()

