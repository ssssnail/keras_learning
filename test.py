def preprocess_data(dataset):
    data = [{'time': item['time'],
             'Sex': item['Sex'],
             'name': item['name'],
             'City': item['City']}
            for item in dataset]

    return data

pre_data = preprocess_data('D:\RNN\\unix.csv')