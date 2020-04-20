def _split_by_class(x, y):
    data_per_class = []
    for _ in range(7):         
        data_per_class.append([])

    for i in range(len(x)):
        data_per_class[int(y[i])].append(x[i, 1:].tolist() + y[i].tolist())


def RUV(x, y):
    pass
